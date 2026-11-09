"""Collect existing Lab 4 report images and metrics from a server run.

Run this from the project root on the machine holding the results. It copies
representative images and recorded measurements; it does not calculate FID.
"""

import argparse
import base64
import csv
import json
import re
import shutil
from pathlib import Path


FIELDS = ("task", "model", "steps", "time_seconds", "samples_per_second",
          "speedup", "fid", "image_count", "source_dir", "copied_dir")
FID_RE = re.compile(r"\bFID\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
STEP_RE = re.compile(r"step=(\d+)-(\d+)\.png$")


def source_path(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def safe_label(label):
    path = Path(label)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"Unsafe output label: {label}")
    return path


def copy(source, destination, output, manifest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manifest["files"].append({"source": str(source),
                              "output": str(destination.relative_to(output))})


def images_in(directory):
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"))


def samples(source, destination, limit, output, manifest):
    if not source.is_dir():
        manifest["missing"].append(f"Sample directory: {source}")
        return 0
    images = images_in(source)
    for path in images[:limit]:
        copy(path, destination / path.name, output, manifest)
    if not images:
        manifest["missing"].append(f"Images in: {source}")
    return len(images)


def training(run, task, limit, output, manifest):
    if run is None or not run.is_dir():
        manifest["missing"].append(f"{task} training run")
        return
    manifest["sources"][f"{task}_run"] = str(run)
    target = output / task / "training"
    for name in ("loss.png", "config.json"):
        if (run / name).is_file():
            copy(run / name, target / name, output, manifest)
        else:
            manifest["missing"].append(f"{task} {name}: {run}")
    candidates = []
    for path in run.glob("step=*-*.png"):
        match = STEP_RE.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), int(match.group(2)), path))
    if candidates:
        last_step = max(step for step, _, _ in candidates)
        manifest["training_steps"][task] = last_step
        selected = sorted((item for item in candidates if item[0] == last_step),
                          key=lambda item: item[1])[:limit]
        for _, _, path in selected:
            copy(path, target / "samples" / path.name, output, manifest)
    else:
        manifest["missing"].append(f"{task} training sample PNGs: {run}")


def notebook_outputs(notebook, output, manifest):
    if notebook is None or not notebook.is_file():
        manifest["missing"].append("Task 1 executed notebook (--task1-notebook)")
        return
    manifest["sources"]["task1_notebook"] = str(notebook)
    for cell_index, cell in enumerate(json.loads(notebook.read_text()).get("cells", []), 1):
        image_index = 0
        for result in cell.get("outputs", []):
            encoded = result.get("data", {}).get("image/png")
            if encoded:
                image_index += 1
                destination = output / "task1" / f"cell_{cell_index:02d}_{image_index}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(base64.b64decode("".join(encoded)))
                manifest["files"].append({"source": f"{notebook}#cell-{cell_index}",
                                          "output": str(destination.relative_to(output))})
            result_text = "".join(result.get("text", []))
            match = re.search(r"Chamfer Distance\s*:\s*([-+]?\d+(?:\.\d+)?)", result_text)
            if match:
                manifest["task1_chamfer_distance"] = float(match.group(1))
    if not any(item["output"].startswith("task1/") for item in manifest["files"]):
        manifest["missing"].append("Task 1 notebook images")
    if "task1_chamfer_distance" not in manifest:
        manifest["missing"].append("Task 1 Chamfer Distance notebook output")


def fid_values(root, logs, manifest):
    values = {}
    for label, path in logs:
        safe_label(label)
        path = source_path(root, path)
        if not path.is_file():
            manifest["missing"].append(f"FID log: {path}")
            continue
        matches = FID_RE.findall(path.read_text(errors="replace"))
        if matches:
            values[label] = float(matches[-1])
            manifest["sources"][f"fid/{label}"] = str(path)
        else:
            manifest["missing"].append(f"FID value in log: {path}")
    return values


def evaluation(root, directory, task, limit, output, fids, manifest):
    if directory is None or not directory.is_dir():
        manifest["missing"].append(f"{task} evaluation directory")
        return
    result_file = directory / "evaluation_results.json"
    if not result_file.is_file():
        manifest["missing"].append(f"{task} evaluation_results.json: {directory}")
        return
    manifest["sources"][f"{task}_evaluation"] = str(directory)
    target = output / task
    results = json.loads(result_file.read_text())
    copy(result_file, target / result_file.name, output, manifest)
    plot = "sampling_time_comparison.png" if task == "task3" else "instaflow_comparison.png"
    if (directory / plot).is_file():
        copy(directory / plot, target / plot, output, manifest)
    else:
        manifest["missing"].append(f"{task} {plot}: {directory}")
    for model, entry in results.items():
        entries = entry.items() if task == "task3" else [(str(entry.get("steps", "")), entry)]
        for steps, data in entries:
            name = Path(data.get("save_dir", "")).name
            if not name:
                manifest["missing"].append(f"Sample path for {task} {model}")
                continue
            sibling = directory / name
            source = sibling if sibling.is_dir() else source_path(root, data["save_dir"])
            destination = target / safe_label(name)
            count = samples(source, destination, limit, output, manifest)
            label = f"{task}/{name}"
            fid = fids.get(label)
            if fid is None:
                manifest["missing"].append(f"FID for {label} (--fid-log {label}=FILE)")
            speedup = data.get("speedup", "")
            if task == "task3" and model != "Base FM":
                base = results.get("Base FM", {}).get(str(steps), {})
                if base.get("time") and data.get("time"):
                    speedup = base["time"] / data["time"]
            manifest["metrics"].append({
                "task": task, "model": model,
                "steps": int(steps) if str(steps).isdigit() else "",
                "time_seconds": data.get("time", ""),
                "samples_per_second": data.get("samples_per_sec", ""),
                "speedup": speedup, "fid": fid,
                "image_count": count, "source_dir": str(source),
                "copied_dir": str(destination.relative_to(output)),
            })


def write_indexes(output, manifest):
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in manifest["metrics"]:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "")
                             for field in FIELDS})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Lab 4 report results", "",
             "Copied sample PNGs are representative; image_count in metrics.csv is the full source count.",
             "FID appears only when a measured log was supplied.", ""]
    if "task1_chamfer_distance" in manifest:
        lines += [f"Task 1 Chamfer Distance: {manifest['task1_chamfer_distance']}", ""]
    lines += ["## Metrics", "", "See [metrics.csv](metrics.csv).", "", "## Files", ""]
    lines += [f"- [{item['output']}]({item['output']})" for item in manifest["files"]]
    lines += ["", "## Missing inputs or measurements", ""]
    lines += [f"- {item}" for item in manifest["missing"]] or ["- None"]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def collect_results(root, output, *, task1_notebook=None, task2_run=None,
                    task3_run=None, task4_run=None, task3_eval=None, task4_eval=None,
                    sample_dirs=(), fid_logs=(), max_samples=8):
    root, output = Path(root).resolve(), Path(output).resolve()
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {}, "files": [], "metrics": [], "training_steps": {}, "missing": []}
    fids = fid_values(root, fid_logs, manifest)
    notebook_outputs(source_path(root, task1_notebook) if task1_notebook else None,
                     output, manifest)
    for task, run in (("task2", task2_run), ("task3", task3_run), ("task4", task4_run)):
        training(source_path(root, run) if run else None, task, max_samples, output, manifest)
    for task, directory in (("task3", task3_eval), ("task4", task4_eval)):
        evaluation(root, source_path(root, directory) if directory else None,
                   task, max_samples, output, fids, manifest)
    for label, directory in sample_dirs:
        part = safe_label(label)
        if part.parts[0] not in ("task2", "task3", "task4"):
            raise ValueError(f"Sample label must start with task2/, task3/, or task4/: {label}")
        source = source_path(root, directory)
        count = samples(source, output / part, max_samples, output, manifest)
        fid = fids.get(label)
        if fid is None:
            manifest["missing"].append(f"FID for {label} (--fid-log {label}=FILE)")
        manifest["metrics"].append({
            "task": part.parts[0], "model": label, "steps": "", "time_seconds": "",
            "samples_per_second": "", "speedup": "", "fid": fid,
            "image_count": count, "source_dir": str(source), "copied_dir": str(part),
        })
    write_indexes(output, manifest)
    return manifest


def label_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    return label, path


def latest_run(root, pattern):
    candidates = [path for path in (root / "results").glob(pattern) if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root on server")
    parser.add_argument("--output", type=Path, help="Default: ROOT/report/results")
    parser.add_argument("--task1-notebook", type=Path, help="Executed fm_tutorial.ipynb with saved outputs")
    parser.add_argument("--task2-run", type=Path, help="Task 2 training run directory")
    parser.add_argument("--task3-run", type=Path, help="Task 3 training run directory")
    parser.add_argument("--task4-run", type=Path, help="Task 4 training run directory")
    parser.add_argument("--task3-eval", type=Path, help="Directory with Task 3 evaluation_results.json")
    parser.add_argument("--task4-eval", type=Path, help="Directory with Task 4 evaluation_results.json")
    parser.add_argument("--sample-dir", type=label_path, action="append", default=[], metavar="TASK/LABEL=DIR",
                        help="Extra samples, e.g. task2/cfg7.5_steps20=results/samples20")
    parser.add_argument("--fid-log", type=label_path, action="append", default=[], metavar="TASK/LABEL=FILE",
                        help="Saved measure_fid.py log matching a sample label")
    parser.add_argument("--max-samples", type=int, default=8, help="PNGs copied per condition")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "report" / "results"
    task2_run = args.task2_run or latest_run(root, "cfg_fm-*")
    task3_run = args.task3_run or latest_run(root, "rectified_fm_1-*")
    task4_run = args.task4_run or latest_run(root, "instaflow-*")
    task3_eval = args.task3_eval or root / "results" / "evaluation"
    task4_eval = args.task4_eval or next(
        (path for path in (root / "results" / "instaflow_eval",
                          root / "results" / "instaflow_evaluation") if path.is_dir()), None)
    manifest = collect_results(
        root, output, task1_notebook=args.task1_notebook,
        task2_run=task2_run, task3_run=task3_run, task4_run=task4_run,
        task3_eval=task3_eval, task4_eval=task4_eval,
        sample_dirs=args.sample_dir, fid_logs=args.fid_log,
        max_samples=args.max_samples,
    )
    print(f"Collected {len(manifest['files'])} files and {len(manifest['metrics'])} metric rows in {output}")
    print(f"See {output / 'README.md'} and {output / 'metrics.csv'}")
    if manifest["missing"]:
        print(f"Missing or unprovided items: {len(manifest['missing'])} (listed in README.md)")


if __name__ == "__main__":
    main()
