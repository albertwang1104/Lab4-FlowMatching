import json
import base64
import tempfile
import unittest
from pathlib import Path

from report.collect_results import collect_results


class CollectResultsTests(unittest.TestCase):
    def test_rejects_training_run_as_inference_sample_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results" / "cfg_fm-example"
            run.mkdir(parents=True)
            (run / "loss.png").write_bytes(b"loss")
            (run / "step=200-0.png").write_bytes(b"training sample")
            output = root / "report" / "results"

            with self.assertRaisesRegex(ValueError, "inference sample directory"):
                collect_results(root, output,
                                sample_dirs=[("task2/cfg7.5_steps20", run)])
            self.assertFalse(output.exists())

    def test_replaces_only_previously_collected_samples_on_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "results" / "task2_samples"
            source.mkdir(parents=True)
            (source / "0.png").write_bytes(b"new sample")
            (source / "loss.png").write_bytes(b"wrong image")
            output = root / "report" / "results"
            destination = output / "task2" / "cfg7.5_steps20"
            destination.mkdir(parents=True)
            (destination / "loss.png").write_bytes(b"old copied loss")
            (destination / "personal.png").write_bytes(b"user image")
            (output / "manifest.json").write_text(json.dumps({"files": [
                {"source": "old", "output": "task2/cfg7.5_steps20/loss.png"}
            ]}))

            collect_results(root, output,
                            sample_dirs=[("task2/cfg7.5_steps20", source)])

            self.assertEqual((destination / "0.png").read_bytes(), b"new sample")
            self.assertFalse((destination / "loss.png").exists())
            self.assertTrue((destination / "personal.png").exists())

    def test_extracts_saved_notebook_figure_and_chamfer_distance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = root / "fm_tutorial.ipynb"
            notebook.write_text(json.dumps({"cells": [{"outputs": [
                {"data": {"image/png": base64.b64encode(b"figure").decode()}},
                {"text": ["Chamfer Distance: 31.25\n"]},
            ]}]}))

            manifest = collect_results(root, root / "report" / "results",
                                       task1_notebook=notebook)

            self.assertEqual(manifest["task1_chamfer_distance"], 31.25)
            self.assertEqual((root / "report" / "results" / "task1" / "cell_01_1.png").read_bytes(),
                             b"figure")

    def test_collects_latest_training_samples_and_evaluation_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results" / "cfg_fm-example"
            run.mkdir(parents=True)
            (run / "loss.png").write_bytes(b"loss")
            for name in ("step=200-0.png", "step=400-0.png", "step=400-1.png"):
                (run / name).write_bytes(name.encode())

            evaluation = root / "results" / "evaluation"
            samples = evaluation / "base_fm_steps5"
            samples.mkdir(parents=True)
            for index in range(3):
                (samples / f"sample_{index:04d}.png").write_bytes(b"image")
            rectified_samples = evaluation / "rectified_1_steps5"
            rectified_samples.mkdir()
            (rectified_samples / "sample_0000.png").write_bytes(b"image")
            (evaluation / "evaluation_results.json").write_text(json.dumps({
                "Base FM": {"5": {"time": 12.0, "samples_per_sec": 25.0,
                                   "save_dir": "results/evaluation/base_fm_steps5"}},
                "1-Rectified Flow": {"5": {"time": 4.0, "samples_per_sec": 75.0,
                                            "save_dir": "results/evaluation/rectified_1_steps5"}},
            }))
            fid_log = root / "fid_base5.txt"
            fid_log.write_text("Calculating FID...\nFID: 29.75\n")

            manifest = collect_results(
                root, root / "report" / "results", task2_run=run,
                task3_eval=evaluation, fid_logs=[("task3/base_fm_steps5", fid_log)],
                max_samples=2,
            )

            output = root / "report" / "results"
            self.assertEqual((output / "task2" / "training" / "loss.png").read_bytes(), b"loss")
            self.assertEqual(sorted(p.name for p in (output / "task2" / "training" / "samples").glob("*.png")),
                             ["step=400-0.png", "step=400-1.png"])
            self.assertEqual(len(list((output / "task3" / "base_fm_steps5").glob("*.png"))), 2)
            self.assertEqual(manifest["metrics"][0]["fid"], 29.75)
            self.assertEqual(manifest["metrics"][0]["image_count"], 3)
            self.assertEqual(manifest["metrics"][1]["speedup"], 3.0)
            self.assertIn("29.75", (output / "metrics.csv").read_text())

    def test_missing_fid_is_blank_and_not_invented(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "results" / "instaflow_eval"
            samples = evaluation / "instaflow_1step"
            samples.mkdir(parents=True)
            (samples / "sample_0000.png").write_bytes(b"image")
            (evaluation / "evaluation_results.json").write_text(json.dumps({
                "InstaFlow (1 step)": {"steps": 1, "time": 2.5,
                                         "samples_per_sec": 20.0, "speedup": 8.0,
                                         "save_dir": "results/instaflow_eval/instaflow_1step"}
            }))

            manifest = collect_results(root, root / "report" / "results", task4_eval=evaluation)

            self.assertIsNone(manifest["metrics"][0]["fid"])
            self.assertIn("FID", " ".join(manifest["missing"]))


if __name__ == "__main__":
    unittest.main()
