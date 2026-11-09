# Collecting figures and measurements for the report

Run `collect_results.py` **on the server, from the Lab4-FlowMatching project root**.
It only needs Python's standard library. It copies selected images into
`report/results/`, writes `metrics.csv` and `manifest.json`, and lists missing
inputs in `report/results/README.md`. It does not run training or calculate FID.

```bash
python report/collect_results.py \
  --task1-notebook task1_2d_flow_matching/fm_tutorial.ipynb \
  --task2-run results/cfg_fm-YOUR_RUN \
  --task3-run results/rectified_fm_1-YOUR_RUN \
  --task4-run results/instaflow-YOUR_RUN \
  --task3-eval results/evaluation \
  --task4-eval results/instaflow_eval \
  --sample-dir task2/cfg7.5_steps20=results/YOUR_TASK2_SAMPLES \
  --max-samples 8
```

The training run options select the correct experiment if several runs exist.
If omitted, the script chooses the most recently modified `cfg_fm-*`,
`rectified_fm_1-*`, and `instaflow-*` directories. For evaluation, it reads
`evaluation_results.json` and copies the sample directories named there.
Add one `--sample-dir TASK/LABEL=DIR` for each extra image condition, such as
Task 2 CFG scales or inference step counts. This must point to a dedicated
inference output directory with numbered images (`0.png`, `1.png`, ...), not a
training run directory containing `loss.png` and `step=...png`. On rerun, the
script removes only files it previously copied for the same sample label.
The notebook must have saved cell
outputs to contribute Task 1 figures and Chamfer Distance.

The FID program prints its result but does not save it. Save each measurement
on the server, then pass the log with the matching sample label:

```bash
python -m fid.measure_fid data/afhq/eval results/evaluation/rectified_1_steps5 \
  | tee results/fid_rectified_1_steps5.txt

python report/collect_results.py \
  --task3-eval results/evaluation \
  --fid-log task3/rectified_1_steps5=results/fid_rectified_1_steps5.txt
```

For Task 4, a typical label is `task4/instaflow_1step`. For manually supplied
Task 2 samples, use the same label in `--sample-dir` and `--fid-log`. Without a
log, FID stays blank. The report bundle contains selected PNGs, while
`image_count` records the full number of images in the source directory.
Check `manifest.json` to see exactly which server run each file came from.
