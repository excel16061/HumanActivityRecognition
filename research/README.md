# Research evaluation upgrade (work in progress)

This folder separates reproducible evaluation from the original video demonstration.

## Important validity warning

The original `Reportgenearation.py` trains on scikit-learn's synthetic `make_circles` dataset. Its reported metrics **do not measure human activity recognition**. Do not cite them as HAR results. The previously advertised 96.95% accuracy is **unverified** by the current repository; do not present it as reproduced.

The existing `human_activity_reco.py` performs inference with a pretrained ResNet-34 Kinetics ONNX model on 16-frame windows. The separate `classify_image.py` performs ImageNet still-image classification and is not an HAR evaluation.

## Evaluate held-out video predictions

Install: `python -m pip install -r research/requirements.txt`

Prepare a CSV of **real predictions** from a held-out labeled dataset:

```csv
video_id,true_label,predicted_label
```

This is a schema illustration only; do not fabricate rows.

Run:
```bash
python research/evaluate_predictions.py --predictions path/to/held_out_predictions.csv
```

Output: `results/evaluation/metrics.json` with accuracy, macro precision/recall/F1, per-class scores and confusion matrix. The script refuses empty files, missing labels and duplicate video IDs.

## Before publishing metrics

1. Document dataset source, license, class mapping and official split.
2. Record model weights provenance and verify label compatibility with the dataset.
3. Generate one prediction per held-out video using a deterministic clip aggregation rule; do not use a random training/test split of overlapping frames.
4. Record hardware, package versions, random seeds, preprocessing, and model checkpoint.
5. Add simple baselines and repeat across official splits.
6. Run actual experiments; no accuracy or latency claims are implied by this scaffold.

**Current status:** Evaluation utility implemented; dataset adapter, video inference export, baseline training, tests and benchmarking are still pending.
