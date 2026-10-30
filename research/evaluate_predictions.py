"""Evaluate real labeled video predictions, not synthetic examples.

CSV schema: video_id,true_label,predicted_label
Each row must represent one held-out video, with labels from a documented dataset.
"""
import argparse
import csv
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", default="results/evaluation", type=Path)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video_id", "true_label", "predicted_label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Nonempty CSV requires video_id,true_label,predicted_label columns")
    ids = [row["video_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video_id: evaluation must contain one row per video")
    if any(not all(row.get(k, "").strip() for k in required) for row in rows):
        raise ValueError("Missing video ID or label")
    truth = [r["true_label"] for r in rows]
    predicted = [r["predicted_label"] for r in rows]
    labels = sorted(set(truth) | set(predicted))
    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    output = {"num_videos": len(rows), "accuracy": accuracy_score(truth, predicted),
              "macro_f1": report["macro avg"]["f1-score"],
              "macro_precision": report["macro avg"]["precision"],
              "macro_recall": report["macro avg"]["recall"],
              "labels": labels, "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
              "per_class": report}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("num_videos", "accuracy", "macro_f1")}, indent=2))

if __name__ == "__main__":
    main()
