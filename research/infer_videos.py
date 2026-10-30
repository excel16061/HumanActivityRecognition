"""Export deterministic per-video predictions from the existing 16-frame ONNX model.

Manifest CSV columns: video_id,video_path,true_label
All true labels must use the same vocabulary as the ONNX checkpoint.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

def predict_video(net, path, labels):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    scores = []
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (400, round(frame.shape[0] * 400 / frame.shape[1]))))
            if len(frames) == 16:
                blob = cv2.dnn.blobFromImages(frames, 1.0, (112, 112),
                        (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
                blob = np.expand_dims(np.transpose(blob, (1, 0, 2, 3)), 0)
                net.setInput(blob)
                scores.append(np.asarray(net.forward()).reshape(-1))
                frames = []
    finally:
        cap.release()
    if not scores:
        raise ValueError(f"Video has fewer than 16 readable frames: {path}")
    if any(len(s) != len(labels) for s in scores):
        raise ValueError("Model output dimension does not match label count")
    return labels[int(np.argmax(np.mean(scores, axis=0)))], len(scores)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--classes", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/predictions.csv"))
    a = p.parse_args()
    labels = [s.strip() for s in a.classes.read_text().splitlines() if s.strip()]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Class labels must be nonempty and unique")
    net = cv2.dnn.readNet(str(a.model))
    with a.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"video_id", "video_path", "true_label"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires video_id,video_path,true_label")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate video IDs")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id","true_label","predicted_label","num_clips"])
        writer.writeheader()
        for row in rows:
            if row["true_label"] not in labels:
                raise ValueError(f"Ground-truth label not in model vocabulary: {row['true_label']}")
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = a.manifest.parent / video_path
            pred, n = predict_video(net, video_path, labels)
            writer.writerow({"video_id":row["video_id"],"true_label":row["true_label"],
                             "predicted_label":pred,"num_clips":n})
            print(f"{row['video_id']}: {pred} ({n} clips)")

if __name__ == "__main__":
    main()
