#!/usr/bin/env python3
"""Local benchmark for ResNet-34 Kinetics video recognition.

Produces sampled per-clip predictions and timing; does not claim accuracy.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

MODES = {'fast': 5.0, 'balanced': 2.0, 'accurate': None}
MEAN = (114.7748, 107.7354, 99.4750)


def clip_blob(frames):
    resized = []
    for frame in frames:
        height, width = frame.shape[:2]
        if not width or not height:
            raise ValueError('Invalid frame dimensions')
        resized.append(cv2.resize(frame, (400, max(1, round(height * 400 / width)))))
    blob = cv2.dnn.blobFromImages(resized, 1.0, (112, 112), MEAN,
                                  swapRB=True, crop=True)
    return np.ascontiguousarray(np.expand_dims(np.transpose(blob, (1, 0, 2, 3)), 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--mode', choices=MODES, default='fast')
    parser.add_argument('--model', type=Path, default=Path('resnet-34_kinetics.onnx'))
    parser.add_argument('--classes', type=Path, default=Path('action_recognition_kinetics.txt'))
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    if args.threads < 1:
        parser.error('--threads must be >= 1')
    for path in (args.input, args.model, args.classes):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    labels = [line.strip() for line in args.classes.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not labels:
        parser.error('Class label file is empty')

    cv2.setNumThreads(args.threads)
    net = cv2.dnn.readNet(str(args.model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        parser.error(f'Cannot open video: {args.input}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        parser.error('Video FPS is missing or invalid')

    frame_step = 1 if MODES[args.mode] is None else max(1, round(MODES[args.mode] * fps))
    frame_index = 0
    next_start = 0
    clips = []
    clip_frames = []
    clip_start = None
    inference_ms = []
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not clip_frames:
                if frame_index < next_start:
                    frame_index += 1
                    continue
                clip_start = frame_index
            clip_frames.append(frame)
            if len(clip_frames) == 16:
                blob = clip_blob(clip_frames)
                net.setInput(blob)
                t0 = time.perf_counter()
                scores = np.asarray(net.forward()).reshape(-1)
                elapsed = (time.perf_counter() - t0) * 1000
                if len(scores) != len(labels):
                    raise ValueError(f'Model returned {len(scores)} scores but label file has {len(labels)} classes')
                if not np.all(np.isfinite(scores)):
                    raise ValueError('Model returned non-finite scores')
                prediction = labels[int(np.argmax(scores))]
                clips.append({'start_seconds': round(clip_start / fps, 3),
                              'label': prediction, 'inference_ms': round(elapsed, 2)})
                inference_ms.append(elapsed)
                print(f'{clip_start / fps:7.1f}s | {prediction} | {elapsed:.1f} ms', flush=True)
                clip_frames.clear()
                next_start = clip_start + frame_step if MODES[args.mode] is not None else frame_index + 1
            frame_index += 1
    finally:
        cap.release()
    wall = time.perf_counter() - start
    if not clips:
        raise RuntimeError('No complete 16-frame clip could be processed')
    output = args.output or Path('results') / f'{args.input.stem}_{args.mode}.json'
    report = {
        'input': str(args.input), 'mode': args.mode, 'model': str(args.model),
        'threads': args.threads, 'fps': fps, 'frames_read': frame_index,
        'clips_processed': len(clips), 'wall_time_seconds': round(wall, 3),
        'total_model_inference_seconds': round(sum(inference_ms) / 1000, 3),
        'median_inference_ms': round(float(np.median(inference_ms)), 3),
        'mean_inference_ms': round(float(np.mean(inference_ms)), 3),
        'clips': clips,
        'notes': 'Predictions are not validated accuracy; sparse modes may miss brief actions. Inference time excludes decoding and preprocessing.'
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nDone: {len(clips)} clips in {wall:.2f}s; median inference {report["median_inference_ms"]}ms')
    print(f'Report: {output}')


if __name__ == '__main__':
    main()
