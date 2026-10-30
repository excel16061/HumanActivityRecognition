# Proposed experimental protocol

**Research question:** What is the accuracy/efficiency trade-off for video activity recognition under constrained compute?

- Dataset candidate: UCF101, with official train/test splits and documented licensing.
- Existing video model: pretrained ResNet-34 Kinetics ONNX; confirm checkpoint provenance and Kinetics-to-UCF101 label mapping before comparing.
- Baselines: majority class; reproducible lightweight video baseline.
- Metrics: per-video top-1 accuracy, macro precision/recall/F1, per-class F1, confusion matrix.
- Efficiency: median and p95 latency per clip, peak memory, model bytes, measured on named hardware.
- Ablations: clip sampling, resolution and supported quantization, holding evaluation split fixed.
- Report exact environment, deterministic sampling policy, all split results, and limitations.

No results have been run or verified as part of this documentation.
