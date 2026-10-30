# Model card — provenance verification required

**Model file:** `resnet-34_kinetics.onnx`

**Current usage:** 16 consecutive frames per clip, resized to 112 × 112, OpenCV DNN preprocessing with mean subtraction and RGB swap. Inference averages raw model outputs across non-overlapping complete clips.

**Training data:** Kinetics is suggested by the filename, but the exact training dataset version, checkpoint author, architecture implementation, training procedure, license, and weight download URL are **not yet verified**.

**Label file:** `action_recognition_kinetics.txt`. Label ordering must match checkpoint output.

**Intended use:** Research demonstration and evaluation only.

**Limitations:** No published held-out dataset results in this branch. Clip-level aggregation has not been validated against a reference implementation. Videos with fewer than 16 readable frames are rejected. The exporter uses only complete non-overlapping clips; trailing frames are dropped.

**Cross-dataset caution:** UCF101 class names and Kinetics class names are not automatically equivalent. Do not evaluate on UCF101 without a documented, validated mapping or a model trained for its classes.

**Claim status:** The previously stated 96.95% accuracy is not reproduced or substantiated by the files reviewed.
