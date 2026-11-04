# HumanActivityRecognition

Video activity recognition using the ResNet-34 Kinetics ONNX model through OpenCV DNN.

## Setup

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The video pipeline needs OpenCV and NumPy. TensorFlow is not required.

## Model

| Setting | Value |
| --- | --- |
| Model file | `resnet-34_kinetics.onnx` |
| Labels | `action_recognition_kinetics.txt` |
| Inference engine | OpenCV DNN |
| Clip length | 16 consecutive frames |
| Spatial input | 112 × 112 pixels |
| Tensor layout | N × C × T × H × W (1 × 3 × 16 × 112 × 112) |
| Preprocessing | Scale 1.0, mean (114.7748, 107.7354, 99.4750), swapRB=True, crop=True |

Place the ONNX file and matching labels in the project folder. The launcher uses these names by default. Model weights are separate from Python packages; installing requirements does not download them. The filename identifies the intended model; the code checks that its output count matches the labels, but does not verify checkpoint provenance.

## Run

```bash
python mainRun.py --input test1.mp4
```

The video script also works directly with the same defaults:

```bash
python human_activity_reco.py --input test1.mp4
```

To specify files explicitly:

```bash
python mainRun.py \
  --model resnet-34_kinetics.onnx \
  --classes action_recognition_kinetics.txt \
  --input test1.mp4
```

Omit `--input` to use camera 0. Press **q** in the video window or **Ctrl+C** in the terminal to stop. Ctrl+Z suspends the process.

Predictions are printed per clip and displayed over the frames. A final clip with fewer than 16 frames is skipped; a source with no complete clip returns an error. Model and label defaults resolve relative to the scripts; an explicit input path resolves relative to your current directory.

## Legacy scripts

`classify_image.py` is a separate TensorFlow/Keras ImageNet still-image classifier, with VGG16 as its default. It is not the video activity model and is no longer launched by `mainRun.py`.

`Reportgenearation.py` trains a demonstration network on synthetic circles. Its metrics do not measure HAR performance. It is no longer launched automatically. Both legacy scripts retain their own dependencies and may need separate compatibility fixes.

## Troubleshooting

- **Missing cv2 or numpy:** activate the environment and run `python -m pip install -r requirements.txt`.
- **Model file not found:** place `resnet-34_kinetics.onnx` in the project folder or pass `--model /path/to/model.onnx`.
- **Could not open video source:** check that the video is readable; for a camera, check macOS camera permission.
- **No complete 16-frame clip:** use a readable video containing at least 16 frames.
- **Model/label output mismatch:** use the labels supplied for that checkpoint, in their original order.
