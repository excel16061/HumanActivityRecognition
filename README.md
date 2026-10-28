# HumanActivityRecognition

Extracts frames from a video, classifies one of them with a pretrained image model, runs human activity recognition on the video, and generates a report.

## Requirements

Python 3.12 is required for TensorFlow. Python 3.15 is too new for TensorFlow to install.

| Package | Import name | Used by |
| --- | --- | --- |
| opencv-python | cv2 | mainRun.py (frame capture) |
| numpy | numpy | OpenCV dependency |
| tensorflow | tensorflow | classify_image.py (Keras) |
| imutils | imutils | human_activity_reco.py |
| scikit-learn | sklearn | Reportgenearation.py |
| pillow | PIL | Keras image loading in classify_image.py |
| seaborn | seaborn | Reportgenearation.py |

Install into a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorflow imutils scikit-learn opencv-python numpy pillow seaborn
```

To record exact versions:

```bash
python -m pip freeze > requirements.txt
```

## Model files

The following must be in the project folder:

- `resnet-34_kinetics.onnx`
- `action_recognition_kinetics.txt`

## Usage

Run with a video file. A still image will not work with `human_activity_reco.py`.

```bash
python mainRun.py --input test1.mp4
```

`mainRun.py` does the following:

1. Clears the `frames` folder from the previous run.
2. Saves up to 26 frames from the video to `frames/`.
3. Runs `classify_image.py` on one randomly chosen saved frame.
4. Runs `human_activity_reco.py` on the video.
5. Runs `Reportgenearation.py`.

## Changes from the 2022 version

- Added the missing `shutil` import.
- Frame capture stops at the first failed read and checks that the video opened.
- The random frame is chosen only from frames that were saved.
- Helper scripts run as argument lists using `sys.executable`, so they use the same virtual environment.

## Troubleshooting

- **`No module named 'cv2'`**: install `opencv-python` into the active environment.
- **`No matching distribution found for tensorflow`**: the Python version is too new. Use Python 3.12.
- **`Could not open video`**: check the file name and extension with `ls`.
- **`no frame read from stream`**: `--input` must be a video file, not an image.
- **`No module named 'PIL'`**: install `pillow`.
- **`No module named 'seaborn'`**: install `seaborn`.
