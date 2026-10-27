import cv2
import argparse
import os
import shutil
import subprocess
import sys
from random import randrange

FRAME_DIR = 'frames'
os.makedirs(FRAME_DIR, exist_ok=True)

# Clear out frames from the previous run
for filename in os.listdir(FRAME_DIR):
    file_path = os.path.join(FRAME_DIR, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="optional path to video file")
args = vars(ap.parse_args())


def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    if not vidObj.isOpened():
        raise SystemExit(f"Could not open video: {path!r}")

    count = 0
    while count <= 25:
        success, image = vidObj.read()
        if not success:
            break  # end of video, or a bad read
        cv2.imwrite(os.path.join(FRAME_DIR, f"frame{count}.jpg"), image)
        count += 1

    vidObj.release()
    return count  # number of frames actually saved


# Driver Code
if __name__ == '__main__':
    saved = FrameCapture(args["input"])
    if saved == 0:
        raise SystemExit("No frames were read from the video.")

    # Pick only from frames that were actually saved
    inputImage = os.path.join(FRAME_DIR, f"frame{randrange(saved)}.jpg")

    subprocess.call([sys.executable, "classify_image.py", "-i", inputImage])

    subprocess.call([
        sys.executable, "human_activity_reco.py",
        "--model", "resnet-34_kinetics.onnx",
        "--classes", "action_recognition_kinetics.txt",
        "--input", args["input"],
    ])

    subprocess.call([sys.executable, "Reportgenearation.py"])
