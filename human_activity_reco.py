"""Classify 16-frame video clips using OpenCV DNN and ResNet-34 ONNX."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", default=str(ROOT / "resnet-34_kinetics.onnx"),
                        help="ONNX model path (default: project ResNet-34 model)")
    parser.add_argument("-c", "--classes", default=str(ROOT / "action_recognition_kinetics.txt"),
                        help="class labels path (default: project Kinetics labels)")
    parser.add_argument("-i", "--input", default="",
                        help="video path; omit to use camera 0")
    args = parser.parse_args()
    for name in ("model", "classes"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            parser.error(f"{name} file not found: {path}")
        setattr(args, name, str(path))
    if args.input:
        source = Path(args.input).expanduser()
        if not source.is_file():
            parser.error(f"video file not found: {source}")
        args.input = str(source)

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Run: "
                       "python -m pip install -r requirements.txt\n")

    classes = Path(args.classes).read_text(encoding="utf-8").splitlines()
    classes = [label.strip() for label in classes if label.strip()]
    if not classes:
        parser.error("class labels file is empty")

    vs = None
    try:
        print(f"[INFO] loading model: {args.model}", flush=True)
        net = cv2.dnn.readNetFromONNX(args.model)
        print("[INFO] accessing video stream...", flush=True)
        vs = cv2.VideoCapture(args.input if args.input else 0)
        if not vs.isOpened():
            raise ValueError(f"Could not open video source: {args.input or 'camera 0'}")
        clips = 0
        while True:
            frames = []
            for _ in range(16):
                grabbed, frame = vs.read()
                if not grabbed:
                    break
                height, width = frame.shape[:2]
                frames.append(cv2.resize(frame, (400, max(1, round(height * 400 / width)))))
            if len(frames) < 16:
                if clips == 0:
                    raise ValueError("No complete 16-frame clip could be read from the source.")
                print("[INFO] end of stream; incomplete final clip skipped")
                return 0
            blob = cv2.dnn.blobFromImages(
                frames, 1.0, (112, 112), (114.7748, 107.7354, 99.4750),
                swapRB=True, crop=True)
            blob = np.expand_dims(np.transpose(blob, (1, 0, 2, 3)), axis=0)
            net.setInput(blob)
            outputs = net.forward().reshape(-1)
            if outputs.size != len(classes):
                raise ValueError(f"Model has {outputs.size} outputs, but labels file "
                                 f"has {len(classes)} entries. Use matching model and labels.")
            label = classes[int(np.argmax(outputs))]
            clips += 1
            print(f"[INFO] clip {clips}: {label}", flush=True)
            for frame in frames:
                cv2.rectangle(frame, (0, 0), (400, 40), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                cv2.imshow("Activity Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 0
    except KeyboardInterrupt:
        return 130
    except (cv2.error, ValueError) as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1
    finally:
        if vs is not None:
            vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
