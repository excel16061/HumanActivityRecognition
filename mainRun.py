"""Run video activity recognition with the project's ONNX model."""
import subprocess
import sys
from pathlib import Path


def main():
    script = Path(__file__).resolve().with_name("human_activity_reco.py")
    try:
        return subprocess.call([sys.executable, str(script), *sys.argv[1:]])
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
