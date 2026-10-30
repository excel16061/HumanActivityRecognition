"""Small synthetic-label unit tests; not model-performance evidence."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("evaluate_predictions.py")

class EvaluationTests(unittest.TestCase):
    def test_known_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "predictions.csv"
            with source.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["video_id","true_label","predicted_label"])
                writer.writerows([["a","walking","walking"],["b","running","walking"],
                                  ["c","running","running"],["d","walking","walking"]])
            subprocess.run([sys.executable,str(SCRIPT),"--predictions",str(source),
                            "--output",str(base / "out")],check=True,capture_output=True,text=True)
            metrics = json.loads((base / "out" / "metrics.json").read_text())
            self.assertEqual(metrics["num_videos"],4)
            self.assertAlmostEqual(metrics["accuracy"],0.75)
            self.assertEqual(metrics["confusion_matrix"],[[1,1],[0,2]])

if __name__ == "__main__":
    unittest.main()
