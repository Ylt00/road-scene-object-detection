from pathlib import Path
import csv
import json
import tempfile
import unittest

import cv2
import numpy as np

from road_scene.visualize import (
    create_contact_sheet,
    draw_yolo_labels,
    plot_overall_metrics,
    plot_training_curves,
)


def _write_image(path: Path, value: int = 128) -> None:
    image = np.full((96, 128, 3), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode test image")
    path.write_bytes(encoded.tobytes())


class VisualizationTests(unittest.TestCase):
    def test_draw_yolo_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "image.jpg"
            label_path = root / "image.txt"
            output_path = root / "annotated.jpg"
            _write_image(image_path)
            label_path.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

            result = draw_yolo_labels(image_path, label_path, output_path)
            self.assertTrue(result.exists())
            self.assertGreater(result.stat().st_size, 0)

    def test_create_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            output = root / "grid.jpg"
            _write_image(first, 100)
            _write_image(second, 180)

            result = create_contact_sheet([first, second], output, columns=2)
            image = cv2.imdecode(np.frombuffer(result.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[1], 840)

    def test_metric_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metrics_path = root / "metrics.json"
            results_path = root / "results.csv"
            metrics_path.write_text(
                json.dumps(
                    {
                        "metrics/precision(B)": 0.6,
                        "metrics/recall(B)": 0.4,
                        "metrics/mAP50(B)": 0.3,
                        "metrics/mAP50-95(B)": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            with results_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=[
                        "epoch",
                        "metrics/precision(B)",
                        "metrics/recall(B)",
                        "metrics/mAP50(B)",
                        "metrics/mAP50-95(B)",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "metrics/precision(B)": 0.6,
                        "metrics/recall(B)": 0.4,
                        "metrics/mAP50(B)": 0.3,
                        "metrics/mAP50-95(B)": 0.2,
                    }
                )

            self.assertTrue(plot_overall_metrics(metrics_path, root / "metrics.png").exists())
            self.assertTrue(plot_training_curves(results_path, root / "curves.png").exists())


if __name__ == "__main__":
    unittest.main()