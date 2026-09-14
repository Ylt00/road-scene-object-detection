from pathlib import Path
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

from road_scene.data import extract_zip_safely, prepare_road_dataset, remap_label_lines
from road_scene.dataset import validate_yolo_dataset


def _encoded_image() -> bytes:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode test image")
    return encoded.tobytes()


class DataRemappingTests(unittest.TestCase):
    def test_remap_selected_and_filtered_classes(self) -> None:
        lines = [
            "0 0.5 0.5 0.2 0.2",
            "14 0.4 0.4 0.1 0.1",
            "28 0.6 0.6 0.3 0.3",
        ]
        output, kept, filtered, errors = remap_label_lines(lines)
        self.assertEqual(errors, [])
        self.assertEqual(kept, 2)
        self.assertEqual(filtered, 1)
        self.assertIn("0 0.500000", output[0])
        self.assertIn("15 0.600000", output[1])

    def test_invalid_line_is_reported(self) -> None:
        output, kept, filtered, errors = remap_label_lines(["0 0.5 0.5"])
        self.assertEqual(output, [])
        self.assertEqual(kept, 0)
        self.assertEqual(filtered, 0)
        self.assertEqual(len(errors), 1)


class DatasetPreparationTests(unittest.TestCase):
    def _create_archive(self, root: Path) -> Path:
        source = root / "source"
        images = source / "coco128" / "images" / "train2017"
        labels = source / "coco128" / "labels" / "train2017"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        label_lines = {
            "image_1.jpg": ["0 0.5 0.5 0.2 0.2", "14 0.5 0.5 0.1 0.1"],
            "image_2.jpg": ["2 0.5 0.5 0.3 0.3"],
            "image_3.jpg": ["16 0.5 0.5 0.2 0.2"],
            "image_4.jpg": [],
            "image_5.jpg": ["28 0.5 0.5 0.1 0.1"],
        }
        for index, (name, lines) in enumerate(label_lines.items(), 1):
            (images / name).write_bytes(_encoded_image())
            (labels / name.replace(".jpg", ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
        archive = root / "coco128.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for path in source.rglob("*"):
                if path.is_file():
                    output.write(path, path.relative_to(source))
        return archive

    def test_prepare_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = self._create_archive(root)
            data_yaml, report_path, stats = prepare_road_dataset(
                archive=archive,
                raw_root=root / "raw",
                output_root=root / "processed",
                val_ratio=0.4,
                seed=7,
                force=True,
                expected_size=None,
                expected_sha256=None,
            )

            report = validate_yolo_dataset(data_yaml)
            self.assertTrue(report.is_valid, report.errors)
            self.assertEqual(report.images, 5)
            self.assertEqual(report.labels, 5)
            self.assertGreater(report.objects, 0)
            self.assertTrue(report_path.exists())
            self.assertEqual(stats["filtered_objects"], 1)

    def test_unsafe_zip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "no")
            with self.assertRaisesRegex(ValueError, "escapes destination"):
                extract_zip_safely(archive, root / "output")


if __name__ == "__main__":
    unittest.main()
