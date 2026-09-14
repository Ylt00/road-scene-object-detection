from pathlib import Path
import json
import tempfile
import unittest

from road_scene.compare import compare_models


class CompareModelsTests(unittest.TestCase):
    def test_compare_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            output_json = root / "comparison.json"
            output_png = root / "comparison.png"
            reference.write_text(
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
            candidate.write_text(
                json.dumps(
                    {
                        "metrics/precision(B)": 0.5,
                        "metrics/recall(B)": 0.3,
                        "metrics/mAP50(B)": 0.2,
                        "metrics/mAP50-95(B)": 0.1,
                    }
                ),
                encoding="utf-8",
            )

            comparison = compare_models(
                "baseline",
                reference,
                "p2",
                candidate,
                output_json,
                output_png,
            )
            self.assertTrue(output_json.exists())
            self.assertTrue(output_png.exists())
            self.assertAlmostEqual(comparison["delta"]["metrics/mAP50(B)"], -0.1)


if __name__ == "__main__":
    unittest.main()