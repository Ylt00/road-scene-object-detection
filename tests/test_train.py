from pathlib import Path
import tempfile
import unittest

import yaml

from road_scene.train import load_experiment_config, metrics_to_dict


class TrainConfigTests(unittest.TestCase):
    def test_config_paths_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "test.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": "yolov8n.pt",
                        "data": "data/processed/road-scene/data.yaml",
                        "project": "runs/train",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            config, project_root = load_experiment_config(config_path)
            self.assertEqual(project_root, root)
            self.assertEqual(config["data"], str((root / "data/processed/road-scene/data.yaml").resolve()))
            self.assertEqual(config["project"], str((root / "runs/train").resolve()))

    def test_metrics_to_dict(self) -> None:
        class FakeMetrics:
            results_dict = {"metrics/mAP50(B)": 0.75, "metrics/mAP50-95(B)": 0.5}

        metrics = metrics_to_dict(FakeMetrics())
        self.assertEqual(metrics["metrics/mAP50(B)"], 0.75)


if __name__ == "__main__":
    unittest.main()