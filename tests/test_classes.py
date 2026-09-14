import unittest

from road_scene.classes import COCO_TO_ROAD_ID, ROAD_CLASSES, ROAD_CLASS_NAMES


class RoadClassTests(unittest.TestCase):
    def test_class_ids_are_contiguous(self) -> None:
        self.assertEqual(list(COCO_TO_ROAD_ID.values()), list(range(len(ROAD_CLASSES))))

    def test_expected_class_names(self) -> None:
        self.assertIn("person", ROAD_CLASS_NAMES)
        self.assertIn("car", ROAD_CLASS_NAMES)
        self.assertIn("traffic light", ROAD_CLASS_NAMES)
        self.assertIn("dog", ROAD_CLASS_NAMES)
        self.assertIn("suitcase", ROAD_CLASS_NAMES)

    def test_known_coco_mapping(self) -> None:
        self.assertEqual(COCO_TO_ROAD_ID[0], 0)
        self.assertEqual(COCO_TO_ROAD_ID[2], 2)
        self.assertEqual(COCO_TO_ROAD_ID[28], len(ROAD_CLASSES) - 1)


if __name__ == "__main__":
    unittest.main()
