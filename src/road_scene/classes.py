"""Road scene class definitions and COCO class mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoadClass:
    """A selected road-scene class."""

    name: str
    category: str
    coco_id: int


ROAD_CLASSES: tuple[RoadClass, ...] = (
    RoadClass("person", "road_user", 0),
    RoadClass("bicycle", "vehicle", 1),
    RoadClass("car", "vehicle", 2),
    RoadClass("motorcycle", "vehicle", 3),
    RoadClass("bus", "vehicle", 5),
    RoadClass("truck", "vehicle", 7),
    RoadClass("traffic light", "traffic_facility", 9),
    RoadClass("stop sign", "traffic_facility", 11),
    RoadClass("cat", "animal", 15),
    RoadClass("dog", "animal", 16),
    RoadClass("horse", "animal", 17),
    RoadClass("sheep", "animal", 18),
    RoadClass("cow", "animal", 19),
    RoadClass("backpack", "carried_object", 24),
    RoadClass("umbrella", "carried_object", 25),
    RoadClass("suitcase", "carried_object", 28),
)

ROAD_CLASS_NAMES: tuple[str, ...] = tuple(item.name for item in ROAD_CLASSES)
COCO_TO_ROAD_ID: dict[int, int] = {item.coco_id: index for index, item in enumerate(ROAD_CLASSES)}
COCO_ID_TO_NAME: dict[int, str] = {item.coco_id: item.name for item in ROAD_CLASSES}

if len(set(COCO_TO_ROAD_ID.values())) != len(ROAD_CLASSES):
    raise RuntimeError("Road class IDs must be unique.")
if len(set(COCO_TO_ROAD_ID.keys())) != len(ROAD_CLASSES):
    raise RuntimeError("COCO class IDs must be unique.")
