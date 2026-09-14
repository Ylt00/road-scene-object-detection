# Road Scene Dataset

The first dataset version is a curated subset of COCO128. Images are split into training and validation sets, and only road-related classes are retained.

## Selected Classes

| New ID | Class | Category | COCO ID |
|---:|---|---|---:|
| 0 | person | road_user | 0 |
| 1 | bicycle | vehicle | 1 |
| 2 | car | vehicle | 2 |
| 3 | motorcycle | vehicle | 3 |
| 4 | bus | vehicle | 5 |
| 5 | truck | vehicle | 7 |
| 6 | traffic light | traffic_facility | 9 |
| 7 | stop sign | traffic_facility | 11 |
| 8 | cat | animal | 15 |
| 9 | dog | animal | 16 |
| 10 | horse | animal | 17 |
| 11 | sheep | animal | 18 |
| 12 | cow | animal | 19 |
| 13 | backpack | carried_object | 24 |
| 14 | umbrella | carried_object | 25 |
| 15 | suitcase | carried_object | 28 |

## Processing Pipeline

1. Download the official COCO128 archive.
2. Verify the archive size and optional SHA256 value.
3. Extract the ZIP file with path traversal protection.
4. Filter unselected COCO classes.
5. Remap original COCO IDs to contiguous road-scene IDs.
6. Keep images without selected boxes as background images.
7. Split images into deterministic train and validation sets.
8. Generate `data.yaml` and a JSON statistics report.

## Notes

COCO128 is used to validate the complete engineering pipeline. It is not a production-scale autonomous-driving dataset. BDD100K and road-animal datasets can be added after the baseline pipeline is verified.
