# YOLOv8n Road Scene Baseline

Date: 2026-09-14

## Objective
Establish the first reproducible YOLOv8n baseline on the curated road-scene COCO subset.

## Environment
- CPU: Intel Core i5-1155G7
- Python: 3.10.21
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.140
- Model: yolov8n.pt

## Dataset
- Training images: 102
- Validation images: 26
- Retained objects: 389
- Classes: 16

## Training
- Image size: 320
- Batch size: 8
- Epochs: 10
- Device: CPU
- Seed: 42

## Overall Metrics
| Metric | Value |
|---|---:|
| Precision | 0.655 |
| Recall | 0.344 |
| mAP50 | 0.343 |
| mAP50-95 | 0.215 |

## Selected Per-Class Results
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.297 | 0.557 | 0.414 | 0.259 |
| motorcycle | 1.000 | 0.852 | 0.995 | 0.646 |
| cat | 0.290 | 1.000 | 0.995 | 0.597 |
| car | 1.000 | 0.000 | 0.000 | 0.000 |
| truck | 0.000 | 0.000 | 0.000 | 0.000 |

## Outputs
- `runs/train/road-yolov8n-baseline/weights/best.pt`
- `runs/train/road-yolov8n-baseline/results.csv`
- `runs/val/road-yolov8n-baseline/metrics.json`
- `runs/predict/road-yolov8n-baseline/`

## Interpretation
The dataset is small and class-imbalanced. Person dominates the retained objects, and several classes have only one or two validation instances. The baseline is suitable for pipeline verification and later comparison, not for production deployment.

## Next Step
Train the YOLOv8n-P2 configuration with identical data split, image size, batch size, and evaluation settings, then compare overall and small-object performance.
