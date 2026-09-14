# Road Scene Object Detection

面向智能驾驶和车载视觉的多类别道路场景目标检测项目。

## Project Goals

- 检测车辆、行人、交通设施、小动物和其他道路物体
- 分析小目标的检测性能和漏检原因
- 生成检测结果图和指标分析图
- 对比 YOLOv8n、YOLOv8n-P2 等模型
- 通过 GitHub Actions 自动测试
- 提供可复现的数据、训练和评估流程

## Target Classes

- Vehicles: car, bus, truck, motorcycle, bicycle
- Road users: person
- Traffic facilities: traffic light, stop sign
- Animals: dog, cat, bird, horse, cow, sheep
- Other objects: backpack, umbrella, suitcase

## Planned Pipeline

Dataset preparation

Class filtering and remapping

Dataset validation

YOLOv8n baseline training

YOLOv8n-P2 small-object experiment

Evaluation and error analysis

Visualization and report generation

## Planned Analysis

- Precision, Recall, mAP50, mAP50-95
- Per-class detection performance
- AP_small, AP_medium, AP_large
- Confusion matrix
- Precision-recall curves
- Training and validation loss curves
- Confidence distribution
- Small-object miss detection cases
- Model size, FLOPs, and inference latency

## Dataset Plan

The first version uses a curated road-related subset of COCO128 to validate the complete pipeline. Later versions will extend to BDD100K and custom road-animal datasets when GPU resources are available.

## Dataset Preparation

The first version prepares a curated road-scene subset from COCO128.

Current prepared dataset:

| Item | Value |
|---|---:|
| Source | COCO128 |
| Total images | 128 |
| Training images | 102 |
| Validation images | 26 |
| Retained objects | 389 |
| Filtered objects | 540 |
| Small objects | 153 |
| Medium objects | 114 |
| Large objects | 122 |
| Validation errors | 0 |

Detailed class mapping and processing steps are documented in `docs/dataset.md`.
## YOLOv8n Baseline

| Metric | Value |
|---|---:|
| Precision | 0.655 |
| Recall | 0.344 |
| mAP50 | 0.343 |
| mAP50-95 | 0.215 |

Training and evaluation details are recorded in `docs/experiments/road-yolov8n-baseline.md`.
## Status

The YOLOv8n baseline is trained and evaluated. Visualization, P2 comparison, and report generation will be added in later stages.

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib
- Git and GitHub Actions