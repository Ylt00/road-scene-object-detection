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

## Status

Project initialization. The repository structure, data pipeline, training, evaluation, and visualization components will be added incrementally.

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib
- Git and GitHub Actions