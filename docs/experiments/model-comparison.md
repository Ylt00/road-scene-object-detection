# YOLOv8n and P2 Model Comparison

Date: 2026-09-14

## Objective

Test whether adding a P2 high-resolution detection head improves the road-scene model in the current small-data CPU setting.

## Controlled Setup

- Same 102 training images
- Same 26 validation images
- Same 16 classes
- Image size: 320
- Batch size: 8
- Training epochs: 10
- CPU device
- Seed: 42

The YOLOv8n baseline uses pretrained weights. The P2 model initializes compatible backbone and neck weights from the same YOLOv8n checkpoint; new P2 layers start from random initialization.

## Results

| Model | Precision | Recall | mAP50 | mAP50-95 | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n pretrained | 0.655 | 0.344 | 0.343 | 0.215 | 3,008,768 | 8.1 |
| YOLOv8n-P2 | 0.746 | 0.010 | 0.010 | 0.003 | 2,923,152 | 12.2 |

## Result

The P2 experiment does not improve performance under the current setup. mAP50 decreases by 0.334 and mAP50-95 decreases by 0.212 compared with the baseline.

## Why This Happened

- The dataset contains only 102 training images.
- The P2 detection head adds randomly initialized layers.
- Only 10 CPU epochs were used.
- Several classes have very few training and validation examples.
- Higher-resolution prediction layers increase optimization difficulty without enough data.

## Correct Interpretation

This is a negative result for one small-data, short-training configuration. It does not prove that P2 is ineffective in general. A valid future comparison should use more images, longer training, multiple seeds, and a P2 model initialized from a compatible pretrained P2 checkpoint when available.

## Files

- `reports/road-yolov8n-p2-metrics.json`
- `reports/model-comparison.json`
- `docs/analysis/model-comparison.png`