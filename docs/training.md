# YOLOv8n Baseline Training

## Dataset
- Training images: 102
- Validation images: 26
- Retained objects: 389
- Classes: 16

## Train
```powershell
python -m road_scene.cli train --config configs/road-scene-cpu.yaml
```

Configuration: YOLOv8n, imgsz=320, batch=8, device=cpu, epochs=10, seed=42.

Outputs are stored in `runs/train/road-yolov8n-baseline/`.

## Evaluate
```powershell
python -m road_scene.cli evaluate --weights runs/train/road-yolov8n-baseline/weights/best.pt --data data/processed/road-scene/data.yaml --imgsz 320 --device cpu
```

The metrics JSON is written to `runs/val/road-yolov8n-baseline/metrics.json`.

## Predict
```powershell
python -m road_scene.cli predict --weights runs/train/road-yolov8n-baseline/weights/best.pt --source data/processed/road-scene/images/val --imgsz 320 --device cpu
```

Predicted images are stored under `runs/predict/road-yolov8n-baseline/`.

## P2 Experiment
After the baseline, train `configs/road-scene-p2-cpu.yaml` with the same data split, image size, and evaluation settings.
