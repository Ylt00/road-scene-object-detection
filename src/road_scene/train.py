"""Training, evaluation, and prediction wrappers for Ultralytics YOLO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML config and resolve project-relative paths."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    project_root = path.parent.parent if path.parent.name.lower() == "configs" else Path.cwd()
    if "model" in data:
        model_path = Path(str(data["model"])).expanduser()
        if not model_path.is_absolute() and len(model_path.parts) == 1:
            local_weight = project_root / "weights" / model_path
            if local_weight.exists():
                data["model"] = str(local_weight.resolve())
    if "pretrained_model" in data:
        pretrained_path = Path(str(data["pretrained_model"])).expanduser()
        if not pretrained_path.is_absolute():
            pretrained_path = project_root / pretrained_path
        data["pretrained_model"] = str(pretrained_path.resolve())
    if "data" in data:
        data_path = Path(str(data["data"])).expanduser()
        if not data_path.is_absolute():
            data_path = project_root / data_path
        data["data"] = str(data_path.resolve())
    if "project" in data:
        project_path = Path(str(data["project"])).expanduser()
        if not project_path.is_absolute():
            project_path = project_root / project_path
        data["project"] = str(project_path.resolve())
    return data, project_root


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for training. Install with: pip install -e '.[train]'"
        ) from exc
    return YOLO


def metrics_to_dict(metrics: Any) -> dict[str, float]:
    """Convert Ultralytics metrics into a JSON-serializable mapping."""

    results = getattr(metrics, "results_dict", {})
    return {str(key): float(value) for key, value in results.items()}


def train_model(config_path: str | Path, overrides: dict[str, Any] | None = None) -> Any:
    """Train a model from a YAML experiment config."""

    config, _ = load_experiment_config(config_path)
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    model_name = config.pop("model", "yolov8n.pt")
    pretrained_model = config.pop("pretrained_model", None)
    if "data" not in config:
        raise ValueError("Training config must define 'data'.")

    model = _import_yolo()(model_name)
    if pretrained_model:
        model.load(str(pretrained_model))
    return model.train(**config)


def evaluate_model(
    weights: str | Path,
    data: str | Path,
    imgsz: int = 320,
    device: str = "cpu",
    project: str | Path = "runs/val",
    name: str = "road-yolov8n-baseline",
) -> tuple[Any, Path]:
    """Evaluate model weights and write a metrics JSON file."""

    weights_path = Path(weights).expanduser().resolve()
    data_path = Path(data).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    model = _import_yolo()(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=imgsz,
        device=device,
        project=str(Path(project).expanduser().resolve()),
        name=name,
        plots=True,
    )
    output_dir = Path(project).expanduser().resolve() / name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_to_dict(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics, metrics_path


def predict_model(
    weights: str | Path,
    source: str | Path,
    imgsz: int = 320,
    device: str = "cpu",
    project: str | Path = "runs/predict",
    name: str = "road-yolov8n-baseline",
    conf: float = 0.25,
) -> Any:
    """Run inference on an image, directory, video, or stream."""

    weights_path = Path(weights).expanduser().resolve()
    source_path = Path(source).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction source does not exist: {source_path}")

    model = _import_yolo()(str(weights_path))
    return model.predict(
        source=str(source_path),
        imgsz=imgsz,
        device=device,
        project=str(Path(project).expanduser().resolve()),
        name=name,
        conf=conf,
        save=True,
        plots=True,
    )
