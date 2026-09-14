"""Generate image grids, training curves, and evaluation charts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .classes import ROAD_CLASS_NAMES


COLORS = (
    (52, 152, 219),
    (46, 204, 113),
    (231, 76, 60),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (26, 188, 156),
    (149, 165, 166),
)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def draw_yolo_labels(
    image_path: str | Path,
    label_path: str | Path,
    output_path: str | Path,
    class_names: tuple[str, ...] = ROAD_CLASS_NAMES,
) -> Path:
    """Draw normalized YOLO bounding boxes on an image."""

    image = _read_image(Path(image_path))
    height, width = image.shape[:2]
    labels = Path(label_path)
    if labels.exists():
        for line in labels.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            class_id = int(values[0])
            center_x, center_y, box_width, box_height = map(float, values[1:])
            x1 = int((center_x - box_width / 2) * width)
            y1 = int((center_y - box_height / 2) * height)
            x2 = int((center_x + box_width / 2) * width)
            y2 = int((center_y + box_height / 2) * height)
            color = COLORS[class_id % len(COLORS)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(image, name, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    output = Path(output_path)
    _write_image(output, image)
    return output


def _resize_with_pad(image: np.ndarray, cell_width: int, cell_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(cell_width / width, cell_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell_height, cell_width, 3), 245, dtype=np.uint8)
    x_offset = (cell_width - new_width) // 2
    y_offset = (cell_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    return canvas


def create_contact_sheet(
    image_paths: list[str | Path],
    output_path: str | Path,
    columns: int = 3,
    cell_width: int = 420,
    cell_height: int = 320,
) -> Path:
    """Create a padded grid from a list of images."""

    if not image_paths:
        raise ValueError("At least one image is required for a contact sheet")
    columns = max(1, columns)
    rows = math.ceil(len(image_paths) / columns)
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
    for index, image_path in enumerate(image_paths):
        image = _read_image(Path(image_path))
        cell = _resize_with_pad(image, cell_width, cell_height)
        row = index // columns
        column = index % columns
        sheet[
            row * cell_height:(row + 1) * cell_height,
            column * cell_width:(column + 1) * cell_width,
        ] = cell
    output = Path(output_path)
    _write_image(output, sheet)
    return output


def select_small_object_samples(
    images_dir: str | Path,
    labels_dir: str | Path,
    limit: int = 9,
) -> list[Path]:
    """Select validation images containing the smallest normalized boxes."""

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    ranked: list[tuple[float, Path]] = []
    for image_path in sorted(images_path.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_path / image_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        areas = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) == 5:
                areas.append(float(values[3]) * float(values[4]))
        if areas:
            ranked.append((min(areas), image_path))
    ranked.sort(key=lambda item: item[0])
    return [path for _, path in ranked[:limit]]


def plot_overall_metrics(metrics_path: str | Path, output_path: str | Path) -> Path:
    """Create a bar chart for overall evaluation metrics."""

    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    values = [float(data[key]) for key in keys]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("YOLOv8n Road Scene Baseline Metrics")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_training_curves(results_csv: str | Path, output_path: str | Path) -> Path:
    """Plot validation metrics from an Ultralytics results.csv file."""

    epochs: list[int] = []
    precision: list[float] = []
    recall: list[float] = []
    map50: list[float] = []
    map5095: list[float] = []
    with Path(results_csv).open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            epochs.append(int(float(row["epoch"])))
            precision.append(float(row["metrics/precision(B)"]))
            recall.append(float(row["metrics/recall(B)"]))
            map50.append(float(row["metrics/mAP50(B)"]))
            map5095.append(float(row["metrics/mAP50-95(B)"]))
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(epochs, precision, marker="o", label="Precision")
    axis.plot(epochs, recall, marker="o", label="Recall")
    axis.plot(epochs, map50, marker="o", label="mAP50")
    axis.plot(epochs, map5095, marker="o", label="mAP50-95")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_title("Training and Validation Metrics")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def copy_analysis_plot(source: str | Path, output_path: str | Path) -> Path | None:
    """Copy an existing analysis plot if it exists."""

    source_path = Path(source)
    if not source_path.exists():
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output)
    return output


def build_visual_report(
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    metrics_path: str | Path,
    results_csv: str | Path,
    plots_dir: str | Path,
    assets_dir: str | Path = "docs/assets",
    analysis_dir: str | Path = "docs/analysis",
    limit: int = 9,
) -> dict[str, list[str]]:
    """Build all report images used by the README and experiment documentation."""

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    predictions_path = Path(predictions_dir)
    assets_path = Path(assets_dir)
    analysis_path = Path(analysis_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    analysis_path.mkdir(parents=True, exist_ok=True)

    prediction_images = sorted(
        path for path in predictions_path.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:limit]
    create_contact_sheet(prediction_images, assets_path / "prediction-grid.jpg")

    source_images = sorted(
        path for path in images_path.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:limit]
    ground_truth_images: list[Path] = []
    temp_dir = assets_path.parent / ".ground-truth-temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for image_path in source_images:
        label_path = labels_path / image_path.with_suffix(".txt").name
        output_path = temp_dir / image_path.name
        draw_yolo_labels(image_path, label_path, output_path)
        ground_truth_images.append(output_path)
    create_contact_sheet(ground_truth_images, assets_path / "ground-truth-grid.jpg")
    shutil.rmtree(temp_dir, ignore_errors=True)

    small_images = select_small_object_samples(images_path, labels_path, limit=limit)
    small_annotated: list[Path] = []
    small_temp_dir = assets_path.parent / ".small-object-temp"
    small_temp_dir.mkdir(parents=True, exist_ok=True)
    for image_path in small_images:
        label_path = labels_path / image_path.with_suffix(".txt").name
        output_path = small_temp_dir / image_path.name
        draw_yolo_labels(image_path, label_path, output_path)
        small_annotated.append(output_path)
    if small_annotated:
        create_contact_sheet(small_annotated, assets_path / "small-object-grid.jpg")
    shutil.rmtree(small_temp_dir, ignore_errors=True)

    plot_overall_metrics(metrics_path, analysis_path / "overall-metrics.png")
    plot_training_curves(results_csv, analysis_path / "training-curves.png")

    copied: list[str] = []
    for source_name, output_name in (
        ("confusion_matrix.png", "confusion-matrix.png"),
        ("BoxPR_curve.png", "precision-recall-curve.png"),
        ("BoxF1_curve.png", "f1-curve.png"),
    ):
        result = copy_analysis_plot(Path(plots_dir) / source_name, analysis_path / output_name)
        if result is not None:
            copied.append(str(result))

    return {
        "assets": [str(path) for path in assets_path.glob("*") if path.is_file()],
        "analysis": [str(path) for path in analysis_path.glob("*") if path.is_file()],
        "copied": copied,
    }