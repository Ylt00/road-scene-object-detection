"""Prepare a road-scene subset from the COCO128 detection dataset."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import urllib.request
import zipfile

import cv2
import numpy as np
import yaml

from .classes import COCO_TO_ROAD_ID, ROAD_CLASSES


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO128_SIZE = 6_983_030
COCO128_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA256 digest of a file."""

    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: str | Path, timeout: int = 60) -> Path:
    """Download a URL to a destination file."""

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "road-scene-object-detection/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response, output.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)
    return output


def verify_archive(
    path: str | Path,
    expected_size: int | None = COCO128_SIZE,
    expected_sha256: str | None = None,
) -> None:
    """Verify the downloaded dataset archive."""

    archive = Path(path)
    if expected_size is not None and archive.stat().st_size != expected_size:
        raise ValueError(
            f"Archive size mismatch for {archive}: expected {expected_size}, got {archive.stat().st_size}"
        )
    if expected_sha256 is not None:
        actual = file_sha256(archive)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(f"SHA256 mismatch for {archive}: expected {expected_sha256}, got {actual}")


def extract_zip_safely(archive: str | Path, destination: str | Path) -> Path:
    """Extract a ZIP file while rejecting paths that escape the destination."""

    archive_path = Path(archive).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as source:
            for member in source.infolist():
                target = (destination_path / member.filename).resolve()
                if target != destination_path and destination_path not in target.parents:
                    raise ValueError(f"Unsafe ZIP member escapes destination: {member.filename}")
            source.extractall(destination_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {archive_path}") from exc
    return destination_path


def remap_label_lines(
    lines: list[str],
    class_mapping: dict[int, int] = COCO_TO_ROAD_ID,
) -> tuple[list[str], int, int, list[str]]:
    """Filter selected classes and remap original class IDs to contiguous IDs."""

    output: list[str] = []
    kept = 0
    filtered = 0
    errors: list[str] = []
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        values = stripped.split()
        if len(values) != 5:
            errors.append(f"line {line_number}: expected 5 fields, got {len(values)}")
            continue
        try:
            original_class_id = int(values[0])
            center_x, center_y, box_width, box_height = map(float, values[1:])
        except ValueError:
            errors.append(f"line {line_number}: class and coordinates must be numeric")
            continue
        if original_class_id not in class_mapping:
            filtered += 1
            continue
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_width, box_height)):
            errors.append(f"line {line_number}: coordinates must be in [0, 1]")
            continue
        if box_width <= 0.0 or box_height <= 0.0:
            errors.append(f"line {line_number}: box size must be positive")
            continue
        new_class_id = class_mapping[original_class_id]
        output.append(
            f"{new_class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
        kept += 1
    return output, kept, filtered, errors


def _find_coco_root(raw_root: Path) -> Path:
    for candidate in (raw_root / "coco128", raw_root):
        if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find COCO128 images and labels under {raw_root}")


def _find_image_directory(coco_root: Path) -> Path:
    preferred = coco_root / "images" / "train2017"
    return preferred if preferred.is_dir() else coco_root / "images"


def _label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir)
    return (labels_dir / relative).with_suffix(".txt")


def _decode_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def prepare_road_dataset(
    archive: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    report_path: str | Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    force: bool = False,
    expected_size: int | None = COCO128_SIZE,
    expected_sha256: str | None = COCO128_SHA256,
) -> tuple[Path, Path, dict[str, object]]:
    """Create a filtered and remapped road-scene YOLO dataset."""

    archive_path = Path(archive).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive does not exist: {archive_path}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    verify_archive(archive_path, expected_size=expected_size, expected_sha256=expected_sha256)
    raw_path = Path(raw_root).expanduser().resolve()
    extract_zip_safely(archive_path, raw_path)
    coco_root = _find_coco_root(raw_path)

    images_dir = _find_image_directory(coco_root)
    if images_dir.name == "train2017":
        labels_dir = coco_root / "labels" / "train2017"
    else:
        labels_dir = coco_root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {labels_dir}")

    images = sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found under {images_dir}")

    rng = random.Random(seed)
    rng.shuffle(images)
    val_count = max(1, min(len(images) - 1, round(len(images) * val_ratio)))
    splits = {
        "train": images[val_count:],
        "val": images[:val_count],
    }

    output_path = Path(output_root).expanduser().resolve()
    if force and output_path.exists():
        shutil.rmtree(output_path)
    for split in splits:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: dict[str, object] = {
        "source_archive": archive_path.name,
        "source_archive_size": archive_path.stat().st_size,
        "source_archive_sha256": file_sha256(archive_path),
        "seed": seed,
        "val_ratio": val_ratio,
        "splits": {},
        "classes": {index: item.name for index, item in enumerate(ROAD_CLASSES)},
        "class_objects": {index: 0 for index in range(len(ROAD_CLASSES))},
        "filtered_objects": 0,
        "size_distribution": {"small": 0, "medium": 0, "large": 0},
        "errors": [],
    }

    for split, split_images in splits.items():
        split_objects = 0
        for image_path in split_images:
            label_path = _label_path_for_image(image_path, images_dir, labels_dir)
            source_lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
            new_lines, kept, filtered, errors = remap_label_lines(source_lines)
            stats["filtered_objects"] = int(stats["filtered_objects"]) + filtered
            stats["errors"].extend(f"{image_path}: {message}" for message in errors)

            destination_image = output_path / "images" / split / image_path.name
            destination_label = (output_path / "labels" / split / image_path.name).with_suffix(".txt")
            shutil.copyfile(image_path, destination_image)
            destination_label.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

            image = _decode_image(image_path)
            image_height, image_width = image.shape[:2]
            split_objects += kept
            for line in new_lines:
                class_id = int(line.split()[0])
                _, _, _, box_width, box_height = map(float, line.split())
                stats["class_objects"][class_id] = int(stats["class_objects"][class_id]) + 1
                pixel_area = box_width * image_width * box_height * image_height
                if pixel_area < 32**2:
                    size_name = "small"
                elif pixel_area < 96**2:
                    size_name = "medium"
                else:
                    size_name = "large"
                stats["size_distribution"][size_name] = int(stats["size_distribution"][size_name]) + 1

        stats["splits"][split] = {"images": len(split_images), "objects": split_objects}

    data = {
        "path": output_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {index: item.name for index, item in enumerate(ROAD_CLASSES)},
        "category_groups": {
            item.category: [cls.name for cls in ROAD_CLASSES if cls.category == item.category]
            for item in ROAD_CLASSES
        },
    }
    data_yaml = output_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = Path(report_path).expanduser().resolve() if report_path else output_path.parent / "road-dataset-stats.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return data_yaml, report, stats
