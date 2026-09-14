"""Command-line interface for the road scene detection project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .data import prepare_road_dataset
from .dataset import validate_yolo_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="road-scene", description="Road scene object detection tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Print the package version")

    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare the road scene COCO subset")
    prepare_parser.add_argument("--archive", required=True)
    prepare_parser.add_argument("--raw", default="data/raw")
    prepare_parser.add_argument("--output", default="data/processed/road-scene")
    prepare_parser.add_argument("--report", default=None)
    prepare_parser.add_argument("--val-ratio", type=float, default=0.2)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--force", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate a YOLO dataset")
    validate_parser.add_argument("--data", required=True)

    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument("--config", default="configs/road-scene-cpu.yaml")
    train_parser.add_argument("--model")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--imgsz", type=int)
    train_parser.add_argument("--device")
    train_parser.add_argument("--name")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate YOLO weights")
    evaluate_parser.add_argument("--weights", required=True)
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--imgsz", type=int, default=320)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--project", default="runs/val")
    evaluate_parser.add_argument("--name", default="road-yolov8n-baseline")

    predict_parser = subparsers.add_parser("predict", help="Run YOLO inference")
    predict_parser.add_argument("--weights", required=True)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--imgsz", type=int, default=320)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--project", default="runs/predict")
    predict_parser.add_argument("--name", default="road-yolov8n-baseline")
    predict_parser.add_argument("--conf", type=float, default=0.25)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "prepare-data":
        data_yaml, report, stats = prepare_road_dataset(
            archive=args.archive,
            raw_root=args.raw,
            output_root=args.output,
            report_path=args.report,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force=args.force,
        )
        print(f"Prepared dataset: {data_yaml}")
        print(f"Report: {report}")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if args.command == "validate":
        report = validate_yolo_dataset(args.data)
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
        return 0 if report.is_valid else 1

    if args.command == "train":
        from .train import train_model

        train_model(
            args.config,
            overrides={
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "device": args.device,
                "name": args.name,
            },
        )
        return 0

    if args.command == "evaluate":
        from .train import evaluate_model

        _, metrics_path = evaluate_model(
            weights=args.weights,
            data=args.data,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
        )
        print(f"Metrics saved to: {metrics_path}")
        return 0

    if args.command == "predict":
        from .train import predict_model

        predict_model(
            weights=args.weights,
            source=args.source,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            conf=args.conf,
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())