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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
