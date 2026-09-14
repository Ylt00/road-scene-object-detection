"""Compare evaluation metrics across models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = ("Precision", "Recall", "mAP50", "mAP50-95")
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def load_metrics(path: str | Path) -> dict[str, float]:
    """Load a metrics JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: float(data[key]) for key in METRIC_KEYS}


def compare_models(
    reference_name: str,
    reference_metrics: str | Path,
    candidate_name: str,
    candidate_metrics: str | Path,
    output_json: str | Path,
    output_png: str | Path,
) -> dict[str, object]:
    """Compare two models and save JSON and a grouped bar chart."""

    reference = load_metrics(reference_metrics)
    candidate = load_metrics(candidate_metrics)
    comparison = {
        "reference": {"name": reference_name, "metrics": reference},
        "candidate": {"name": candidate_name, "metrics": candidate},
        "delta": {
            key: candidate[key] - reference[key]
            for key in METRIC_KEYS
        },
    }

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")

    x = np.arange(len(METRIC_KEYS))
    width = 0.36
    reference_values = [reference[key] for key in METRIC_KEYS]
    candidate_values = [candidate[key] for key in METRIC_KEYS]

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - width / 2, reference_values, width, label=reference_name, color="#4C78A8")
    axis.bar(x + width / 2, candidate_values, width, label=candidate_name, color="#E45756")
    axis.set_xticks(x)
    axis.set_xticklabels(METRIC_LABELS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("YOLOv8n Baseline vs P2 Small-Object Experiment")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for index, value in enumerate(reference_values):
        axis.text(index - width / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    for index, value in enumerate(candidate_values):
        axis.text(index + width / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    png_path = Path(output_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return comparison