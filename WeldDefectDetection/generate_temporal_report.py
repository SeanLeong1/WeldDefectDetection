from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import aggregate_segmentation_metrics


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments" / "temporal_progression_20260901"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _per_class(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["planned_class"]].append(
            {key: int(row[key]) for key in ("tp", "tn", "fp", "fn")}
        )
    return {name: aggregate_segmentation_metrics(values) for name, values in sorted(grouped.items())}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "mean_development": ROOT / "experiments/final_sine_temporal_mean_nonoverlap/test/summary.json",
        "learned_development": ROOT / "experiments/final_sine_temporal_learned_nonoverlap/test/summary.json",
        "mean_confirmation": ROOT / "experiments/confirmation_temporal_nonoverlap_20260901/frozen_mean_evaluation/summary.json",
        "learned_confirmation": ROOT / "experiments/confirmation_temporal_nonoverlap_20260901/frozen_evaluation/summary.json",
    }
    summaries = {name: _json(path) for name, path in paths.items()}
    comparison_rows: list[dict] = []
    for name, summary in summaries.items():
        evaluation = summary["evaluation_summary"]
        comparison_rows.append(
            {
                "run": name,
                "window_count": evaluation["window_count"],
                "pixel_iou": evaluation["pixel"]["iou"],
                "pixel_iou_ci_lower": evaluation["bootstrap"]["pixel_iou"]["ci_lower"],
                "pixel_iou_ci_upper": evaluation["bootstrap"]["pixel_iou"]["ci_upper"],
                "balanced_accuracy": evaluation["image_classification"]["balanced_accuracy"],
                "balanced_accuracy_ci_lower": evaluation["bootstrap"]["image_balanced_accuracy"]["ci_lower"],
                "balanced_accuracy_ci_upper": evaluation["bootstrap"]["image_balanced_accuracy"]["ci_upper"],
            }
        )
    with (OUTPUT / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    labels = ["Mean\ndev test", "Learned\ndev test", "Mean\nconfirmation", "Learned\nconfirmation"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "balanced_accuracy", "Image balanced accuracy"),
        (axes[1], "pixel_iou", "Pixel IoU"),
    ):
        values = np.array([row[metric] for row in comparison_rows])
        lower = np.array([row[f"{metric}_ci_lower"] for row in comparison_rows])
        upper = np.array([row[f"{metric}_ci_upper"] for row in comparison_rows])
        colors = ["#3569a8", "#d9822b", "#3569a8", "#d9822b"]
        axis.bar(np.arange(4), values, color=colors, width=0.72)
        axis.errorbar(np.arange(4), values, yerr=np.vstack([values - lower, upper - values]), fmt="none", ecolor="black", capsize=4)
        axis.set_xticks(np.arange(4), labels)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Bidirectional surface prediction on non-overlapping weld patches")
    figure.savefig(OUTPUT / "model_comparison.png", dpi=180)
    plt.close(figure)

    metric_paths = {
        "mean_development": ROOT / "experiments/final_sine_temporal_mean_nonoverlap/test/evaluation_metrics.csv",
        "learned_development": ROOT / "experiments/final_sine_temporal_learned_nonoverlap/test/evaluation_metrics.csv",
        "mean_confirmation": ROOT / "experiments/confirmation_temporal_nonoverlap_20260901/frozen_mean_evaluation/evaluation_metrics.csv",
        "learned_confirmation": ROOT / "experiments/confirmation_temporal_nonoverlap_20260901/frozen_evaluation/evaluation_metrics.csv",
    }
    class_metrics = {name: _per_class(_csv(path)) for name, path in metric_paths.items()}
    defect_classes = [
        "keyhole_pore", "intrinsic_pore", "crack_solidification",
        "crack_liquidation", "lack_of_fusion", "undercut",
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(defect_classes))
    width = 0.19
    for offset, name, color in (
        (-1.5, "mean_development", "#244b78"),
        (-0.5, "learned_development", "#a95b16"),
        (0.5, "mean_confirmation", "#5d94ce"),
        (1.5, "learned_confirmation", "#e4a05e"),
    ):
        axis.bar(
            x + offset * width,
            [class_metrics[name][class_name]["iou"] for class_name in defect_classes],
            width,
            label=name.replace("_", " "),
            color=color,
        )
    axis.set_xticks(x, [name.replace("_", "\n") for name in defect_classes])
    axis.set_ylabel("Pixel IoU")
    axis.set_ylim(0, 1)
    axis.set_title("Per-class surface localization")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.savefig(OUTPUT / "per_class_iou.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "ok",
        "preferred_temporal_method": "unaligned four-frame mean residual",
        "reason": "higher development and confirmation performance than the learned predictor",
        "comparison": comparison_rows,
        "per_class": class_metrics,
        "limitations": [
            "bidirectional and noncausal because future frames are required",
            "spatial adjacent-patch sequence, not physical time evolution",
            "surface-only synthetic TOPO evaluation",
            "undercut remains undetected by the learned predictor",
            "confirmation performance is lower than development test but confidence intervals overlap",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
