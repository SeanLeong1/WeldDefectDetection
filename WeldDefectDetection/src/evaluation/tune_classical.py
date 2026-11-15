from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from src.algorithms.classical.morphology_detector import MorphologyDetector
from src.data.datasimulation_dataset import DataSimulationSegmentationDataset
from src.utils.file_io import load_config, write_csv, write_json
from src.utils.metrics import aggregate_segmentation_metrics, evaluate_segmentation_masks


DEFAULT_PARAMETER_GRID: dict[str, tuple[int | float, ...]] = {
    "thresh_c": (16, 24, 32, 40, 48),
    "blackhat_kernel_size": (31, 41),
    "opening_kernel_size": (3,),
    "min_defect_area": (5, 15),
    "weld_roi_erode_size": (1,),
    "max_component_aspect_ratio": (20.0, 30.0, 1000.0),
    "max_component_span_fraction": (0.40, 0.60, 1.00),
}


def _parameter_candidates(
    grid: dict[str, Iterable[int | float]],
) -> Iterable[dict[str, int | float]]:
    names = list(grid)
    for values in product(*(grid[name] for name in names)):
        yield dict(zip(names, values))


def _load_native_samples(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    data_cfg = config.get("data", {})
    dataset = DataSimulationSegmentationDataset(
        dataset_root=data_cfg["dataset_root"],
        split=split,
        image_channel=data_cfg.get("image_channel", "topo"),
        mask_key=data_cfg.get("mask_key", "mask"),
        input_size=0,
    )
    samples: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        samples.append(
            {
                "sample_id": str(sample["sample_id"]),
                "image": sample["image"][0].detach().cpu().numpy(),
                "mask": (sample["mask"][0].detach().cpu().numpy() > 0.5).astype(np.uint8),
            }
        )
    return samples


def _score_candidate(
    base_config: dict[str, Any],
    parameters: dict[str, int | float],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_config = deepcopy(base_config)
    candidate_config.setdefault("classical", {}).update(parameters)
    detector = MorphologyDetector(candidate_config)
    metric_rows: list[dict[str, float | int]] = []
    image_ious: list[float] = []
    image_tp = image_tn = image_fp = image_fn = 0
    for sample in samples:
        image_u8 = (np.clip(sample["image"], 0.0, 1.0) * 255.0).astype(np.uint8)
        prediction = detector.predict_binary_mask(image_u8)
        target = sample["mask"]
        row = evaluate_segmentation_masks(prediction, target)
        metric_rows.append(row)
        target_positive = bool(np.any(target > 0))
        prediction_positive = bool(np.any(prediction > 0))
        image_tp += int(target_positive and prediction_positive)
        image_tn += int(not target_positive and not prediction_positive)
        image_fp += int(not target_positive and prediction_positive)
        image_fn += int(target_positive and not prediction_positive)
        union = int(np.logical_or(prediction > 0, target > 0).sum())
        image_ious.append(float(row["iou"]) if union else 1.0)
    aggregate = aggregate_segmentation_metrics(metric_rows)
    image_precision = image_tp / (image_tp + image_fp) if image_tp + image_fp else 0.0
    image_recall = image_tp / (image_tp + image_fn) if image_tp + image_fn else 0.0
    image_specificity = image_tn / (image_tn + image_fp) if image_tn + image_fp else 0.0
    image_balanced_accuracy = 0.5 * (image_recall + image_specificity)
    image_f1 = (
        2.0 * image_precision * image_recall / (image_precision + image_recall)
        if image_precision + image_recall
        else 0.0
    )
    return {
        **parameters,
        "pixel_iou": float(aggregate["iou"]),
        "pixel_f1": float(aggregate["f1"]),
        "pixel_precision": float(aggregate["precision"]),
        "pixel_recall": float(aggregate["recall"]),
        "macro_image_iou": float(np.mean(image_ious)) if image_ious else 0.0,
        "image_f1": float(image_f1),
        "image_precision": float(image_precision),
        "image_recall": float(image_recall),
        "image_specificity": float(image_specificity),
        "image_balanced_accuracy": float(image_balanced_accuracy),
        "image_tp": int(image_tp),
        "image_tn": int(image_tn),
        "image_fp": int(image_fp),
        "image_fn": int(image_fn),
        "tp": int(aggregate["tp"]),
        "fp": int(aggregate["fp"]),
        "fn": int(aggregate["fn"]),
    }


def tune_classical_baseline(
    config_path: str | Path,
    *,
    split: str = "val",
    output_dir: str | Path,
    parameter_grid: dict[str, Iterable[int | float]] | None = None,
) -> dict[str, Any]:
    if str(split).strip().lower() == "test":
        raise ValueError("The test split must not be used for classical parameter selection.")
    config = load_config(config_path)
    samples = _load_native_samples(config, split)
    rows = [
        _score_candidate(config, parameters, samples)
        for parameters in _parameter_candidates(parameter_grid or DEFAULT_PARAMETER_GRID)
    ]
    rows.sort(
        key=lambda row: (
            float(row["image_balanced_accuracy"]),
            float(row["image_f1"]),
            float(row["pixel_iou"]),
            float(row["macro_image_iou"]),
        ),
        reverse=True,
    )
    best = rows[0]
    parameter_names = list((parameter_grid or DEFAULT_PARAMETER_GRID).keys())
    best_parameters = {name: best[name] for name in parameter_names}
    tuned_config = deepcopy(config)
    tuned_config.setdefault("classical", {}).update(best_parameters)

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "candidate_results.csv", rows)
    tuned_config_path = root / "tuned_config.yaml"
    tuned_config_path.write_text(yaml.safe_dump(tuned_config, sort_keys=False), encoding="utf-8")
    summary = {
        "status": "ok",
        "selection_split": split,
        "selection_metric": "image_balanced_accuracy_then_f1_then_pixel_iou",
        "candidate_count": len(rows),
        "sample_count": len(samples),
        "best_parameters": best_parameters,
        "best_validation_metrics": {
            key: best[key]
            for key in (
                "pixel_iou",
                "pixel_f1",
                "pixel_precision",
                "pixel_recall",
                "macro_image_iou",
                "image_f1",
                "image_precision",
                "image_recall",
                "image_specificity",
                "image_balanced_accuracy",
                "image_tp",
                "image_tn",
                "image_fp",
                "image_fn",
                "tp",
                "fp",
                "fn",
            )
        },
        "tuned_config": str(tuned_config_path),
        "candidate_results": str(root / "candidate_results.csv"),
    }
    write_json(root / "tuning_summary.json", summary)
    return summary
