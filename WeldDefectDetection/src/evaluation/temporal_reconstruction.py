from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from src.data.dataset_contract import load_contract_dataset
from src.data.sequence_audit import audit_sequence_dataset
from src.utils.file_io import write_csv, write_json
from src.utils.metrics import aggregate_segmentation_metrics, binary_confusion_counts


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read temporal image: {path}")
    dtype = image.dtype
    image = image.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        image /= float(np.iinfo(dtype).max)
    return np.clip(image, 0.0, 1.0)


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read temporal mask: {path}")
    return (mask > 0).astype(np.uint8)


def _filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    output = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            output[labels == label] = 1
    return output


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["target_is_defective"] and row["predicted_is_defective"] for row in rows)
    tn = sum(not row["target_is_defective"] and not row["predicted_is_defective"] for row in rows)
    fp = sum(not row["target_is_defective"] and row["predicted_is_defective"] for row in rows)
    fn = sum(row["target_is_defective"] and not row["predicted_is_defective"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "precision": float(precision), "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "f1": float(f1),
    }


def _prepare_windows(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    descriptor, records = load_contract_dataset(dataset_root)
    windows_name = descriptor.get("windows")
    if not windows_name:
        raise ValueError("Sequential dataset descriptor must declare windows.jsonl.")
    windows = _load_jsonl(dataset_root / str(windows_name))
    record_map = {str(record["sample_id"]): record for record in records}
    sequence_manifest_path = dataset_root / "sequence_manifest.json"
    sequence_manifest = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
    planned_class = {row["sequence_id"]: row["planned_class"] for row in sequence_manifest}
    image_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[str, np.ndarray] = {}
    prepared: list[dict[str, Any]] = []

    def image_for(sample_id: str) -> np.ndarray:
        if sample_id not in image_cache:
            image_cache[sample_id] = _load_gray(
                dataset_root / record_map[sample_id]["channels"]["topo"]
            )
        return image_cache[sample_id]

    def mask_for(sample_id: str) -> np.ndarray:
        if sample_id not in mask_cache:
            mask_cache[sample_id] = _load_mask(dataset_root / record_map[sample_id]["mask_path"])
        return mask_cache[sample_id]

    for window in windows:
        target_id = str(window["target_sample_id"])
        context_ids = [
            *[str(value) for value in window["previous_sample_ids"]],
            *[str(value) for value in window["future_sample_ids"]],
        ]
        target = image_for(target_id)
        reconstruction = np.mean([image_for(sample_id) for sample_id in context_ids], axis=0)
        residual = np.abs(target - reconstruction).astype(np.float32)
        target_record = record_map[target_id]
        prepared.append(
            {
                "window_id": str(window["window_id"]),
                "sequence_id": str(window["sequence_id"]),
                "split": str(window["split"]),
                "planned_class": str(planned_class[str(window["sequence_id"])]),
                "target_defect_type": str(target_record["defect_type"]),
                "beam_regime": str(target_record["beam_regime"]),
                "target": target,
                "reconstruction": reconstruction.astype(np.float32),
                "residual": residual,
                "mask": mask_for(target_id),
            }
        )
    return prepared, image_cache


def _evaluate_candidate(
    windows: list[dict[str, Any]], threshold: float, min_area: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        prediction = _filter_components(window["residual"] >= threshold, min_area)
        counts = binary_confusion_counts(prediction, window["mask"])
        rows.append(
            {
                "window_id": window["window_id"],
                "sequence_id": window["sequence_id"],
                "split": window["split"],
                "planned_class": window["planned_class"],
                "target_defect_type": window["target_defect_type"],
                "beam_regime": window["beam_regime"],
                "threshold": float(threshold),
                "min_area": int(min_area),
                "target_is_defective": bool(np.any(window["mask"])),
                "predicted_is_defective": bool(np.any(prediction)),
                **counts,
            }
        )
    pixel = aggregate_segmentation_metrics(rows)
    classification = _classification_metrics(rows)
    summary = {
        "threshold": float(threshold),
        "min_area": int(min_area),
        "window_count": len(rows),
        "pixel": pixel,
        "image_classification": classification,
    }
    return summary, rows


def _bootstrap_by_sequence(
    rows: list[dict[str, Any]], *, seed: int, iterations: int = 500
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence_id"])].append(row)
    sequence_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    pixel_ious: list[float] = []
    balanced_accuracies: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(sequence_ids, size=len(sequence_ids), replace=True)
        sampled_rows = [row for sequence_id in sampled for row in grouped[str(sequence_id)]]
        pixel_ious.append(float(aggregate_segmentation_metrics(sampled_rows)["iou"]))
        balanced_accuracies.append(
            float(_classification_metrics(sampled_rows)["balanced_accuracy"])
        )

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
        }

    return {
        "pixel_iou": interval(pixel_ious),
        "image_balanced_accuracy": interval(balanced_accuracies),
        "iterations": iterations,
        "resampling_unit": "sequence_id",
    }


def _save_overlays(
    root: Path,
    windows: list[dict[str, Any]],
    threshold: float,
    min_area: int,
    max_per_class: int = 3,
) -> None:
    counts: dict[str, int] = defaultdict(int)
    overlay_dir = root / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for window in windows:
        class_name = str(window["planned_class"])
        if counts[class_name] >= max_per_class:
            continue
        prediction = _filter_components(window["residual"] >= threshold, min_area)
        target = window["target"]
        reconstruction = window["reconstruction"]
        residual = window["residual"]
        residual_scale = max(float(np.quantile(residual, 0.99)), 1e-6)
        panels = [
            (target * 255).astype(np.uint8),
            (np.clip(reconstruction, 0, 1) * 255).astype(np.uint8),
            (np.clip(residual / residual_scale, 0, 1) * 255).astype(np.uint8),
        ]
        overlay = cv2.cvtColor(panels[0], cv2.COLOR_GRAY2BGR)
        truth = window["mask"] > 0
        pred = prediction > 0
        overlay[truth] = (0, 180, 0)
        overlay[pred & ~truth] = (0, 0, 255)
        overlay[pred & truth] = (0, 255, 255)
        canvas = np.concatenate(
            [cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR) for panel in panels] + [overlay], axis=1
        )
        cv2.imwrite(str(overlay_dir / f"{window['window_id']}.png"), canvas)
        counts[class_name] += 1


def run_temporal_reconstruction(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    selection_split: str = "val",
    evaluation_split: str = "test",
    seed: int = 20260831,
    frozen_threshold: float | None = None,
    frozen_min_area: int | None = None,
    development_dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if (frozen_threshold is None) != (frozen_min_area is None):
        raise ValueError("frozen_threshold and frozen_min_area must be supplied together.")
    if selection_split == "test":
        raise ValueError("The test split must not be used to select temporal residual parameters.")
    dataset_audit = None
    if str(evaluation_split).strip().lower() == "all":
        if frozen_threshold is None:
            raise ValueError(
                "evaluation_split='all' requires externally frozen residual parameters."
            )
        if development_dataset_root is None:
            raise ValueError(
                "evaluation_split='all' requires development_dataset_root for independence auditing."
            )
        dataset_audit = audit_sequence_dataset(
            dataset_root, compare_roots=[development_dataset_root]
        )
        if dataset_audit["status"] != "ok":
            raise ValueError(
                "Confirmation sequence independence audit failed: "
                + "; ".join(dataset_audit["errors"])
            )
    dataset_path = Path(dataset_root).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepared, _ = _prepare_windows(dataset_path)
    selection = [row for row in prepared if row["split"] == selection_split]
    evaluation = [
        row for row in prepared if evaluation_split == "all" or row["split"] == evaluation_split
    ]
    candidates: list[dict[str, Any]] = []
    if frozen_threshold is None:
        for threshold in np.linspace(0.01, 0.40, 40):
            for min_area in (1, 5, 15, 30):
                summary, _ = _evaluate_candidate(selection, float(threshold), min_area)
                candidates.append(summary)
        best = max(
            candidates,
            key=lambda row: (
                float(row["image_classification"]["balanced_accuracy"]),
                float(row["image_classification"]["f1"]),
                float(row["pixel"]["iou"]),
            ),
        )
        _, selection_rows = _evaluate_candidate(
            selection, float(best["threshold"]), int(best["min_area"])
        )
        selection_source = selection_split
    else:
        best = {
            "threshold": float(frozen_threshold),
            "min_area": int(frozen_min_area),
            "selection_policy": "externally_frozen_parameters",
        }
        selection_rows = []
        selection_source = "externally_frozen"
    evaluation_summary, evaluation_rows = _evaluate_candidate(
        evaluation, float(best["threshold"]), int(best["min_area"])
    )
    evaluation_summary["bootstrap"] = _bootstrap_by_sequence(
        evaluation_rows, seed=seed
    )
    _save_overlays(
        root,
        evaluation,
        float(best["threshold"]),
        int(best["min_area"]),
    )
    write_csv(root / "selection_candidates.csv", [
        {
            "threshold": row["threshold"],
            "min_area": row["min_area"],
            "balanced_accuracy": row["image_classification"]["balanced_accuracy"],
            "image_f1": row["image_classification"]["f1"],
            "pixel_iou": row["pixel"]["iou"],
        }
        for row in candidates
    ])
    write_csv(root / "selection_metrics.csv", selection_rows)
    write_csv(root / "evaluation_metrics.csv", evaluation_rows)
    summary = {
        "status": "ok",
        "method": "unaligned mean of two previous and two future TOPO frames",
        "causality": "bidirectional_noncausal",
        "dataset_root": str(dataset_path),
        "selection_split": selection_split,
        "parameter_selection_source": selection_source,
        "evaluation_split": evaluation_split,
        "selected_parameters": {
            "threshold": best["threshold"],
            "min_area": best["min_area"],
        },
        "selection_summary": best,
        "evaluation_summary": evaluation_summary,
        "output_dir": str(root),
        "parameter_count": 0,
        "runtime_seconds": float(time.perf_counter() - started_at),
        "confirmation_independence_audit": dataset_audit,
    }
    write_json(root / "summary.json", summary)
    return summary
