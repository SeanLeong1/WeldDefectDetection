from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from src.utils.image_preprocessing import PREPROCESSING_VERSION, filter_small_components

from src.algorithms.classical.cnn_detector import CNNDetector, TorchNotInstalledError
from src.algorithms.classical.morphology_detector import MorphologyDetector
from src.data.datasimulation_dataset import DataSimulationSegmentationDataset
from src.data.static_audit import audit_static_dataset
from src.utils.file_io import create_run_dir, load_config, save_config_snapshot, write_csv, write_json
from src.utils.metrics import aggregate_segmentation_metrics, evaluate_segmentation_masks


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(bool(row["target_is_defective"]) and bool(row["predicted_is_defective"]) for row in rows)
    tn = sum(not bool(row["target_is_defective"]) and not bool(row["predicted_is_defective"]) for row in rows)
    fp = sum(not bool(row["target_is_defective"]) and bool(row["predicted_is_defective"]) for row in rows)
    fn = sum(bool(row["target_is_defective"]) and not bool(row["predicted_is_defective"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(rows)) if rows else 0.0,
    }


def _per_class_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        class_name = str(row["defect_types"] or "clean").split(";")[0]
        grouped.setdefault(class_name, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for class_name, class_rows in sorted(grouped.items()):
        metrics = aggregate_segmentation_metrics(class_rows)
        metrics["iou_defined"] = bool(
            int(metrics["tp"]) + int(metrics["fp"]) + int(metrics["fn"])
        )
        if not metrics["iou_defined"]:
            metrics["iou"] = None
        summary[class_name] = metrics
    return summary


def _per_regime_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["beam_regime"]), []).append(row)
    return {
        regime: aggregate_segmentation_metrics(regime_rows)
        for regime, regime_rows in sorted(grouped.items())
    }


def _subgroup_summary(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "pixel": aggregate_segmentation_metrics(group_rows),
            "image_classification": _classification_summary(group_rows),
        }
        for name, group_rows in sorted(grouped.items())
    }


def _defect_size_band(target_area_mm2: float) -> str:
    if target_area_mm2 <= 0.0:
        return "clean"
    if target_area_mm2 < 0.25:
        return "small_lt_0.25_mm2"
    if target_area_mm2 < 1.0:
        return "medium_0.25_to_1.0_mm2"
    return "large_ge_1.0_mm2"


def _defect_contrast_band(contrast: float | None) -> str:
    if contrast is None:
        return "clean"
    if contrast < 0.05:
        return "low_lt_0.05"
    if contrast < 0.15:
        return "medium_0.05_to_0.15"
    return "high_ge_0.15"


def _bootstrap_by_image(
    rows: list[dict[str, Any]], *, seed: int = 20260831, iterations: int = 500
) -> dict[str, Any]:
    """Estimate uncertainty by resampling independent static images."""
    if not rows:
        return {
            "iterations": 0,
            "resampling_unit": "image",
            "pixel_iou": None,
            "image_balanced_accuracy": None,
        }
    rng = np.random.default_rng(seed)
    pixel_ious: list[float] = []
    balanced_accuracies: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(rows), size=len(rows))
        sampled = [rows[int(index)] for index in indices]
        pixel_ious.append(float(aggregate_segmentation_metrics(sampled)["iou"]))
        balanced_accuracies.append(
            float(_classification_summary(sampled)["balanced_accuracy"])
        )

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
        }

    return {
        "iterations": iterations,
        "resampling_unit": "image",
        "pixel_iou": interval(pixel_ious),
        "image_balanced_accuracy": interval(balanced_accuracies),
    }


def _overlay_image(image: np.ndarray, pred_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    base = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.stack([base, base, base], axis=-1)
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    rgb[target] = 0.60 * rgb[target] + 0.40 * np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rgb[pred & ~target] = 0.55 * rgb[pred & ~target] + 0.45 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[pred & target] = 0.50 * rgb[pred & target] + 0.50 * np.array([1.0, 1.0, 0.0], dtype=np.float32)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _predict_mask(algorithm: str, detector, image_gray: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(image_gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    if algorithm == "morphology":
        return detector.predict_binary_mask(image_u8).astype(np.uint8)
    if algorithm == "cnn":
        return detector.predict_binary_mask(image_gray).astype(np.uint8)
    raise ValueError(f"Unknown algorithm '{algorithm}'. Expected 'morphology' or 'cnn'.")


def evaluate_dataset(
    config_path: str | Path,
    algorithm: str,
    split: str,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    dataset_root_override: str | Path | None = None,
) -> dict[str, Any]:
    algorithm_key = str(algorithm).strip().lower()
    if algorithm_key not in {"morphology", "cnn"}:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Expected 'morphology' or 'cnn'.")

    config = load_config(config_path)
    data_cfg = deepcopy(config.get("data", {}))
    eval_cfg = deepcopy(config.get("eval", {}))
    cnn_cfg = deepcopy(config.get("cnn", {}))

    dataset_root = (
        str(Path(dataset_root_override).expanduser().resolve())
        if dataset_root_override is not None
        else data_cfg.get("dataset_root")
    )
    if not dataset_root:
        raise ValueError("config.data.dataset_root must be set before evaluation.")
    config.setdefault("data", {})["dataset_root"] = str(dataset_root)
    dataset_audit = None
    if str(split).strip().lower() == "all":
        development_root = data_cfg.get("development_dataset_root")
        if not development_root:
            raise ValueError(
                "split='all' is reserved for frozen confirmation scoring and requires "
                "config.data.development_dataset_root."
            )
        dataset_audit = audit_static_dataset(
            dataset_root, compare_roots=[development_root]
        )
        if dataset_audit["status"] != "ok":
            raise ValueError(
                "Confirmation dataset independence audit failed: "
                + "; ".join(dataset_audit["errors"])
            )
    input_size = (
        int(config.get("classical", {}).get("input_size", 0))
        if algorithm_key == "morphology"
        else int(cnn_cfg.get("input_size", 256))
    )
    dataset = DataSimulationSegmentationDataset(
        dataset_root=dataset_root,
        split=split,
        image_channel=data_cfg.get("image_channel", "topo"),
        mask_key=data_cfg.get("mask_key", "mask"),
        input_size=input_size,
    )

    if algorithm_key == "morphology":
        detector = MorphologyDetector(config)
        model_identity: dict[str, Any] = {
            "algorithm": "morphology",
            "parameter_count": 0,
            "parameters": deepcopy(config.get("classical", {})),
        }
    else:
        if checkpoint is not None:
            config.setdefault("cnn", {})["checkpoint_path"] = str(Path(checkpoint).expanduser().resolve())
        detector = CNNDetector(config)
        checkpoint_path = config.get("cnn", {}).get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("CNN evaluation requires a checkpoint path.")
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        model_identity = {
            "algorithm": "cnn",
            "preprocessing_version": PREPROCESSING_VERSION,
            "mask_policy": "8-connected-minimum-pixel-area/v2",
            "checkpoint": str(checkpoint_file),
            "checkpoint_sha256": _sha256(checkpoint_file),
            "threshold": float(detector.threshold),
            "min_defect_area": int(detector.min_defect_area),
            "parameter_count": int(
                sum(parameter.numel() for parameter in detector.model.parameters())
            ),
        }

    run_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else create_run_dir(eval_cfg.get("experiment_root", "experiments/evaluations"), f"{algorithm_key}_{split}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, run_dir)
    overlay_dir = run_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    inference_started_at = time.perf_counter()
    max_overlay_images = int(eval_cfg.get("max_overlay_images", 8))
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        image = sample["image"][0].detach().cpu().numpy().astype(np.float32)
        target_mask = (sample["mask"][0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        pred_mask = _predict_mask(algorithm_key, detector, image)
        metrics = evaluate_segmentation_masks(pred_mask, target_mask)
        target_positive = target_mask > 0
        target_pixels = int(np.count_nonzero(target_positive))
        target_area_mm2 = float(
            target_pixels * float(sample["pixel_area_mm2"])
        )
        defect_contrast = None
        if target_pixels and np.any(~target_positive):
            defect_contrast = float(
                abs(
                    float(np.mean(image[target_positive]))
                    - float(np.mean(image[~target_positive]))
                )
            )
        row = {
            "dataset_index": int(sample["index"]),
            "sample_id": str(sample["sample_id"]),
            "split": str(sample["split"]),
            "defect_types": ";".join(str(value) for value in sample["defects"]),
            "beam_regime": str(sample["beam_regime"]),
            "target_is_defective": bool(np.any(target_mask)),
            "predicted_is_defective": bool(np.any(pred_mask)),
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
            "noise_profile": str(sample["noise_profile"]),
            "pixel_size_mm": float(sample["pixel_size_mm"]),
            "target_positive_pixels": target_pixels,
            "target_area_mm2": target_area_mm2,
            "defect_size_band": _defect_size_band(target_area_mm2),
            "defect_contrast": defect_contrast,
            "defect_contrast_band": _defect_contrast_band(defect_contrast),
            "raw_topo_dynamic_range": sample["raw_topo_dynamic_range"],
            **metrics,
        }
        rows.append(row)

        if sample_idx < max_overlay_images:
            overlay = _overlay_image(image, pred_mask, target_mask)
            overlay_path = overlay_dir / f"{sample_idx:04d}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    inference_runtime_seconds = float(time.perf_counter() - inference_started_at)
    summary = aggregate_segmentation_metrics(rows)
    summary.update(
        {
            "status": "ok",
            "algorithm": algorithm_key,
            "split": str(split),
            "run_dir": str(run_dir),
            "overlay_dir": str(overlay_dir),
            "image_classification": _classification_summary(rows),
            "per_class": _per_class_summary(rows),
            "per_beam_regime": _per_regime_summary(rows),
            "per_noise_profile": _subgroup_summary(rows, "noise_profile"),
            "per_defect_size_band": _subgroup_summary(rows, "defect_size_band"),
            "per_defect_contrast_band": _subgroup_summary(
                rows, "defect_contrast_band"
            ),
            "bootstrap": _bootstrap_by_image(
                rows,
                seed=int(eval_cfg.get("bootstrap_seed", 20260831)),
                iterations=int(eval_cfg.get("bootstrap_iterations", 500)),
            ),
            "confirmation_independence_audit": dataset_audit,
            "model_identity": model_identity,
            "parameter_count": (
                int(sum(parameter.numel() for parameter in detector.model.parameters()))
                if algorithm_key == "cnn"
                else 0
            ),
            "inference_runtime_seconds": inference_runtime_seconds,
            "mean_inference_seconds_per_image": (
                inference_runtime_seconds / len(rows) if rows else None
            ),
        }
    )
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "summary.json", summary)
    return summary
