from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from src.algorithms.temporal_predictor import BidirectionalSurfacePredictor, TorchNotInstalledError, torch
from src.data.temporal_window_dataset import TemporalWindowDataset
from src.data.sequence_audit import audit_sequence_dataset
from src.evaluation.temporal_reconstruction import (
    _bootstrap_by_sequence,
    _evaluate_candidate,
    _filter_components,
)
from src.training.train_temporal_predictor import _prediction_rows
from src.utils.file_io import write_csv, write_json
from src.utils.metrics import aggregate_segmentation_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_overlays(root: Path, windows: list[dict[str, Any]], threshold: float, min_area: int) -> None:
    counts: dict[str, int] = defaultdict(int)
    overlay_dir = root / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for window in windows:
        class_name = str(window["planned_class"])
        if counts[class_name] >= 3:
            continue
        prediction_mask = _filter_components(window["residual"] >= threshold, min_area)
        residual_scale = max(float(np.quantile(window["residual"], 0.99)), 1e-6)
        panels = [
            (np.clip(window["target"], 0, 1) * 255).astype(np.uint8),
            (np.clip(window["reconstruction"], 0, 1) * 255).astype(np.uint8),
            (np.clip(window["residual"] / residual_scale, 0, 1) * 255).astype(np.uint8),
        ]
        overlay = cv2.cvtColor(panels[0], cv2.COLOR_GRAY2BGR)
        truth = window["mask"] > 0
        pred = prediction_mask > 0
        overlay[truth] = (0, 180, 0)
        overlay[pred & ~truth] = (0, 0, 255)
        overlay[pred & truth] = (0, 255, 255)
        canvas = np.concatenate(
            [cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR) for panel in panels] + [overlay], axis=1
        )
        cv2.imwrite(str(overlay_dir / f"{window['window_id']}.png"), canvas)
        counts[class_name] += 1


def evaluate_temporal_predictor(
    *,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    seed: int = 20260901,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if torch is None:
        raise TorchNotInstalledError("PyTorch is required for temporal predictor evaluation")
    device = torch.device("cpu")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if "residual_threshold" not in checkpoint or "min_area" not in checkpoint:
        raise ValueError("Checkpoint does not contain validation-selected residual parameters")
    model = BidirectionalSurfacePredictor(
        context_channels=int(checkpoint.get("context_channels", 4)),
        base_channels=int(checkpoint.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset_audit = None
    if str(split).strip().lower() == "all":
        training_root = checkpoint.get("dataset_root")
        if not training_root:
            raise ValueError(
                "Confirmation scoring requires checkpoint.dataset_root for independence auditing."
            )
        dataset_audit = audit_sequence_dataset(
            dataset_root, compare_roots=[training_root]
        )
        if dataset_audit["status"] != "ok":
            raise ValueError(
                "Confirmation sequence independence audit failed: "
                + "; ".join(dataset_audit["errors"])
            )
    dataset = TemporalWindowDataset(dataset_root, split, clean_sequences_only=False)
    if int(checkpoint.get("context_channels", 4)) != dataset.context_channels:
        raise ValueError("Checkpoint context channels do not match the dataset radius.")
    if int(checkpoint.get("stride_px", dataset.stride_px)) != dataset.stride_px:
        raise ValueError("Checkpoint and dataset spatial stride must match.")
    prediction_rows = _prediction_rows(model, dataset, device)
    threshold = float(checkpoint["residual_threshold"])
    min_area = int(checkpoint["min_area"])
    evaluation_summary, metric_rows = _evaluate_candidate(prediction_rows, threshold, min_area)
    evaluation_summary["bootstrap"] = _bootstrap_by_sequence(metric_rows, seed=seed)
    per_class: dict[str, Any] = {}
    for class_name in sorted({str(row["planned_class"]) for row in metric_rows}):
        class_rows = [row for row in metric_rows if row["planned_class"] == class_name]
        per_class[class_name] = aggregate_segmentation_metrics(class_rows)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _save_overlays(root, prediction_rows, threshold, min_area)
    write_csv(root / "evaluation_metrics.csv", metric_rows)
    summary = {
        "status": "ok",
        "method": "learned bidirectional context TOPO predictor",
        "causality": "bidirectional_noncausal",
        "sequence_semantics": "spatially adjacent simulator crops, not physical time evolution",
        "dataset_root": str(Path(dataset_root).resolve()),
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "checkpoint_training_dataset": checkpoint.get("dataset_root"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "context_preparation": checkpoint.get("context_preparation"),
        "selection_split": checkpoint.get("selection_split"),
        "evaluation_split": split,
        "frozen_parameters": {"threshold": threshold, "min_area": min_area},
        "evaluation_summary": evaluation_summary,
        "per_planned_class": per_class,
        "output_dir": str(root),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "inference_runtime_seconds": float(time.perf_counter() - started_at),
        "confirmation_independence_audit": dataset_audit,
    }
    write_json(root / "summary.json", summary)
    return summary
