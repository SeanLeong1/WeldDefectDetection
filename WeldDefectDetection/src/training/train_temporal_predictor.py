from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.algorithms.temporal_predictor import BidirectionalSurfacePredictor, TorchNotInstalledError, torch
from src.data.temporal_window_dataset import TemporalWindowDataset
from src.evaluation.temporal_reconstruction import _evaluate_candidate
from src.utils.file_io import write_csv, write_json

if torch is not None:
    from torch.utils.data import DataLoader


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reconstruction_loss(prediction, target):
    intensity = torch.mean(torch.abs(prediction - target))
    dx = torch.mean(torch.abs((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])))
    dy = torch.mean(torch.abs((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])))
    return 0.8 * intensity + 0.1 * dx + 0.1 * dy


def _prediction_rows(model, dataset, device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in dataset:
            prediction = model(item["context"][None].to(device))[0, 0].cpu().numpy().astype(np.float32)
            target = item["target"][0].numpy().astype(np.float32)
            rows.append(
                {
                    "window_id": item["window_id"],
                    "sequence_id": item["sequence_id"],
                    "split": item["split"],
                    "planned_class": item["planned_class"],
                    "target_defect_type": item["target_defect_type"],
                    "beam_regime": item["beam_regime"],
                    "target": target,
                    "reconstruction": prediction,
                    "residual": np.abs(target - prediction).astype(np.float32),
                    "mask": item["mask"][0].numpy().astype(np.uint8),
                }
            )
    return rows


def _select_residual_parameters(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    residual_values = np.concatenate([row["residual"].reshape(-1) for row in rows])
    thresholds = sorted(
        set(
            float(value)
            for value in np.concatenate(
                [
                    np.linspace(0.005, 0.30, 45, dtype=np.float32),
                    np.quantile(residual_values, np.linspace(0.90, 0.9995, 30)),
                ]
            )
            if value > 0.0
        )
    )
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        for min_area in (1, 5, 15, 30, 60, 120, 240):
            summary, _ = _evaluate_candidate(rows, threshold, min_area)
            candidates.append(summary)
    best = max(
        candidates,
        key=lambda row: (
            float(row["image_classification"]["balanced_accuracy"]),
            float(row["pixel"]["iou"]),
            float(row["image_classification"]["f1"]),
        ),
    )
    return best, candidates


def train_temporal_predictor(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    seed: int = 20260901,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    base_channels: int = 8,
    device_name: str = "cpu",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if torch is None:
        raise TorchNotInstalledError("PyTorch is required for temporal predictor training")
    _set_seed(seed)
    device = torch.device(device_name)
    train_dataset = TemporalWindowDataset(dataset_root, "train", clean_sequences_only=True)
    clean_val_dataset = TemporalWindowDataset(dataset_root, "val", clean_sequences_only=True)
    full_val_dataset = TemporalWindowDataset(dataset_root, "val", clean_sequences_only=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    clean_val_loader = DataLoader(clean_val_dataset, batch_size=batch_size, shuffle=False)
    model = BidirectionalSurfacePredictor(context_channels=train_dataset.context_channels, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    checkpoint_path = root / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(context)
            loss = _reconstruction_loss(prediction, target)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * context.shape[0]
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in clean_val_loader:
                context = batch["context"].to(device)
                target = batch["target"].to(device)
                val_total += float(_reconstruction_loss(model(context), target).item()) * context.shape[0]
        row = {
            "epoch": epoch,
            "train_clean_reconstruction_loss": train_total / len(train_dataset),
            "val_clean_reconstruction_loss": val_total / len(clean_val_dataset),
        }
        history.append(row)
        if row["val_clean_reconstruction_loss"] <= best_loss:
            best_loss = float(row["val_clean_reconstruction_loss"])
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "normalization_sha256": (hashlib.sha256((Path(dataset_root)/'topo_normalization.json').read_bytes()).hexdigest()
                                             if (Path(dataset_root)/'topo_normalization.json').is_file() else None),
                    "preprocessing_version": "grayscale-dtype-range/v2",
                    "seed": seed,
                    "base_channels": base_channels,
                    "context_channels": train_dataset.context_channels,
                    "context_radius": train_dataset.context_radius,
                    "training_policy": "clean_sequences_only",
                    "context_preparation": train_dataset[0]["context_preparation"],
                    "stride_px": train_dataset.stride_px,
                    "dataset_root": str(Path(dataset_root).resolve()),
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    val_rows = _prediction_rows(model, full_val_dataset, device)
    best_parameters, candidates = _select_residual_parameters(val_rows)
    checkpoint["residual_threshold"] = float(best_parameters["threshold"])
    checkpoint["min_area"] = int(best_parameters["min_area"])
    checkpoint["selection_split"] = "val"
    checkpoint["selection_summary"] = best_parameters
    torch.save(checkpoint, checkpoint_path)
    write_csv(root / "history.csv", history)
    write_csv(
        root / "selection_candidates.csv",
        [
            {
                "threshold": row["threshold"],
                "min_area": row["min_area"],
                "pixel_iou": row["pixel"]["iou"],
                "balanced_accuracy": row["image_classification"]["balanced_accuracy"],
                "image_f1": row["image_classification"]["f1"],
            }
            for row in candidates
        ],
    )
    summary = {
        "status": "ok",
        "method": "learned bidirectional four-frame TOPO predictor",
        "causality": "bidirectional_noncausal",
        "sequence_semantics": "spatially adjacent simulator crops, not physical time evolution",
        "training_policy": "clean_sequences_only",
        "context_preparation": checkpoint["context_preparation"],
        "dataset_root": str(Path(dataset_root).resolve()),
        "seed": seed,
        "epochs": epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_clean_reconstruction_loss": best_loss,
        "selected_parameters": {
            "threshold": checkpoint["residual_threshold"],
            "min_area": checkpoint["min_area"],
        },
        "selection_summary": best_parameters,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(root),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_runtime_seconds": float(time.perf_counter() - started_at),
    }
    write_json(root / "summary.json", summary)
    return summary
