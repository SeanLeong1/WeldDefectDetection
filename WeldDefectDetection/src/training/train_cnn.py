from __future__ import annotations

import random
import hashlib
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from src.utils.image_preprocessing import PREPROCESSING_VERSION, filter_small_components

from src.algorithms.classical.cnn_detector import TinyDefectSegNet, TorchNotInstalledError, torch
from src.data.datasimulation_dataset import DataSimulationSegmentationDataset
from src.utils.file_io import create_run_dir, load_config, save_config_snapshot, write_csv, write_json
from src.utils.metrics import aggregate_segmentation_metrics, binary_confusion_counts

if torch is not None:
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
else:  # pragma: no cover - optional dependency
    nn = None
    DataLoader = None
    WeightedRandomSampler = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _iterate(loader, description: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=description, leave=False)


def _segmentation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack image tensors while retaining variable-length metadata as lists."""
    if not batch:
        return {}
    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        output[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return output


def _build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    seed: int,
    balance_clean_defective: bool = False,
):
    if DataLoader is None:
        raise TorchNotInstalledError("PyTorch is not installed. Add torch before training the CNN.")
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if balance_clean_defective:
        if WeightedRandomSampler is None:
            raise TorchNotInstalledError("WeightedRandomSampler is unavailable.")
        positives = [bool(record.get("defects")) for record in dataset.records]
        positive_count = sum(positives)
        negative_count = len(positives) - positive_count
        if positive_count and negative_count:
            weights = [
                0.5 / positive_count if positive else 0.5 / negative_count
                for positive in positives
            ]
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        generator=generator,
        collate_fn=_segmentation_collate,
    )


class WeightedBCEDiceLoss(nn.Module if nn is not None else object):
    def __init__(self, positive_weight: float, dice_weight: float = 0.5):
        if nn is None or torch is None:
            raise TorchNotInstalledError("PyTorch is not installed.")
        super().__init__()
        self.dice_weight = float(np.clip(dice_weight, 0.0, 1.0))
        self.register_buffer("positive_weight", torch.tensor([float(positive_weight)]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.positive_weight)

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        intersection = torch.sum(probabilities * targets, dim=dims)
        denominator = torch.sum(probabilities, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        dice_loss = 1.0 - dice.mean()
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _positive_pixel_weight(dataset, maximum: float) -> tuple[float, dict[str, int]]:
    positive = 0
    total = 0
    for index in range(len(dataset)):
        mask = dataset[index]["mask"]
        positive += int((mask > 0.5).sum().item())
        total += int(mask.numel())
    negative = total - positive
    raw = np.sqrt(negative / max(positive, 1))
    return float(np.clip(raw, 1.0, maximum)), {
        "positive_pixels": positive,
        "negative_pixels": negative,
    }


def _metrics_for_thresholds(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    thresholds: np.ndarray,
    min_area: int = 1,
) -> tuple[float, dict[str, float | int]]:
    candidates: list[tuple[float, dict[str, float | int]]] = []
    for threshold in thresholds:
        rows: list[dict[str, int]] = []
        for probability, target in zip(probabilities, targets):
            rows.append(binary_confusion_counts(filter_small_components(probability >= threshold, min_area), target > 0))
        metrics = aggregate_segmentation_metrics(rows)
        candidates.append((float(threshold), metrics))
    return max(
        candidates,
        key=lambda item: (float(item[1]["iou"]), float(item[1]["f1"])),
    )


def _evaluate_model(model, loader, device, criterion, min_area=1) -> tuple[float, float, dict[str, float | int]]:
    if torch is None:
        raise TorchNotInstalledError("PyTorch is not installed.")
    model.eval()
    total_loss = 0.0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += float(loss.item()) * images.shape[0]
            batch_probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            batch_targets = masks.detach().cpu().numpy()
            probabilities.extend(probability[0] for probability in batch_probabilities)
            targets.extend(target[0] for target in batch_targets)
    threshold, summary = _metrics_for_thresholds(
        probabilities,
        targets,
        np.linspace(0.10, 0.90, 33, dtype=np.float32),
        min_area=min_area,
    )
    summary["loss"] = total_loss / max(len(loader.dataset), 1)
    summary["selected_threshold"] = threshold
    return float(summary["loss"]), threshold, summary


def train_cnn(config_path: str | Path, seed_override: int | None = None) -> dict[str, Any]:
    if torch is None or nn is None:
        raise TorchNotInstalledError("PyTorch is not installed. Add torch before training the CNN.")

    started_at = time.perf_counter()
    config = load_config(config_path)
    cnn_cfg = deepcopy(config.get("cnn", {}))
    data_cfg = deepcopy(config.get("data", {}))
    train_cfg = deepcopy(config.get("train", {}))

    device = torch.device(cnn_cfg.get("device", "cpu"))
    input_size = int(cnn_cfg.get("input_size", 256))
    batch_size = int(train_cfg.get("batch_size", 4))
    epochs = int(train_cfg.get("epochs", 10))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    threshold = float(cnn_cfg.get("threshold", 0.5))
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(seed_override if seed_override is not None else train_cfg.get("seed", 20260831))
    train_cfg["seed"] = seed
    config["train"] = train_cfg
    dice_weight = float(train_cfg.get("dice_weight", 0.5))
    maximum_positive_weight = float(train_cfg.get("max_positive_weight", 20.0))
    balance_batches = bool(train_cfg.get("balance_clean_defective_batches", True))
    _set_deterministic_seed(seed)

    dataset_root = data_cfg.get("dataset_root")
    if not dataset_root:
        raise ValueError("config.data.dataset_root must be set before training.")

    train_dataset = DataSimulationSegmentationDataset(
        dataset_root=dataset_root,
        split="train",
        image_channel=data_cfg.get("image_channel", "topo"),
        mask_key=data_cfg.get("mask_key", "mask"),
        input_size=input_size,
    )
    val_dataset = DataSimulationSegmentationDataset(
        dataset_root=dataset_root,
        split="val",
        image_channel=data_cfg.get("image_channel", "topo"),
        mask_key=data_cfg.get("mask_key", "mask"),
        input_size=input_size,
    )
    train_loader = _build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        balance_clean_defective=balance_batches,
    )
    val_loader = _build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 1,
    )

    model = TinyDefectSegNet(
        in_channels=int(cnn_cfg.get("in_channels", 1)),
        base_channels=int(cnn_cfg.get("base_channels", 16)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positive_weight, pixel_counts = _positive_pixel_weight(
        train_dataset, maximum=maximum_positive_weight
    )
    criterion = WeightedBCEDiceLoss(
        positive_weight=positive_weight,
        dice_weight=dice_weight,
    ).to(device)

    run_dir = create_run_dir(train_cfg.get("experiment_root", "experiments/cnn_training"), "cnn_train")
    save_config_snapshot(config, run_dir)

    history: list[dict[str, Any]] = []
    best_iou = float("-inf")
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_rows: list[dict[str, float | int]] = []
        for batch in _iterate(train_loader, f"train epoch {epoch}/{epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * images.shape[0]

            preds = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()
            for pred_mask, target_mask in zip(preds, targets):
                train_rows.append(binary_confusion_counts(filter_small_components(pred_mask[0], int(cnn_cfg.get("min_defect_area", 8))), target_mask[0] > 0))

        train_metrics = aggregate_segmentation_metrics(train_rows)
        train_metrics["loss"] = total_train_loss / max(len(train_loader.dataset), 1)
        _, val_threshold, val_metrics = _evaluate_model(model, val_loader, device, criterion, min_area=int(cnn_cfg.get("min_defect_area", 8)))

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "train_iou": float(train_metrics["iou"]),
            "val_loss": float(val_metrics["loss"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_iou": float(val_metrics["iou"]),
            "val_threshold": float(val_threshold),
        }
        history.append(epoch_row)

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "config": config,
            "threshold": float(val_threshold),
            "min_defect_area": int(cnn_cfg.get("min_defect_area", 8)),
            "preprocessing_version": PREPROCESSING_VERSION,
            "normalization_sha256": (hashlib.sha256((Path(dataset_root)/'topo_normalization.json').read_bytes()).hexdigest()
                                     if (Path(dataset_root)/'topo_normalization.json').is_file() else None),
            "mask_policy": "8-connected-minimum-pixel-area/v2",
            "positive_pixel_weight": positive_weight,
        }
        torch.save(checkpoint, last_checkpoint_path)
        if float(val_metrics["iou"]) >= best_iou:
            best_iou = float(val_metrics["iou"])
            torch.save(checkpoint, best_checkpoint_path)

    history_csv = write_csv(run_dir / "history.csv", history)
    history_json = write_json(run_dir / "history.json", history)
    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "history_csv": str(history_csv),
        "history_json": str(history_json),
        "epochs": epochs,
        "best_val_iou": best_iou,
        "seed": seed,
        "positive_pixel_weight": positive_weight,
        "pixel_counts": pixel_counts,
        "loss": "weighted_bce_dice",
        "dice_weight": dice_weight,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_runtime_seconds": float(time.perf_counter() - started_at),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
