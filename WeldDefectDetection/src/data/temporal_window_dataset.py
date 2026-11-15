from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dataset_contract import load_contract_dataset, validate_dataset_contract

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


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
    return (mask > 0).astype(np.float32)


def align_context_frames(
    frames: list[np.ndarray], frame_indices: list[int], target_index: int, stride_px: int
) -> np.ndarray:
    """Register overlapping spatial crops into the target crop coordinate system."""
    if len(frames) != len(frame_indices) or not frames:
        raise ValueError("frames and frame_indices must have the same non-zero length")
    height, width = frames[0].shape
    aligned = np.full((len(frames), height, width), np.nan, dtype=np.float32)
    for channel, (frame, frame_index) in enumerate(zip(frames, frame_indices)):
        if frame.shape != (height, width):
            raise ValueError("All temporal context frames must have equal shape")
        delta = int((target_index - frame_index) * stride_px)
        if delta >= 0:
            if delta < width:
                aligned[channel, :, : width - delta] = frame[:, delta:]
        else:
            shift = -delta
            if shift < width:
                aligned[channel, :, shift:] = frame[:, : width - shift]
    available_mean = np.nanmean(aligned, axis=0)
    if np.any(~np.isfinite(available_mean)):
        raise ValueError("Context crops do not cover the complete target frame")
    for channel in range(aligned.shape[0]):
        missing = ~np.isfinite(aligned[channel])
        aligned[channel, missing] = available_mean[missing]
    return aligned.astype(np.float32)


class TemporalWindowDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        clean_sequences_only: bool = False,
    ):
        if torch is None:
            raise ImportError(f"PyTorch is required for TemporalWindowDataset: {_TORCH_IMPORT_ERROR}")
        self.root = Path(dataset_root).expanduser().resolve()
        validation = validate_dataset_contract(self.root, inspect_images=False)
        if validation["status"] != "ok":
            raise ValueError("Temporal contract failed: " + "; ".join(validation["errors"]))
        descriptor, records = load_contract_dataset(self.root)
        if descriptor.get("dataset_kind") != "sequential":
            raise ValueError("TemporalWindowDataset requires a sequential dataset contract")
        self.stride_px = int(descriptor["stride_px"])
        self.context_radius = int(descriptor["context_radius"])
        if self.context_radius < 1:
            raise ValueError("context_radius must be positive.")
        self.context_channels = 2 * self.context_radius
        self.record_map = {str(row["sample_id"]): row for row in records}
        sequence_manifest = json.loads((self.root / "sequence_manifest.json").read_text(encoding="utf-8"))
        self.planned_class = {str(row["sequence_id"]): str(row["planned_class"]) for row in sequence_manifest}
        windows = _load_jsonl(self.root / str(descriptor["windows"]))
        self.windows = [
            row
            for row in windows
            if (str(split) == "all" or str(row["split"]) == str(split))
            and (not clean_sequences_only or self.planned_class[str(row["sequence_id"])] == "clean")
        ]
        if not self.windows:
            raise ValueError(f"No temporal windows found for split '{split}'")
        self._image_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        # Check actual crop positions and dimensions, including inconsistent manifests.
        for window in self.windows:
            target_id = str(window["target_sample_id"])
            target = self.record_map[target_id]
            ids = [*window["previous_sample_ids"], *window["future_sample_ids"]]
            if (len(window["previous_sample_ids"]) != self.context_radius
                    or len(window["future_sample_ids"]) != self.context_radius
                    or len(set(ids)) != len(ids)):
                raise ValueError("Window context does not match declared radius.")
            target_width = self._image(target_id).shape[1]
            start = int(target["crop_x0_px"])
            for sample_id in ids:
                row = self.record_map[str(sample_id)]
                if row["sequence_id"] != target["sequence_id"] or row["split"] != target["split"]:
                    raise ValueError("Context and target must share sequence and split.")
                width = self._image(str(sample_id)).shape[1]
                context_start = int(row["crop_x0_px"])
                if max(start, context_start) < min(start + target_width, context_start + width):
                    raise ValueError("Unseen-region prediction forbids target/context pixel overlap.")

    def __len__(self) -> int:
        return len(self.windows)

    def _image(self, sample_id: str) -> np.ndarray:
        if sample_id not in self._image_cache:
            self._image_cache[sample_id] = _load_gray(
                self.root / self.record_map[sample_id]["channels"]["topo"]
            )
        return self._image_cache[sample_id]

    def _mask(self, sample_id: str) -> np.ndarray:
        if sample_id not in self._mask_cache:
            self._mask_cache[sample_id] = _load_mask(
                self.root / self.record_map[sample_id]["mask_path"]
            )
        return self._mask_cache[sample_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        target_id = str(window["target_sample_id"])
        context_ids = [
            *[str(value) for value in window["previous_sample_ids"]],
            *[str(value) for value in window["future_sample_ids"]],
        ]
        target_index = int(window["target_frame_index"])
        frame_indices = [int(self.record_map[sample_id]["frame_index"]) for sample_id in context_ids]
        context_frames = [self._image(sample_id) for sample_id in context_ids]
        context = np.stack(context_frames).astype(np.float32)
        context_preparation = "non_overlapping_native_coordinates"
        sequence_id = str(window["sequence_id"])
        return {
            "context": torch.from_numpy(context).float(),
            "target": torch.from_numpy(self._image(target_id)[None]).float(),
            "mask": torch.from_numpy(self._mask(target_id)[None]).float(),
            "window_id": str(window["window_id"]),
            "sequence_id": sequence_id,
            "split": str(window["split"]),
            "planned_class": self.planned_class[sequence_id],
            "target_defect_type": str(self.record_map[target_id]["defect_type"]),
            "beam_regime": str(self.record_map[target_id]["beam_regime"]),
            "context_preparation": context_preparation,
        }
