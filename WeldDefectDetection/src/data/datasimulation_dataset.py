from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.image_preprocessing import normalize_grayscale

from .dataset_contract import load_contract_dataset, resolve_record_paths, validate_dataset_contract

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _load_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return normalize_grayscale(image)


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to load mask: {path}")
    return (mask > 0).astype(np.float32)


class DataSimulationSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_channel: str = "topo",
        mask_key: str = "mask",
        input_size: int = 256,
    ):
        if torch is None:
            raise ImportError(
                f"PyTorch is required for DataSimulationSegmentationDataset: {_TORCH_IMPORT_ERROR}"
            )
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        manifest_path = self.dataset_root / "manifest.json"
        contract_path = self.dataset_root / "dataset_contract.yaml"
        manifest: list[dict[str, Any]] = []
        if not contract_path.exists():
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Neither dataset_contract.yaml nor legacy manifest.json exists under "
                    f"{self.dataset_root}."
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, list):
                raise ValueError(f"Dataset manifest must be a JSON list: {manifest_path}")

        self.split = str(split)
        self.image_channel = str(image_channel)
        self.mask_key = str(mask_key)
        self.input_size = int(input_size)
        self.records: list[dict[str, Any]] = []
        image_key = f"{self.image_channel}_u16"

        if contract_path.exists():
            validation = validate_dataset_contract(self.dataset_root, inspect_images=False)
            if validation["status"] != "ok":
                raise ValueError("Dataset contract failed: " + "; ".join(validation["errors"]))
            _, contract_records = load_contract_dataset(self.dataset_root)
            for entry_index, entry in enumerate(contract_records):
                entry_split = str(entry.get("split"))
                if self.split != "all" and entry_split != self.split:
                    continue
                resolved = resolve_record_paths(entry, self.dataset_root)
                image_path = resolved["channels"].get(self.image_channel)
                mask_path = resolved["mask_path"]
                if image_path is None:
                    continue
                self.records.append(
                    {
                        "index": entry_index,
                        "sample_id": str(entry.get("sample_id")),
                        "sequence_id": str(entry.get("sequence_id")),
                        "frame_index": int(entry.get("frame_index", 0)),
                        "split": entry_split,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "defects": list(entry.get("defect_types") or []),
                        "beam_regime": str(entry.get("beam_regime", "randomized")),
                        "emission_b_mode": str(entry.get("emission_b_mode", "constant")),
                        "emission_b_value": float(entry.get("emission_b_value", 0.0)),
                        "pixel_size_mm": float(entry.get("pixel_size_mm", 1.0)),
                        "noise_profile": str(
                            entry.get(
                                "noise_profile",
                                "development" if entry.get("noise_enabled") else "none",
                            )
                        ),
                        "raw_topo_dynamic_range": entry.get("raw_topo_dynamic_range"),
                    }
                )
        else:
            for entry in manifest:
                entry_split = str(entry.get("split"))
                if self.split != "all" and entry_split != self.split:
                    continue
                artifacts = entry.get("artifacts") or {}
                image_path = artifacts.get(image_key)
                mask_path = artifacts.get(self.mask_key)
                if not image_path or not mask_path:
                    continue
                entry_index = int(entry.get("index", len(self.records)))
                self.records.append(
                    {
                        "index": entry_index,
                        "sample_id": f"legacy-{entry_index:05d}",
                        "sequence_id": f"legacy-static-{entry_index:05d}",
                        "frame_index": 0,
                        "split": entry_split,
                        "image_path": Path(str(image_path)).expanduser().resolve(),
                        "mask_path": Path(str(mask_path)).expanduser().resolve(),
                        "defects": list(entry.get("defects") or []),
                        "beam_regime": "randomized",
                        "emission_b_mode": str(
                            (entry.get("metadata", {}).get("emission_b") or {}).get(
                                "mode", "constant"
                            )
                        ),
                        "emission_b_value": float(
                            (
                                (entry.get("metadata", {}).get("emission_b") or {}).get(
                                    "b_stats", {}
                                )
                                or {}
                            ).get("mean", 0.0)
                        ),
                        "pixel_size_mm": float(entry.get("pixel_size_mm", 1.0)),
                        "noise_profile": (
                            "development"
                            if bool(entry.get("noise_enabled", False))
                            else "none"
                        ),
                        "raw_topo_dynamic_range": None,
                    }
                )

        if not self.records:
            source_path = contract_path if contract_path.exists() else manifest_path
            raise ValueError(f"No records found for split '{self.split}' in {source_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _load_gray_image(record["image_path"])
        mask = _load_mask(record["mask_path"])
        original_height, original_width = image.shape
        if mask.shape != image.shape:
            raise ValueError("Image and mask dimensions must match before resizing.")
        if self.input_size > 0:
            image = cv2.resize(
                image,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            mask = cv2.resize(
                mask,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
        return {
            "image": torch.from_numpy(image[None, ...]).float(),
            "mask": torch.from_numpy(mask[None, ...]).float(),
            "index": int(record["index"]),
            "sample_id": record["sample_id"],
            "sequence_id": record["sequence_id"],
            "frame_index": int(record["frame_index"]),
            "split": record["split"],
            "image_path": str(record["image_path"]),
            "mask_path": str(record["mask_path"]),
            "defects": record["defects"],
            "beam_regime": record["beam_regime"],
            "emission_b_mode": record["emission_b_mode"],
            "emission_b_value": float(record["emission_b_value"]),
            "pixel_size_mm": float(record["pixel_size_mm"]) * np.sqrt(original_width * original_height / image.size),
            "native_pixel_size_mm": float(record["pixel_size_mm"]),
            "pixel_size_x_mm": float(record["pixel_size_mm"]) * original_width / image.shape[1],
            "pixel_size_y_mm": float(record["pixel_size_mm"]) * original_height / image.shape[0],
            "pixel_area_mm2": float(record["pixel_size_mm"]) ** 2 * original_width * original_height / image.size,
            "noise_profile": record["noise_profile"],
            "raw_topo_dynamic_range": record["raw_topo_dynamic_range"],
        }
