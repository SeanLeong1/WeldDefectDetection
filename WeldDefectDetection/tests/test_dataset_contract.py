from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.data.dataset_contract import validate_dataset_contract
from src.data.datasimulation_dataset import DataSimulationSegmentationDataset, torch


def _write_dataset(root: Path, *, leak_sequence: bool = False) -> Path:
    descriptor = {
        "contract_version": "weld-topo-segmentation/v2",
        "task": "binary_surface_defect_segmentation",
        "dataset_kind": "static",
        "records": "samples.jsonl",
        "required_channels": ["topo"],
        "noise_policy": "disabled",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_contract.yaml").write_text(
        yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8"
    )
    records = []
    for index, split in enumerate(("train", "val", "test")):
        stem = f"{index:05d}"
        sequence_id = "leaking-sequence" if leak_sequence and index < 2 else f"static-{stem}"
        channel_paths = {}
        for channel in ("topo",):
            relative = Path("images") / channel / split / f"{stem}.png"
            absolute = root / relative
            absolute.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((12, 12), 40 + 10 * index, dtype=np.uint8)
            cv2.imwrite(str(absolute), image)
            channel_paths[channel] = relative.as_posix()
        mask_relative = Path("masks") / split / f"{stem}.png"
        mask_absolute = root / mask_relative
        mask_absolute.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((12, 12), dtype=np.uint8)
        defect_types: list[str] = []
        if index == 1:
            mask[4:8, 4:8] = 255
            defect_types = ["undercut"]
        cv2.imwrite(str(mask_absolute), mask)
        records.append(
            {
                "contract_version": "weld-topo-segmentation/v2",
                "sample_id": f"sample-{stem}",
                "sequence_id": sequence_id,
                "frame_index": index if leak_sequence and index < 2 else 0,
                "split": split,
                "channels": channel_paths,
                "mask_path": mask_relative.as_posix(),
                "defect_type": defect_types[0] if defect_types else "clean",
                "defect_types": defect_types,
                "is_defective": bool(defect_types),
                "seed": 100 + index,
                "beam_regime": "mid",
                "emission_b_mode": "pooled_constant_fit",
                "emission_b_value": -0.77,
                "render_view_mode": "top_surface",
                "pixel_size_mm": 0.05,
                "noise_enabled": False,
            }
        )
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return root


def test_contract_validator_accepts_valid_static_dataset(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    report = validate_dataset_contract(root)
    assert report["status"] == "ok"
    assert report["record_count"] == 3
    assert report["sequence_count"] == 3
    assert report["sequence_leakage_count"] == 0
    assert report["split_counts"] == {"test": 1, "train": 1, "val": 1}


def test_contract_validator_rejects_sequence_split_leakage(tmp_path):
    root = _write_dataset(tmp_path / "dataset", leak_sequence=True)
    report = validate_dataset_contract(root)
    assert report["status"] == "failed"
    assert report["sequence_leakage_count"] == 1
    assert any("Sequence leakage" in error for error in report["errors"])


def test_contract_validator_checks_sliding_window_order(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    descriptor = yaml.safe_load((root / "dataset_contract.yaml").read_text(encoding="utf-8"))
    descriptor.update(
        {
            "dataset_kind": "sequential",
            "context_radius": 1,
            "windows": "windows.jsonl",
        }
    )
    (root / "dataset_contract.yaml").write_text(
        yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8"
    )
    records = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    for frame_index, record in enumerate(records):
        record["sequence_id"] = "ordered-sequence"
        record["frame_index"] = frame_index
        record["split"] = "train"
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    window = {
        "window_id": "ordered-sequence-window-0001",
        "sequence_id": "ordered-sequence",
        "split": "train",
        "target_sample_id": records[1]["sample_id"],
        "target_frame_index": 1,
        "previous_sample_ids": [records[0]["sample_id"]],
        "future_sample_ids": [records[2]["sample_id"]],
    }
    (root / "windows.jsonl").write_text(json.dumps(window) + "\n", encoding="utf-8")

    valid_report = validate_dataset_contract(root, inspect_images=False)
    assert valid_report["status"] == "ok"
    assert valid_report["window_count"] == 1

    window["future_sample_ids"] = [records[0]["sample_id"]]
    (root / "windows.jsonl").write_text(json.dumps(window) + "\n", encoding="utf-8")
    invalid_report = validate_dataset_contract(root, inspect_images=False)
    assert invalid_report["status"] == "failed"
    assert any("frame order" in error for error in invalid_report["errors"])


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_segmentation_dataset_prefers_contract_records(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    dataset = DataSimulationSegmentationDataset(
        dataset_root=root,
        split="val",
        image_channel="topo",
        input_size=12,
    )
    sample = dataset[0]
    assert sample["sequence_id"] == "static-00001"
    assert sample["frame_index"] == 0
    assert sample["beam_regime"] == "mid"
    assert sample["emission_b_mode"] == "pooled_constant_fit"
    assert sample["image"].shape == (1, 12, 12)
    assert int(sample["mask"].sum()) == 16


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_segmentation_dataset_all_split_preserves_original_split_labels(tmp_path):
    root = _write_dataset(tmp_path / "dataset")
    dataset = DataSimulationSegmentationDataset(
        dataset_root=root,
        split="all",
        image_channel="topo",
        input_size=12,
    )
    assert len(dataset) == 3
    assert [dataset[index]["split"] for index in range(3)] == ["train", "val", "test"]
