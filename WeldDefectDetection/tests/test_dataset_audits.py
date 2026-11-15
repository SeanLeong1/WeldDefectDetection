from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.sequence_audit import audit_sequence_dataset
from src.data.static_audit import audit_static_dataset
from src.data.paired_static_audit import audit_paired_static_datasets


def _record(
    root: Path,
    *,
    sample_id: str,
    sequence_id: str,
    frame_index: int,
    split: str,
    seed: int,
    value: float,
    crop_x0_px: int = 0,
) -> dict:
    raw_relative = Path("raw") / split / f"{sample_id}.npy"
    image_relative = Path("images") / split / f"{sample_id}.png"
    mask_relative = Path("masks") / split / f"{sample_id}.png"
    for relative in (raw_relative, image_relative, mask_relative):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    raw = np.arange(16, dtype=np.float32).reshape(4, 4) + value
    np.save(root / raw_relative, raw, allow_pickle=False)
    cv2.imwrite(str(root / image_relative), np.clip(raw, 0, 255).astype(np.uint8))
    cv2.imwrite(str(root / mask_relative), np.zeros((4, 4), dtype=np.uint8))
    return {
        "contract_version": "weld-topo-segmentation/v2",
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "frame_index": frame_index,
        "split": split,
        "channels": {"topo": image_relative.as_posix()},
        "raw_channels": {"topo": raw_relative.as_posix()},
        "mask_path": mask_relative.as_posix(),
        "defect_type": "clean",
        "defect_types": [],
        "is_defective": False,
        "seed": seed,
        "beam_regime": "mid",
        "emission_b_mode": "lambertian",
        "emission_b_value": 0.0,
        "render_view_mode": "top_surface",
        "pixel_size_mm": 0.05,
        "noise_enabled": True,
        "crop_x0_px": crop_x0_px,
    }


def _write_static(root: Path, *, seed_offset: int, value_offset: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "contract_version": "weld-topo-segmentation/v2",
        "dataset_kind": "static_balanced",
        "records": "samples.jsonl",
        "normalization": "topo_normalization.json",
        "noise_policy": "development",
    }
    records = [
        _record(
            root,
            sample_id=f"sample-{index}",
            sequence_id=f"static-{index}",
            frame_index=0,
            split=split,
            seed=seed_offset + index,
            value=value_offset + 100.0 * index,
        )
        for index, split in enumerate(("train", "val", "test"))
    ]
    (root / "dataset_contract.yaml").write_text(
        yaml.safe_dump(descriptor), encoding="utf-8"
    )
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (root / "topo_normalization.json").write_text(
        json.dumps(
            {
                "source_split": "train",
                "raw_low": 0.0,
                "raw_high": 1.0,
                "fit_performed_on_this_dataset": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_sequence(root: Path, *, seed: int, value_offset: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "contract_version": "weld-topo-segmentation/v2",
        "dataset_kind": "sequential",
        "records": "samples.jsonl",
        "windows": "windows.jsonl",
        "sequence_manifest": "sequence_manifest.json",
        "normalization": "topo_normalization.json",
        "noise_policy": "development",
        "frames_per_sequence": 3,
        "context_radius": 1,
    }
    records = [
        _record(
            root,
            sample_id=f"sequence-0-frame-{index}",
            sequence_id="sequence-0",
            frame_index=index,
            split="train",
            seed=seed,
            value=value_offset + 100.0 * index,
            crop_x0_px=4 * index,
        )
        for index in range(3)
    ]
    (root / "dataset_contract.yaml").write_text(
        yaml.safe_dump(descriptor), encoding="utf-8"
    )
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (root / "windows.jsonl").write_text(
        json.dumps(
            {
                "window_id": "sequence-0-window-1",
                "sequence_id": "sequence-0",
                "split": "train",
                "target_sample_id": records[1]["sample_id"],
                "target_frame_index": 1,
                "previous_sample_ids": [records[0]["sample_id"]],
                "future_sample_ids": [records[2]["sample_id"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "sequence_manifest.json").write_text(
        json.dumps(
            [
                {
                    "sequence_id": "sequence-0",
                    "planned_class": "clean",
                    "planned_defect_types": [],
                    "planned_target_frame_index": None,
                    "seed": seed,
                }
            ]
        ),
        encoding="utf-8",
    )
    (root / "topo_normalization.json").write_text(
        json.dumps(
            {
                "source_split": "train",
                "raw_low": 0.0,
                "raw_high": 1.0,
                "fit_performed_on_this_dataset": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_static_audit_accepts_independent_confirmation(tmp_path: Path) -> None:
    development = _write_static(tmp_path / "development", seed_offset=10, value_offset=0.0)
    confirmation = _write_static(tmp_path / "confirmation", seed_offset=100, value_offset=1000.0)
    import hashlib
    source = development / "topo_normalization.json"
    reference = json.loads(source.read_text())
    reference["fit_performed_on_this_dataset"] = True
    source.write_text(json.dumps(reference))
    (confirmation / "topo_normalization.json").write_text(json.dumps({
        **reference, "fit_performed_on_this_dataset": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
    report = audit_static_dataset(confirmation, compare_roots=[development])
    assert report["status"] == "ok"
    assert report["unique_seed_count"] == 3
    assert report["unique_raw_sha256_count"] == 3
    assert next(iter(report["comparison"].values())) == {
        "shared_raw_sha256_count": 0,
        "shared_seed_count": 0,
    }


def test_static_audit_rejects_shared_seed_and_content(tmp_path: Path) -> None:
    development = _write_static(tmp_path / "development", seed_offset=10, value_offset=0.0)
    confirmation = _write_static(tmp_path / "confirmation", seed_offset=10, value_offset=0.0)
    report = audit_static_dataset(confirmation, compare_roots=[development])
    assert report["status"] == "failed"
    assert any("shares 3 raw TOPO hashes" in error for error in report["errors"])
    assert any("shares 3 sample seeds" in error for error in report["errors"])


def test_sequence_audit_rejects_shared_seed_and_content(tmp_path: Path) -> None:
    development = _write_sequence(tmp_path / "development", seed=10, value_offset=0.0)
    confirmation = _write_sequence(tmp_path / "confirmation", seed=10, value_offset=0.0)
    report = audit_sequence_dataset(confirmation, compare_roots=[development])
    assert report["status"] == "failed"
    assert any("shares 3 raw TOPO hashes" in error for error in report["errors"])
    assert any("shares 1 sample seeds" in error for error in report["errors"])


def test_paired_static_audit_accepts_same_cases_with_changed_images(tmp_path: Path) -> None:
    reference = _write_static(tmp_path / "reference", seed_offset=10, value_offset=0.0)
    variant = _write_static(tmp_path / "variant", seed_offset=10, value_offset=1000.0)
    report = audit_paired_static_datasets(reference, variant)
    assert report["status"] == "ok"
    assert report["paired_record_count"] == 3
    assert report["changed_raw_topo_count"] == 3
    assert report["mask_mismatches"] == []


def test_paired_static_audit_rejects_case_identity_change(tmp_path: Path) -> None:
    reference = _write_static(tmp_path / "reference", seed_offset=10, value_offset=0.0)
    variant = _write_static(tmp_path / "variant", seed_offset=20, value_offset=1000.0)
    report = audit_paired_static_datasets(reference, variant)
    assert report["status"] == "failed"
    assert any("identity fields differ" in error for error in report["errors"])
