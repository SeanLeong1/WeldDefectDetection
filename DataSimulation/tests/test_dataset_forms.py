from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from bse_renderer.domain_randomisation.dataset_forms import (
    DATASET_CLASSES,
    _planned_target_frame,
    _regime_for_sample,
    generate_balanced_topo_dataset,
    generate_topo_sequence_dataset,
)


def _records(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_beam_regime_cycle_is_offset_between_classes():
    assert [_regime_for_sample(index, 0) for index in range(3)] == ["low", "mid", "high"]
    assert [_regime_for_sample(index, 1) for index in range(3)] == ["mid", "high", "low"]


def test_planned_sequence_targets_cycle_across_all_eligible_frames():
    targets = {
        _planned_target_frame(
            repetition,
            class_index=0,
            frames_per_sequence=9,
            context_radius=2,
        )
        for repetition in range(5)
    }
    assert targets == {2, 3, 4, 5, 6}


def test_balanced_static_topo_dataset_contract(tmp_path):
    out = generate_balanced_topo_dataset(
        config_dir="config",
        output_dir=tmp_path / "static",
        samples_per_class=3,
        image_size=64,
        seed=31,
    )
    records = _records(out["records"])
    assert out["record_count"] == 3 * len(DATASET_CLASSES)
    assert set(out["class_counts"]) == set(DATASET_CLASSES)
    assert set(out["class_counts"].values()) == {3}
    assert all(set(record["channels"]) == {"topo"} for record in records)
    assert all(set(record["raw_channels"]) == {"topo"} for record in records)
    assert all((Path(out["output_dir"]) / record["raw_channels"]["topo"]).exists() for record in records)
    assert all(
        float(np.ptp(np.load(Path(out["output_dir"]) / record["raw_channels"]["topo"]))) > 0.0
        for record in records
    )
    assert all(record["frame_index"] == 0 for record in records)
    assert len({record["sequence_id"] for record in records}) == len(records)
    assert not (Path(out["output_dir"]) / "images" / "compo").exists()
    descriptor = yaml.safe_load(Path(out["dataset_contract"]).read_text(encoding="utf-8"))
    assert descriptor["contract_version"] == "weld-topo-segmentation/v2"
    assert descriptor["dataset_kind"] == "static_balanced"
    assert descriptor["normalization"] == "topo_normalization.json"
    assert Path(out["normalization"]["path"]).exists()
    assert descriptor["noise_policy"] == "development"
    assert all(record["noise_enabled"] for record in records)
    provenance_path = Path(out["provenance"]["path"])
    assert provenance_path.exists()
    assert len(out["provenance"]["sha256"]) == 64
    assert out["provenance"]["renderer_source_hashes"]


def test_static_dataset_can_reuse_frozen_training_normalization(tmp_path):
    frozen = tmp_path / "training_dataset" / "topo_normalization.json"
    frozen.parent.mkdir()
    frozen.write_text(
        json.dumps(
            {
                "method": "training_split_global_percentile",
                "source_split": "train",
                "raw_low": 0.01,
                "raw_high": 0.25,
                "tone_curve": "sem_sqrt",
                "raw_format": "float32_npy",
            }
        ),
        encoding="utf-8",
    )
    out = generate_balanced_topo_dataset(
        config_dir="config",
        output_dir=tmp_path / "confirmation",
        samples_per_class=3,
        image_size=64,
        seed=931,
        normalization_path=frozen,
    )
    normalization = json.loads(
        (Path(out["output_dir"]) / "topo_normalization.json").read_text(encoding="utf-8")
    )
    assert normalization["raw_low"] == 0.01
    assert normalization["raw_high"] == 0.25
    assert normalization["fit_performed_on_this_dataset"] is False
    assert normalization["frozen_from_dataset"] == "training_dataset"


def test_static_dataset_refuses_nonempty_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        generate_balanced_topo_dataset(
            config_dir="config",
            output_dir=output,
            samples_per_class=3,
            image_size=64,
            seed=32,
        )
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "preserve me"


def test_static_dataset_rejects_invalid_normalization_before_writing(tmp_path):
    frozen = tmp_path / "invalid_normalization.json"
    frozen.write_text(
        json.dumps(
            {
                "source_split": "test",
                "raw_low": 0.2,
                "raw_high": 0.1,
                "tone_curve": "linear",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "must_remain_absent"
    with pytest.raises(ValueError, match="raw_high"):
        generate_balanced_topo_dataset(
            config_dir="config",
            output_dir=output,
            samples_per_class=3,
            image_size=64,
            seed=33,
            normalization_path=frozen,
        )
    assert not output.exists()


def test_ordered_topo_sequences_are_split_as_complete_groups(tmp_path):
    out = generate_topo_sequence_dataset(
        config_dir="config",
        output_dir=tmp_path / "sequences",
        sequences_per_class=3,
        frames_per_sequence=3,
        frame_size=64,
        stride_px=64,
        context_radius=1,
        seed=41,
    )
    records = _records(out["records"])
    expected_sequences = 3 * len(DATASET_CLASSES)
    assert out["sequence_count"] == expected_sequences
    assert out["record_count"] == expected_sequences * 3
    assert out["window_count"] == expected_sequences
    windows = _records(out["windows"])
    assert all(len(row["previous_sample_ids"]) == 1 for row in windows)
    assert all(len(row["future_sample_ids"]) == 1 for row in windows)
    by_sequence: dict[str, list[dict]] = {}
    for record in records:
        by_sequence.setdefault(record["sequence_id"], []).append(record)
        assert set(record["channels"]) == {"topo"}
        assert set(record["raw_channels"]) == {"topo"}
    for sequence_records in by_sequence.values():
        assert len({record["split"] for record in sequence_records}) == 1
        assert sorted(record["frame_index"] for record in sequence_records) == [0, 1, 2]
        assert all(record["generation_attempt"] >= 1 for record in sequence_records)
        assert all(record["raw_topo_dynamic_range"] > 0.0 for record in sequence_records)
    assert not (Path(out["output_dir"]) / "images" / "compo").exists()


def test_sequence_dataset_can_reuse_frozen_training_normalization(tmp_path):
    frozen = tmp_path / "training_sequences" / "topo_normalization.json"
    frozen.parent.mkdir()
    frozen.write_text(
        json.dumps(
            {
                "method": "training_split_global_percentile",
                "source_split": "train",
                "raw_low": 0.01,
                "raw_high": 0.25,
                "tone_curve": "sem_sqrt",
                "raw_format": "float32_npy",
            }
        ),
        encoding="utf-8",
    )
    out = generate_topo_sequence_dataset(
        config_dir="config",
        output_dir=tmp_path / "confirmation_sequences",
        sequences_per_class=3,
        frames_per_sequence=3,
        frame_size=64,
        stride_px=64,
        context_radius=1,
        seed=941,
        normalization_path=frozen,
    )
    normalization = json.loads(
        (Path(out["output_dir"]) / "topo_normalization.json").read_text(encoding="utf-8")
    )
    assert normalization["raw_low"] == 0.01
    assert normalization["raw_high"] == 0.25
    assert normalization["fit_performed_on_this_dataset"] is False
    assert normalization["frozen_from_dataset"] == "training_sequences"
