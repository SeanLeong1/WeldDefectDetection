from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.compare_robustness import compare_robustness_evaluations


def _summary(path: Path, *, algorithm: str = "cnn", checkpoint_hash: str = "abc") -> Path:
    payload = {
        "status": "ok",
        "algorithm": algorithm,
        "iou": 0.2,
        "precision": 0.3,
        "recall": 0.4,
        "f1": 0.34,
        "image_classification": {"balanced_accuracy": 0.6},
        "per_class": {
            "clean": {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "pore": {"iou": 0.2, "precision": 0.3, "recall": 0.4, "f1": 0.34},
        },
        "model_identity": {
            "algorithm": algorithm,
            "checkpoint_sha256": checkpoint_hash,
            "threshold": 0.5,
        },
        "confirmation_independence_audit": {"dataset_root": str(path.parent)},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_robustness_comparison_requires_same_frozen_model(tmp_path: Path) -> None:
    reference = _summary(tmp_path / "reference.json", checkpoint_hash="one")
    variant = _summary(tmp_path / "variant.json", checkpoint_hash="two")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    result = compare_robustness_evaluations(
        reference_summary=reference,
        variant_summary=variant,
        paired_audit=audit,
        output_path=tmp_path / "result.json",
    )
    assert result["status"] == "failed"
    assert any("same frozen model" in error for error in result["errors"])


def test_robustness_comparison_calculates_variant_deltas(tmp_path: Path) -> None:
    reference = _summary(tmp_path / "reference.json")
    variant = _summary(tmp_path / "variant.json")
    payload = json.loads(variant.read_text(encoding="utf-8"))
    payload["iou"] = 0.25
    payload["image_classification"]["balanced_accuracy"] = 0.55
    payload["per_class"]["pore"]["iou"] = 0.3
    variant.write_text(json.dumps(payload), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    result = compare_robustness_evaluations(
        reference_summary=reference,
        variant_summary=variant,
        paired_audit=audit,
        output_path=tmp_path / "result.json",
    )
    assert result["status"] == "ok"
    assert result["metric_deltas_variant_minus_reference"]["iou"] == pytest.approx(0.05)
    assert result["image_balanced_accuracy_delta"] == pytest.approx(-0.05)
    assert result["per_class_deltas"]["pore"]["iou"] == pytest.approx(0.1)
