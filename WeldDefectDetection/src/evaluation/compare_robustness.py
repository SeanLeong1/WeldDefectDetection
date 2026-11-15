from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.file_io import write_json


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return payload


def _delta(reference: dict[str, Any], variant: dict[str, Any], key: str) -> float:
    return float(variant[key]) - float(reference[key])


def compare_robustness_evaluations(
    *,
    reference_summary: str | Path,
    variant_summary: str | Path,
    paired_audit: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    reference = _read_json(reference_summary)
    variant = _read_json(variant_summary)
    audit = _read_json(paired_audit)
    errors: list[str] = []
    if audit.get("status") != "ok":
        errors.append("Paired dataset audit is not successful.")
    if reference.get("status") != "ok" or variant.get("status") != "ok":
        errors.append("Both evaluation summaries must have status 'ok'.")
    if reference.get("algorithm") != variant.get("algorithm"):
        errors.append("Reference and variant evaluations use different algorithms.")

    reference_identity = reference.get("model_identity")
    variant_identity = variant.get("model_identity")
    if not isinstance(reference_identity, dict) or not isinstance(variant_identity, dict):
        errors.append("Evaluation summaries must include model_identity.")
    elif reference_identity != variant_identity:
        errors.append("Reference and variant evaluations do not use the same frozen model identity.")

    reference_classes = reference.get("per_class", {})
    variant_classes = variant.get("per_class", {})
    if set(reference_classes) != set(variant_classes):
        errors.append("Reference and variant class sets differ.")

    if errors:
        result = {
            "status": "failed",
            "errors": errors,
            "reference_summary": str(Path(reference_summary).resolve()),
            "variant_summary": str(Path(variant_summary).resolve()),
            "paired_audit": str(Path(paired_audit).resolve()),
        }
        write_json(output_path, result)
        return result

    class_deltas = {
        class_name: {
            metric: _delta(reference_classes[class_name], variant_classes[class_name], metric)
            for metric in ("iou", "precision", "recall", "f1")
            if reference_classes[class_name].get(metric) is not None
            and variant_classes[class_name].get(metric) is not None
        }
        for class_name in sorted(reference_classes)
    }
    result = {
        "status": "ok",
        "comparison": "paired_frozen_model_robustness",
        "algorithm": reference["algorithm"],
        "reference_summary": str(Path(reference_summary).resolve()),
        "variant_summary": str(Path(variant_summary).resolve()),
        "paired_audit": str(Path(paired_audit).resolve()),
        "model_identity": reference_identity,
        "reference_dataset": reference.get("confirmation_independence_audit", {}).get(
            "dataset_root"
        ),
        "variant_dataset": variant.get("confirmation_independence_audit", {}).get(
            "dataset_root"
        ),
        "metric_deltas_variant_minus_reference": {
            metric: _delta(reference, variant, metric)
            for metric in ("iou", "precision", "recall", "f1")
        },
        "image_balanced_accuracy_delta": _delta(
            reference["image_classification"],
            variant["image_classification"],
            "balanced_accuracy",
        ),
        "per_class_deltas": class_deltas,
        "errors": [],
    }
    write_json(output_path, result)
    return result
