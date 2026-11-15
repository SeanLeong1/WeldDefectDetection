from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.paired_static_audit import audit_paired_static_datasets
from src.data.sequence_audit import audit_sequence_dataset
from src.data.static_audit import audit_static_dataset
from src.utils.file_io import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_check(function, *args, **kwargs) -> dict[str, Any]:
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        return {
            "status": "failed",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }


def verify_completion_manifest(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, dict):
        raise ValueError("Completion manifest must contain a YAML mapping.")
    base = manifest_file.parent
    errors: list[str] = []
    checks: dict[str, Any] = {}

    datasets = manifest.get("datasets") or {}
    static = datasets.get("static") or {}
    sequence = datasets.get("sequence") or {}
    static_development = _resolve(base, static["development"])
    static_confirmation = _resolve(base, static["confirmation"])
    sequence_development = _resolve(base, sequence["development"])
    sequence_confirmation = _resolve(base, sequence["confirmation"])

    static_development_audit = _safe_check(audit_static_dataset, static_development)
    static_confirmation_audit = _safe_check(
        audit_static_dataset,
        static_confirmation, compare_roots=[static_development]
    )
    sequence_development_audit = _safe_check(
        audit_sequence_dataset, sequence_development
    )
    sequence_confirmation_audit = _safe_check(
        audit_sequence_dataset,
        sequence_confirmation, compare_roots=[sequence_development]
    )
    checks["static_development"] = static_development_audit
    checks["static_confirmation"] = static_confirmation_audit
    checks["sequence_development"] = sequence_development_audit
    checks["sequence_confirmation"] = sequence_confirmation_audit
    for name, audit in checks.copy().items():
        if audit.get("status") != "ok":
            errors.append(f"{name} audit failed.")

    paired_checks: dict[str, Any] = {}
    for row in static.get("paired_variants", []):
        name = str(row["name"])
        variant = _resolve(base, row["path"])
        audit = _safe_check(
            audit_paired_static_datasets, static_confirmation, variant
        )
        paired_checks[name] = audit
        if audit.get("status") != "ok":
            errors.append(f"Paired static variant '{name}' failed its identity audit.")
    checks["paired_static_variants"] = paired_checks

    checkpoint_checks: dict[str, Any] = {}
    for row in manifest.get("checkpoints", []):
        name = str(row["name"])
        path = _resolve(base, row["path"])
        expected = str(row["sha256"]).lower()
        if not path.is_file():
            checkpoint_checks[name] = {"status": "failed", "path": str(path)}
            errors.append(f"Checkpoint '{name}' is missing: {path}")
            continue
        actual = _sha256(path)
        status = "ok" if actual == expected else "failed"
        checkpoint_checks[name] = {
            "status": status,
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
        if status != "ok":
            errors.append(f"Checkpoint '{name}' SHA-256 does not match the manifest.")
    checks["checkpoints"] = checkpoint_checks

    evaluation_checks: dict[str, Any] = {}
    for row in manifest.get("evaluations", []):
        name = str(row["name"])
        path = _resolve(base, row["summary"])
        result: dict[str, Any] = {"path": str(path), "status": "ok", "errors": []}
        if not path.is_file():
            result["status"] = "failed"
            result["errors"].append("summary is missing")
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                payload = {}
                result["errors"].append(f"summary is not readable JSON: {exc}")
            if payload.get("status") != "ok":
                result["errors"].append("summary status is not ok")
            expected_algorithm = row.get("algorithm")
            if expected_algorithm and payload.get("algorithm") != expected_algorithm:
                result["errors"].append("algorithm does not match")
            expected_method = row.get("method")
            if expected_method and payload.get("method") != expected_method:
                result["errors"].append("method does not match")
            expected_dataset = row.get("dataset")
            if expected_dataset:
                expected_dataset_path = str(_resolve(base, expected_dataset))
                observed_dataset = payload.get("dataset_root") or (
                    payload.get("confirmation_independence_audit") or {}
                ).get("dataset_root")
                if observed_dataset is None or str(Path(observed_dataset).resolve()) != expected_dataset_path:
                    result["errors"].append("dataset root does not match")
            expected_checkpoint = row.get("checkpoint_sha256")
            if expected_checkpoint:
                observed_checkpoint = (payload.get("model_identity") or {}).get(
                    "checkpoint_sha256"
                ) or payload.get("checkpoint_sha256")
                if str(observed_checkpoint).lower() != str(expected_checkpoint).lower():
                    result["errors"].append("checkpoint SHA-256 does not match")
            if row.get("requires_confirmation_audit") and (
                payload.get("confirmation_independence_audit") or {}
            ).get("status") != "ok":
                result["errors"].append("embedded confirmation audit is not ok")
            expected_frozen = row.get("frozen_parameters")
            if expected_frozen is not None:
                observed_frozen = payload.get("frozen_parameters") or payload.get(
                    "selected_parameters"
                )
                if observed_frozen != expected_frozen:
                    result["errors"].append("frozen parameters do not match")
            if result["errors"]:
                result["status"] = "failed"
        evaluation_checks[name] = result
        if result["status"] != "ok":
            errors.append(f"Evaluation '{name}' failed manifest verification.")
    checks["evaluations"] = evaluation_checks

    result = {
        "status": "ok" if not errors else "failed",
        "manifest": str(manifest_file),
        "scope": manifest.get("scope"),
        "checks": checks,
        "errors": errors,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
