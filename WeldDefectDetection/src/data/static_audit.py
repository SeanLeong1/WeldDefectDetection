from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .normalization_audit import check_frozen_normalization

from .dataset_contract import load_contract_dataset, validate_dataset_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_hashes(root: Path, records: list[dict[str, Any]]) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        raw_path = root / str(record["raw_channels"]["topo"])
        hashes[_sha256(raw_path)].append(str(record["sample_id"]))
    return hashes


def audit_static_dataset(
    dataset_root: str | Path,
    *,
    compare_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Audit a static dataset and fail on development/confirmation overlap."""
    root = Path(dataset_root).expanduser().resolve()
    contract = validate_dataset_contract(root, inspect_images=True)
    if contract["status"] != "ok":
        return {**contract, "contract_status": contract["status"]}
    descriptor, records = load_contract_dataset(root)
    if any(not (row.get("raw_channels") or {}).get("topo") for row in records):
        return {"status": "failed", "errors": ["Raw TOPO channels are required for this audit."], "warnings": []}
    errors = list(contract["errors"])
    warnings = list(contract["warnings"])

    if not str(descriptor.get("dataset_kind", "")).startswith("static"):
        errors.append("Static audit requires a static dataset contract.")

    hashes = _raw_hashes(root, records)
    duplicate_hash_groups = [sample_ids for sample_ids in hashes.values() if len(sample_ids) > 1]
    if duplicate_hash_groups:
        errors.append(
            f"Raw TOPO duplicate content found in {len(duplicate_hash_groups)} hash groups."
        )

    seed_counts = Counter(int(record["seed"]) for record in records)
    duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if count > 1)
    if duplicate_seeds:
        errors.append(f"Static sample seeds are duplicated: {len(duplicate_seeds)} values.")

    dynamic_ranges: list[float] = []
    for record in records:
        raw_path = root / str(record["raw_channels"]["topo"])
        raw = np.load(raw_path, allow_pickle=False)
        dynamic_ranges.append(float(np.max(raw) - np.min(raw)))

    own_hashes = set(hashes)
    own_seeds = set(seed_counts)
    comparison: dict[str, Any] = {}
    for compare_root in compare_roots or []:
        other = Path(compare_root).expanduser().resolve()
        other_descriptor, other_records = load_contract_dataset(other)
        if not str(other_descriptor.get("dataset_kind", "")).startswith("static"):
            errors.append(f"Comparison dataset is not static: {other}")
            continue
        other_hashes = set(_raw_hashes(other, other_records))
        other_seeds = {int(record["seed"]) for record in other_records}
        shared_hashes = len(own_hashes & other_hashes)
        shared_seeds = len(own_seeds & other_seeds)
        comparison[str(other)] = {
            "shared_raw_sha256_count": shared_hashes,
            "shared_seed_count": shared_seeds,
        }
        if shared_hashes:
            errors.append(
                f"Dataset shares {shared_hashes} raw TOPO hashes with comparison root {other}."
            )
        if shared_seeds:
            errors.append(
                f"Dataset shares {shared_seeds} sample seeds with comparison root {other}."
            )

    normalization = contract.get("normalization")
    if compare_roots:
        if not isinstance(normalization, dict) or normalization.get(
            "fit_performed_on_this_dataset"
        ) is not False:
            errors.append(
                "A compared confirmation dataset must use a frozen external training normalization."
            )

    if compare_roots:
        errors.extend(check_frozen_normalization(root, compare_roots))
    return {
        "status": "ok" if not errors else "failed",
        "dataset_root": str(root),
        "contract_status": contract["status"],
        "record_count": len(records),
        "split_counts": contract.get("split_counts", {}),
        "clean_count": contract.get("clean_count", 0),
        "defective_count": contract.get("defective_count", 0),
        "defect_instance_counts": contract.get("defect_instance_counts", {}),
        "unique_seed_count": len(seed_counts),
        "duplicate_seeds": duplicate_seeds,
        "unique_raw_sha256_count": len(hashes),
        "duplicate_raw_hash_groups": duplicate_hash_groups,
        "zero_dynamic_range_count": sum(value <= 1e-12 for value in dynamic_ranges),
        "minimum_raw_dynamic_range": min(dynamic_ranges, default=None),
        "comparison": comparison,
        "normalization": normalization,
        "errors": errors,
        "warnings": warnings,
    }
