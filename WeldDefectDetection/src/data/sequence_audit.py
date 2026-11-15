from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .normalization_audit import check_frozen_normalization

from .dataset_contract import load_contract_dataset, validate_dataset_contract


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_sequence_dataset(
    dataset_root: str | Path,
    *,
    compare_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    contract = validate_dataset_contract(root, inspect_images=True)
    if contract["status"] != "ok":
        return {**contract, "contract_status": contract["status"]}
    descriptor, records = load_contract_dataset(root)
    if any(not (row.get("raw_channels") or {}).get("topo") for row in records):
        return {"status": "failed", "errors": ["Raw TOPO channels are required for this audit."], "warnings": []}
    windows = _jsonl(root / str(descriptor["windows"]))
    sequence_manifest_path = root / str(
        descriptor.get("sequence_manifest", "sequence_manifest.json")
    )
    sequence_rows = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
    record_map = {str(row["sample_id"]): row for row in records}
    hashes: dict[str, list[str]] = defaultdict(list)
    dynamic_ranges: list[float] = []
    for record in records:
        raw_path = root / record["raw_channels"]["topo"]
        raw_bytes = raw_path.read_bytes()
        hashes[hashlib.sha256(raw_bytes).hexdigest()].append(str(record["sample_id"]))
        raw = np.load(raw_path, allow_pickle=False)
        dynamic_ranges.append(float(np.max(raw) - np.min(raw)))
    duplicate_groups = [ids for ids in hashes.values() if len(ids) > 1]
    sequences_by_seed: dict[int, set[str]] = defaultdict(set)
    for record in records:
        sequences_by_seed[int(record["seed"])].add(str(record["split"]))
    seed_leakage = {str(seed): sorted(splits) for seed, splits in sequences_by_seed.items() if len(splits) > 1}
    sequence_seed_counts = Counter(int(row["seed"]) for row in sequence_rows)
    duplicate_sequence_seeds = sorted(
        seed for seed, count in sequence_seed_counts.items() if count > 1
    )
    planned_target_counts: Counter[int] = Counter()
    planned_target_visibility_errors: list[str] = []
    planned_defect_consistency_errors: list[str] = []
    multiple_defect_sequence_count = 0
    records_by_sequence: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        records_by_sequence[str(record["sequence_id"])][int(record["frame_index"])] = record
    for row in sequence_rows:
        sequence_id = str(row["sequence_id"])
        planned_class = str(row.get("planned_class", "clean"))
        planned_types = [str(value) for value in row.get("planned_defect_types", [])]
        if len(planned_types) > 1:
            multiple_defect_sequence_count += 1
        visible_types = {
            str(value)
            for record in records_by_sequence[sequence_id].values()
            for value in record.get("defect_types", [])
        }
        missing_types = sorted(set(planned_types) - visible_types)
        if missing_types:
            planned_defect_consistency_errors.append(
                f"{sequence_id}: planned defect types not visible: {', '.join(missing_types)}."
            )
        planned_target = row.get("planned_target_frame_index")
        if planned_class == "clean":
            if any(bool(record["is_defective"]) for record in records_by_sequence[sequence_id].values()):
                planned_target_visibility_errors.append(
                    f"{sequence_id}: clean sequence contains a defective frame."
                )
            continue
        if not isinstance(planned_target, int):
            planned_target_visibility_errors.append(
                f"{sequence_id}: defective sequence has no integer planned target frame."
            )
            continue
        planned_target_counts[planned_target] += 1
        target_record = records_by_sequence[sequence_id].get(planned_target)
        if target_record is None or not bool(target_record["is_defective"]):
            planned_target_visibility_errors.append(
                f"{sequence_id}: planned target frame {planned_target} is not visibly defective."
            )
    context_radius = int(descriptor.get("context_radius", 0))
    frames_per_sequence = int(descriptor.get("frames_per_sequence", 0))
    eligible_targets = set(range(context_radius, frames_per_sequence - context_radius))
    missing_planned_targets = sorted(eligible_targets - set(planned_target_counts))
    composition: Counter[str] = Counter()
    overlap_fractions: list[float] = []
    for window in windows:
        target = record_map[str(window["target_sample_id"])]
        context_ids = [*window["previous_sample_ids"], *window["future_sample_ids"]]
        context_defective = any(bool(record_map[str(sample_id)]["is_defective"]) for sample_id in context_ids)
        target_defective = bool(target["is_defective"])
        composition[f"target_{'defective' if target_defective else 'clean'}__context_{'defective' if context_defective else 'clean'}"] += 1
        target_start = int(target.get("crop_x0_px", 0))
        target_width = int(np.load(root / target["raw_channels"]["topo"], mmap_mode="r").shape[1])
        target_pixels: set[int] = set(range(target_start, target_start + target_width))
        context_pixels: set[int] = set()
        for sample_id in context_ids:
            context_record = record_map[str(sample_id)]
            context_start = int(context_record.get("crop_x0_px", 0))
            context_pixels.update(range(context_start, context_start + target_width))
        overlap_fractions.append(len(target_pixels & context_pixels) / target_width)
    comparison: dict[str, Any] = {}
    own_hashes = set(hashes)
    own_seeds = {int(row["seed"]) for row in records}
    for compare_root in compare_roots or []:
        other = Path(compare_root).expanduser().resolve()
        _, other_records = load_contract_dataset(other)
        other_hashes = {
            hashlib.sha256((other / row["raw_channels"]["topo"]).read_bytes()).hexdigest()
            for row in other_records
        }
        other_seeds = {int(row["seed"]) for row in other_records}
        comparison[str(other)] = {
            "shared_raw_sha256_count": len(own_hashes & other_hashes),
            "shared_seed_count": len(own_seeds & other_seeds),
        }
    errors = list(contract["errors"])
    if duplicate_groups:
        errors.append(f"Raw TOPO duplicate content found in {len(duplicate_groups)} hash groups.")
    if seed_leakage:
        errors.append(f"Seed values cross dataset splits: {len(seed_leakage)} seeds.")
    if duplicate_sequence_seeds:
        errors.append(
            f"Sequence seeds are duplicated: {len(duplicate_sequence_seeds)} duplicate values."
        )
    if missing_planned_targets:
        errors.append(
            "Defective planned targets do not cover eligible frames: "
            + ", ".join(str(value) for value in missing_planned_targets)
            + "."
        )
    errors.extend(planned_target_visibility_errors)
    errors.extend(planned_defect_consistency_errors)
    for compare_root, result in comparison.items():
        if result["shared_raw_sha256_count"]:
            errors.append(
                f"Dataset shares {result['shared_raw_sha256_count']} raw TOPO hashes "
                f"with comparison root {compare_root}."
            )
        if result["shared_seed_count"]:
            errors.append(
                f"Dataset shares {result['shared_seed_count']} sample seeds "
                f"with comparison root {compare_root}."
            )
    if compare_roots and (
        not isinstance(contract.get("normalization"), dict)
        or contract["normalization"].get("fit_performed_on_this_dataset") is not False
    ):
        errors.append(
            "A compared confirmation dataset must use a frozen external training normalization."
        )
    warnings = list(contract["warnings"])
    maximum_overlap = max(overlap_fractions, default=0.0)
    if maximum_overlap > 0.0:
        errors.append(
            "Context crops contain target source pixels; this creates reconstruction target leakage."
        )
    if compare_roots:
        errors.extend(check_frozen_normalization(root, compare_roots))
    return {
        "status": "ok" if not errors else "failed",
        "dataset_root": str(root),
        "contract_status": contract["status"],
        "record_count": len(records),
        "window_count": len(windows),
        "unique_raw_sha256_count": len(hashes),
        "duplicate_raw_hash_groups": duplicate_groups,
        "zero_dynamic_range_count": sum(value <= 1e-12 for value in dynamic_ranges),
        "minimum_raw_dynamic_range": min(dynamic_ranges),
        "sequence_seed_split_leakage": seed_leakage,
        "duplicate_sequence_seeds": duplicate_sequence_seeds,
        "eligible_target_frame_indices": sorted(eligible_targets),
        "planned_defect_target_frame_counts": {
            str(key): value for key, value in sorted(planned_target_counts.items())
        },
        "missing_planned_target_frame_indices": missing_planned_targets,
        "planned_target_visibility_errors": planned_target_visibility_errors,
        "planned_defect_consistency_errors": planned_defect_consistency_errors,
        "multiple_defect_sequence_count": multiple_defect_sequence_count,
        "window_defect_composition": dict(sorted(composition.items())),
        "mean_target_pixel_context_overlap_fraction": float(np.mean(overlap_fractions)),
        "maximum_target_pixel_context_overlap_fraction": maximum_overlap,
        "comparison": comparison,
        "normalization": contract.get("normalization"),
        "errors": errors,
        "warnings": warnings,
    }
