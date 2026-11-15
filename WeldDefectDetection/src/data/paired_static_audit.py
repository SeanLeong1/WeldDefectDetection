from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset_contract import load_contract_dataset, validate_dataset_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization_signature(root: Path, descriptor: dict[str, Any]) -> dict[str, Any] | None:
    name = descriptor.get("normalization")
    if not name:
        return None
    payload = json.loads((root / str(name)).read_text(encoding="utf-8"))
    return {
        "raw_low": payload.get("raw_low"),
        "raw_high": payload.get("raw_high"),
        "tone_curve": payload.get("tone_curve"),
        "source_sha256": payload.get("source_sha256"),
    }


def audit_paired_static_datasets(
    reference_root: str | Path,
    variant_root: str | Path,
) -> dict[str, Any]:
    """Verify that two static datasets differ only in image-formation conditions."""
    reference = Path(reference_root).expanduser().resolve()
    variant = Path(variant_root).expanduser().resolve()
    reference_contract = validate_dataset_contract(reference, inspect_images=True)
    variant_contract = validate_dataset_contract(variant, inspect_images=True)
    reference_descriptor, reference_records = load_contract_dataset(reference)
    variant_descriptor, variant_records = load_contract_dataset(variant)
    errors = [
        *[f"reference: {error}" for error in reference_contract["errors"]],
        *[f"variant: {error}" for error in variant_contract["errors"]],
    ]
    warnings = [
        *[f"reference: {warning}" for warning in reference_contract["warnings"]],
        *[f"variant: {warning}" for warning in variant_contract["warnings"]],
    ]
    if not str(reference_descriptor.get("dataset_kind", "")).startswith("static"):
        errors.append("Reference dataset is not static.")
    if not str(variant_descriptor.get("dataset_kind", "")).startswith("static"):
        errors.append("Variant dataset is not static.")

    reference_map = {str(record["sample_id"]): record for record in reference_records}
    variant_map = {str(record["sample_id"]): record for record in variant_records}
    missing_from_variant = sorted(set(reference_map) - set(variant_map))
    extra_in_variant = sorted(set(variant_map) - set(reference_map))
    if missing_from_variant:
        errors.append(f"Variant is missing {len(missing_from_variant)} reference sample IDs.")
    if extra_in_variant:
        errors.append(f"Variant has {len(extra_in_variant)} unexpected sample IDs.")

    field_mismatches: list[dict[str, Any]] = []
    mask_mismatches: list[str] = []
    identical_raw_count = 0
    changed_raw_count = 0
    identity_fields = (
        "sequence_id",
        "frame_index",
        "split",
        "defect_type",
        "defect_types",
        "beam_regime",
        "pixel_size_mm",
        "planned_defect_parameters",
        "realized_non_noise_parameters",
        "renderer_parameters",
        "seed",
        "generation_attempt",
    )
    for sample_id in sorted(set(reference_map) & set(variant_map)):
        left = reference_map[sample_id]
        right = variant_map[sample_id]
        left_seed = left.get("base_seed", left.get("seed"))
        right_seed = right.get("base_seed", right.get("seed"))
        if left_seed != right_seed:
            field_mismatches.append(
                {"sample_id": sample_id, "field": "base_seed", "reference": left_seed, "variant": right_seed}
            )
        for field in identity_fields:
            if field == "planned_defect_parameters" and (
                field not in left or field not in right
            ):
                continue
            if left.get(field) != right.get(field):
                field_mismatches.append(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        "reference": left.get(field),
                        "variant": right.get(field),
                    }
                )
        if _sha256(reference / str(left["mask_path"])) != _sha256(
            variant / str(right["mask_path"])
        ):
            mask_mismatches.append(sample_id)
        raw_equal = _sha256(reference / str(left["raw_channels"]["topo"])) == _sha256(
            variant / str(right["raw_channels"]["topo"])
        )
        identical_raw_count += int(raw_equal)
        changed_raw_count += int(not raw_equal)

    if field_mismatches:
        errors.append(f"Paired identity fields differ in {len(field_mismatches)} comparisons.")
    if mask_mismatches:
        errors.append(f"Paired masks differ for {len(mask_mismatches)} samples.")

    reference_normalization = _normalization_signature(reference, reference_descriptor)
    variant_normalization = _normalization_signature(variant, variant_descriptor)
    if reference_normalization != variant_normalization:
        errors.append("Paired datasets do not use the same normalization mapping.")
    if changed_raw_count == 0 and reference_records:
        warnings.append("All paired raw TOPO arrays are identical; the scenario changed no image data.")
    if any("realized_non_noise_parameters" not in record for record in reference_records):
        warnings.append("Historical records lack complete realized parameters; paired identity is partial.")
    if any("planned_defect_parameters" not in record for record in reference_records):
        warnings.append(
            "Reference records predate planned_defect_parameters; mask equality is the "
            "authoritative paired-geometry check."
        )

    return {
        "status": "ok" if not errors else "failed",
        "reference_root": str(reference),
        "variant_root": str(variant),
        "reference_record_count": len(reference_records),
        "variant_record_count": len(variant_records),
        "paired_record_count": len(set(reference_map) & set(variant_map)),
        "reference_noise_profiles": sorted(
            {str(record.get("noise_profile", "unknown")) for record in reference_records}
        ),
        "variant_noise_profiles": sorted(
            {str(record.get("noise_profile", "unknown")) for record in variant_records}
        ),
        "identical_raw_topo_count": identical_raw_count,
        "changed_raw_topo_count": changed_raw_count,
        "field_mismatches": field_mismatches,
        "mask_mismatches": mask_mismatches,
        "reference_normalization": reference_normalization,
        "variant_normalization": variant_normalization,
        "errors": errors,
        "warnings": warnings,
    }
