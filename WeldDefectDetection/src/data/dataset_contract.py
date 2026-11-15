from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


CONTRACT_VERSION = "weld-topo-segmentation/v2"
ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_BEAM_REGIMES = {"low", "mid", "high", "randomized"}
ALLOWED_RENDER_VIEWS = {"top_surface", "slice"}
REQUIRED_CHANNELS = {"topo"}
PROHIBITED_CHANNELS = {"compo"}
REQUIRED_FIELDS = {
    "contract_version",
    "sample_id",
    "sequence_id",
    "frame_index",
    "split",
    "channels",
    "mask_path",
    "defect_type",
    "defect_types",
    "is_defective",
    "seed",
    "beam_regime",
    "emission_b_mode",
    "emission_b_value",
    "render_view_mode",
    "pixel_size_mm",
    "noise_enabled",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Record at {path}:{line_number} must be a JSON object.")
        records.append(value)
    return records


def load_contract_dataset(dataset_root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(dataset_root).expanduser().resolve()
    descriptor_path = root / "dataset_contract.yaml"
    if not descriptor_path.exists():
        raise FileNotFoundError(f"Dataset contract descriptor not found: {descriptor_path}")
    descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8")) or {}
    if not isinstance(descriptor, dict):
        raise ValueError(f"Dataset contract descriptor must be a mapping: {descriptor_path}")
    records_name = str(descriptor.get("records", "samples.jsonl"))
    records_path = root / records_name
    if not records_path.exists():
        raise FileNotFoundError(f"Dataset contract records not found: {records_path}")
    return descriptor, _load_jsonl(records_path)


def resolve_record_paths(record: dict[str, Any], dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    resolved = dict(record)
    resolved["channels"] = {
        str(name): (root / str(path)).resolve()
        for name, path in (record.get("channels") or {}).items()
    }
    resolved["mask_path"] = (root / str(record.get("mask_path", ""))).resolve()
    resolved["raw_channels"] = {
        str(name): (root / str(path)).resolve()
        for name, path in (record.get("raw_channels") or {}).items()
    }
    return resolved


def _read_shape(path: Path) -> tuple[int, ...] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if image is None else tuple(int(v) for v in image.shape[:2])


def validate_dataset_contract(
    dataset_root: str | Path,
    *,
    inspect_images: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        descriptor, records = load_contract_dataset(root)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "status": "failed",
            "dataset_root": str(root),
            "record_count": 0,
            "errors": [str(exc)],
            "warnings": [],
        }

    if descriptor.get("contract_version") != CONTRACT_VERSION:
        errors.append(
            f"Descriptor contract_version must be '{CONTRACT_VERSION}', got "
            f"'{descriptor.get('contract_version')}'."
        )
    if not records:
        errors.append("Dataset contains no records.")

    sample_ids: set[str] = set()
    records_by_sample_id: dict[str, dict[str, Any]] = {}
    sequence_frames: set[tuple[str, int]] = set()
    sequence_frame_indices: dict[str, list[int]] = defaultdict(list)
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    defective_count = 0
    clean_count = 0

    for row_number, record in enumerate(records, start=1):
        label = f"record {row_number}"
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            errors.append(f"{label}: missing required fields: {', '.join(missing)}")
            continue
        if record.get("contract_version") != CONTRACT_VERSION:
            errors.append(f"{label}: unsupported contract_version '{record.get('contract_version')}'.")

        sample_id = str(record.get("sample_id", ""))
        sequence_id = str(record.get("sequence_id", ""))
        split = str(record.get("split", ""))
        if not sample_id:
            errors.append(f"{label}: sample_id must not be empty.")
        elif sample_id in sample_ids:
            errors.append(f"{label}: duplicate sample_id '{sample_id}'.")
        sample_ids.add(sample_id)
        records_by_sample_id[sample_id] = record
        if not sequence_id:
            errors.append(f"{label}: sequence_id must not be empty.")
        if split not in ALLOWED_SPLITS:
            errors.append(f"{label}: split '{split}' is not one of {sorted(ALLOWED_SPLITS)}.")
        else:
            split_counts[split] += 1
            sequence_splits[sequence_id].add(split)

        frame_index = record.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            errors.append(f"{label}: frame_index must be a non-negative integer.")
        else:
            sequence_frame = (sequence_id, frame_index)
            if sequence_frame in sequence_frames:
                errors.append(
                    f"{label}: duplicate frame_index {frame_index} in sequence '{sequence_id}'."
                )
            sequence_frames.add(sequence_frame)
            sequence_frame_indices[sequence_id].append(frame_index)

        if not isinstance(record.get("seed"), int) or isinstance(record.get("seed"), bool):
            errors.append(f"{label}: seed must be an integer.")
        if record.get("beam_regime") not in ALLOWED_BEAM_REGIMES:
            errors.append(f"{label}: invalid beam_regime '{record.get('beam_regime')}'.")
        if record.get("render_view_mode") not in ALLOWED_RENDER_VIEWS:
            errors.append(f"{label}: invalid render_view_mode '{record.get('render_view_mode')}'.")
        if not isinstance(record.get("emission_b_mode"), str) or not record["emission_b_mode"]:
            errors.append(f"{label}: emission_b_mode must be a non-empty string.")
        if not isinstance(record.get("emission_b_value"), (int, float)):
            errors.append(f"{label}: emission_b_value must be numeric.")
        if not isinstance(record.get("pixel_size_mm"), (int, float)) or float(
            record.get("pixel_size_mm", 0.0)
        ) <= 0.0:
            errors.append(f"{label}: pixel_size_mm must be positive.")
        if not isinstance(record.get("noise_enabled"), bool):
            errors.append(f"{label}: noise_enabled must be boolean.")

        channels = record.get("channels")
        if not isinstance(channels, dict):
            errors.append(f"{label}: channels must be a mapping.")
            continue
        missing_channels = sorted(REQUIRED_CHANNELS - set(channels))
        if missing_channels:
            errors.append(f"{label}: missing required channels: {', '.join(missing_channels)}")
        prohibited_channels = sorted(PROHIBITED_CHANNELS & set(channels))
        if prohibited_channels:
            errors.append(
                f"{label}: prohibited channels present: {', '.join(prohibited_channels)}"
            )

        defect_types = record.get("defect_types")
        if not isinstance(defect_types, list) or any(not isinstance(v, str) for v in defect_types):
            errors.append(f"{label}: defect_types must be a list of strings.")
            defect_types = []
        is_defective = record.get("is_defective")
        if not isinstance(is_defective, bool):
            errors.append(f"{label}: is_defective must be boolean.")
            is_defective = bool(defect_types)
        if bool(defect_types) != bool(is_defective):
            errors.append(f"{label}: is_defective must agree with defect_types.")
        expected_primary = (
            "clean" if not defect_types else defect_types[0] if len(defect_types) == 1 else "multiple"
        )
        if record.get("defect_type") != expected_primary:
            errors.append(
                f"{label}: defect_type '{record.get('defect_type')}' does not match "
                f"defect_types; expected '{expected_primary}'."
            )
        if is_defective:
            defective_count += 1
            class_counts.update(defect_types)
        else:
            clean_count += 1

        raw_channels = record.get("raw_channels") or {}
        if not isinstance(raw_channels, dict):
            errors.append(f"{label}: raw_channels must be a mapping when provided.")
            raw_channels = {}
        path_values = list(channels.values()) + list(raw_channels.values()) + [record.get("mask_path")]
        for relative_path in path_values:
            candidate = Path(str(relative_path))
            if candidate.is_absolute():
                errors.append(f"{label}: paths must be relative to dataset root: {relative_path}")
                continue
            if ".." in candidate.parts:
                errors.append(f"{label}: paths must not escape dataset root: {relative_path}")

        if not inspect_images:
            continue
        resolved = resolve_record_paths(record, root)
        mask_path = resolved["mask_path"]
        mask_shape = _read_shape(mask_path)
        if mask_shape is None:
            errors.append(f"{label}: unreadable mask: {mask_path}")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        mask_nonempty = bool(np.any(mask > 0))
        if mask_nonempty != bool(is_defective):
            errors.append(f"{label}: mask presence does not agree with is_defective.")
        for channel_name in REQUIRED_CHANNELS:
            channel_path = resolved["channels"].get(channel_name)
            if channel_path is None:
                continue
            channel_shape = _read_shape(channel_path)
            if channel_shape is None:
                errors.append(f"{label}: unreadable {channel_name} image: {channel_path}")
            elif channel_shape != mask_shape:
                errors.append(
                    f"{label}: {channel_name} shape {channel_shape} differs from mask {mask_shape}."
                )
        raw_topo_path = resolved["raw_channels"].get("topo")
        if raw_topo_path is not None:
            try:
                raw_topo = np.load(raw_topo_path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                errors.append(f"{label}: unreadable raw TOPO array {raw_topo_path}: {exc}")
            else:
                if tuple(raw_topo.shape[:2]) != mask_shape:
                    errors.append(
                        f"{label}: raw TOPO shape {raw_topo.shape[:2]} differs from mask {mask_shape}."
                    )
                if raw_topo.dtype != np.float32:
                    warnings.append(f"{label}: raw TOPO dtype is {raw_topo.dtype}, expected float32.")
                if not np.all(np.isfinite(raw_topo)):
                    errors.append(f"{label}: raw TOPO contains non-finite values.")
                elif float(np.max(raw_topo) - np.min(raw_topo)) <= 1e-12:
                    errors.append(f"{label}: raw TOPO is degenerate with zero dynamic range.")

    leaked_sequences = {
        sequence_id: sorted(splits)
        for sequence_id, splits in sequence_splits.items()
        if len(splits) > 1
    }
    for sequence_id, splits in leaked_sequences.items():
        errors.append(
            f"Sequence leakage: sequence_id '{sequence_id}' occurs in splits {', '.join(splits)}."
        )

    dataset_kind = str(descriptor.get("dataset_kind", ""))
    if dataset_kind == "sequential":
        for sequence_id, indices in sequence_frame_indices.items():
            ordered = sorted(indices)
            expected = list(range(len(ordered)))
            if ordered != expected:
                errors.append(
                    f"Sequence ordering: sequence_id '{sequence_id}' has frame indices "
                    f"{ordered}; expected contiguous indices {expected}."
                )
    elif dataset_kind.startswith("static"):
        nonzero_static = {
            sequence_id: sorted(indices)
            for sequence_id, indices in sequence_frame_indices.items()
            if indices != [0]
        }
        for sequence_id, indices in nonzero_static.items():
            errors.append(
                f"Static sequence '{sequence_id}' must contain exactly frame_index 0; got {indices}."
            )

    window_count = 0
    windows_name = descriptor.get("windows")
    if dataset_kind == "sequential" and windows_name:
        windows_path = root / str(windows_name)
        if not windows_path.exists():
            errors.append(f"Sequential window index not found: {windows_path}")
        else:
            try:
                windows = _load_jsonl(windows_path)
            except ValueError as exc:
                errors.append(str(exc))
                windows = []
            window_count = len(windows)
            context_radius = int(descriptor.get("context_radius", 0))
            seen_window_ids: set[str] = set()
            for row_number, window in enumerate(windows, start=1):
                label = f"window {row_number}"
                window_id = str(window.get("window_id", ""))
                if not window_id or window_id in seen_window_ids:
                    errors.append(f"{label}: window_id must be non-empty and unique.")
                seen_window_ids.add(window_id)
                target_id = str(window.get("target_sample_id", ""))
                previous_ids = window.get("previous_sample_ids")
                future_ids = window.get("future_sample_ids")
                if not isinstance(previous_ids, list) or not isinstance(future_ids, list):
                    errors.append(f"{label}: previous_sample_ids and future_sample_ids must be lists.")
                    continue
                if context_radius > 0 and (
                    len(previous_ids) != context_radius or len(future_ids) != context_radius
                ):
                    errors.append(
                        f"{label}: expected {context_radius} previous and future samples."
                    )
                ordered_ids = [*previous_ids, target_id, *future_ids]
                missing_ids = [sample_id for sample_id in ordered_ids if sample_id not in records_by_sample_id]
                if missing_ids:
                    errors.append(f"{label}: unknown sample IDs: {', '.join(missing_ids)}.")
                    continue
                ordered_records = [records_by_sample_id[sample_id] for sample_id in ordered_ids]
                sequence_ids = {str(record["sequence_id"]) for record in ordered_records}
                splits = {str(record["split"]) for record in ordered_records}
                if sequence_ids != {str(window.get("sequence_id", ""))}:
                    errors.append(f"{label}: samples cross sequence boundaries.")
                if splits != {str(window.get("split", ""))}:
                    errors.append(f"{label}: samples cross split boundaries.")
                frame_indices = [int(record["frame_index"]) for record in ordered_records]
                target_frame_index = int(window.get("target_frame_index", -1))
                expected_indices = list(
                    range(target_frame_index - len(previous_ids), target_frame_index + len(future_ids) + 1)
                )
                if frame_indices != expected_indices:
                    errors.append(
                        f"{label}: frame order {frame_indices} does not match {expected_indices}."
                    )

    expected_noise_policy = descriptor.get("noise_policy")
    if expected_noise_policy == "disabled":
        noisy_records = sum(1 for record in records if record.get("noise_enabled") is True)
        if noisy_records:
            errors.append(
                f"Descriptor noise_policy is disabled but {noisy_records} records enable noise."
            )
    missing_splits = sorted(ALLOWED_SPLITS - set(split_counts))
    if missing_splits:
        warnings.append(f"No records found for splits: {', '.join(missing_splits)}.")

    normalization_name = descriptor.get("normalization")
    normalization: dict[str, Any] | None = None
    if normalization_name:
        normalization_path = root / str(normalization_name)
        if not normalization_path.exists():
            errors.append(f"Normalization descriptor not found: {normalization_path}")
        else:
            try:
                normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid normalization descriptor {normalization_path}: {exc}")
            else:
                if normalization.get("source_split") != "train":
                    errors.append("TOPO normalization must be fitted from the training split only.")
                low = normalization.get("raw_low")
                high = normalization.get("raw_high")
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or not np.isfinite([low, high]).all() or high <= low:
                    errors.append("Normalization raw_low/raw_high must be numeric with raw_high > raw_low.")

    return {
        "status": "ok" if not errors else "failed",
        "dataset_root": str(root),
        "contract_version": descriptor.get("contract_version"),
        "record_count": len(records),
        "sequence_count": len(sequence_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "clean_count": clean_count,
        "defective_count": defective_count,
        "defect_instance_counts": dict(sorted(class_counts.items())),
        "sequence_leakage_count": len(leaked_sequences),
        "window_count": window_count,
        "normalization": normalization,
        "errors": errors,
        "warnings": warnings,
    }
