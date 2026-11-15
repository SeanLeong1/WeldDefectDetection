"""Versioned pore sequence generator.

The sequence orchestration is copied from the historical dataset_forms implementation
so frozen campaigns remain reproducible. Rendering and contract helpers are shared.
Only class scheduling changes: one pore per scene, no secondary defect, plus a
predeclared endpoint/midpoint pilot grid. No response, label or visibility tuning.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from dataclasses import replace, asdict
import json
import numpy as np
from bse_renderer.config_loader import load_config_bundle, default_slice_runtime
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.domain_randomisation.dataset_forms import (
    CONTRACT_VERSION, _preflight_frozen_normalization, _prepare_new_output_root,
    _freeze_generation_provenance, _split_for_repetition, _regime_for_sample,
    _planned_target_frame, _set_regime, _apply_noise_profile, _emission_fields,
    _save_raw_topo, _relative, _apply_training_global_topo_normalisation,
    _write_contract_bundle, _sample_defect)
from bse_renderer.models import WeldLayoutRuntime
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.output.image_writer import save_mask
DATASET_CLASSES = ("clean", "keyhole_pore", "intrinsic_pore")


def pore_parameters(kind, rng, grid_index=None):
    if kind not in DATASET_CLASSES:
        raise ValueError("Only clean, keyhole_pore and intrinsic_pore are supported.")
    if grid_index is None or kind == "clean":
        return _sample_defect(kind, rng)
    if not 0 <= grid_index < 9:
        raise ValueError("The pilot has exactly nine diameter/depth combinations.")
    diameters = (.50, .85, 1.20) if kind == "keyhole_pore" else (.50, .75, 1.00)
    depths = (.25, .425, .60) if kind == "keyhole_pore" else (.25, .45, .65)
    return [dict(type=kind, params=dict(diameter_mm=diameters[grid_index//3], depth_mm=depths[grid_index%3]))]


def generate_pore_sequences(
    *,
    config_dir: str | Path,
    output_dir: str | Path,
    sequences_per_class: int = 3,
    frames_per_sequence: int = 9,
    frame_size: int = 96,
    stride_px: int | None = None,
    context_radius: int = 2,
    pixel_size_mm: float = 0.05,
    seed: int = 2201,
    render_model_mode: str | None = None,
    emission_b_mode: str | None = None,
    emission_b_path: str | None = None,
    normalization_path: str | Path | None = None,
    weld_layout_mode: str = "local_patch",
    noise_profile: str = "development",
    fixed_energy_kev: float | None = None,
    pilot_grid: bool = False,
) -> dict[str, Any]:
    if sequences_per_class < 3:
        raise ValueError("sequences_per_class must be at least 3 to populate every split.")
    if frames_per_sequence < 3:
        raise ValueError("frames_per_sequence must be at least 3 for sliding windows.")
    if context_radius < 1 or frames_per_sequence < 2 * context_radius + 1:
        raise ValueError(
            "frames_per_sequence must be at least 2 * context_radius + 1."
        )
    stride_px = frame_size if stride_px is None else stride_px
    if stride_px < frame_size:
        raise ValueError("Unseen-region prediction requires stride_px >= frame_size.")
    if frame_size < 32 or stride_px < 1:
        raise ValueError("frame_size must be >=32 and stride_px must be positive.")
    scene_size = int(frame_size + stride_px * (frames_per_sequence - 1))
    frozen_normalization = _preflight_frozen_normalization(normalization_path)
    cfg = load_config_bundle(config_dir)
    root = _prepare_new_output_root(output_dir)
    engine = ScanEngine(
        image_size=scene_size,
        pixel_size_mm=pixel_size_mm,
        render_model_mode=render_model_mode,
        emission_b_mode=emission_b_mode,
        emission_b_path=emission_b_path,
        render_view_mode="top_surface",
        weld_layout=WeldLayoutRuntime(mode=weld_layout_mode),
    )
    provenance = _freeze_generation_provenance(
        root,
        config_dir=config_dir,
        engine=engine,
        seed=seed,
        pixel_size_mm=pixel_size_mm,
        generation_parameters={
            "dataset_kind": "sequential",
            "class_policy": "clean or exactly one surface-anchored pore; no secondary defects",
            "pilot_grid": pilot_grid,
            "fixed_energy_kev": fixed_energy_kev,
            "prediction_task": "unseen_region",
            "sequences_per_class": int(sequences_per_class),
            "frames_per_sequence": int(frames_per_sequence),
            "frame_size": int(frame_size),
            "stride_px": int(stride_px),
            "context_radius": int(context_radius),
            "weld_layout_mode": weld_layout_mode,
            "noise_profile": noise_profile,
            "normalization_path": (
                None if frozen_normalization is None else str(frozen_normalization)
            ),
        },
    )
    controlled_slice = default_slice_runtime(cfg.slice)
    records: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    resampled_degenerate_count = 0
    maximum_generation_attempt = 1
    for class_index, defect_type in enumerate(DATASET_CLASSES):
        for repetition in range(sequences_per_class):
            sequence_number = class_index * sequences_per_class + repetition
            sequence_id = f"sequence-{sequence_number:04d}"
            split = _split_for_repetition(repetition, sequences_per_class)
            base_sequence_seed = int(seed + 10000 * class_index + repetition)
            regime = _regime_for_sample(repetition, class_index)
            planned_target_frame_index = (
                None
                if defect_type == "clean"
                else _planned_target_frame(
                    repetition,
                    class_index,
                    frames_per_sequence=frames_per_sequence,
                    context_radius=context_radius,
                )
            )
            planned_target_x_mm = (
                0.0
                if planned_target_frame_index is None
                else (
                    planned_target_frame_index
                    - 0.5 * (frames_per_sequence - 1)
                )
                * stride_px
                * pixel_size_mm
            )
            planned_defect_types = [defect_type] if defect_type != "clean" else []
            y0 = (scene_size - frame_size) // 2
            y1 = y0 + frame_size
            for attempt in range(25):
                sequence_seed = int(base_sequence_seed + 1_000_000 * attempt)
                rng = np.random.default_rng(sequence_seed)
                sampled = sample_parameters(
                    cfg,
                    seed=sequence_seed,
                    max_defects=0,
                    randomize_defect_location=True,
                )
                sampled = replace(
                    _set_regime(_apply_noise_profile(sampled, noise_profile), regime, fixed_energy_kev),
                    slice=replace(
                        controlled_slice,
                        slice_offset_mm=float(planned_target_x_mm),
                    ),
                    defects=[
                        defect
                        for planned_type in planned_defect_types
                        for defect in pore_parameters(planned_type, rng, repetition if pilot_grid else None)
                    ],
                )
                result = engine.run(sampled)
                full_visible_types = [d.defect_type for d in result.defects]
                visible_ok = all(
                    planned_type in full_visible_types
                    for planned_type in planned_defect_types
                )
                crops = [
                    np.asarray(
                        result.channels.topo[
                            y0:y1,
                            frame_index * stride_px:frame_index * stride_px + frame_size,
                        ],
                        dtype=np.float32,
                    )
                    for frame_index in range(frames_per_sequence)
                ]
                mask_crops = [
                    result.defect_mask[
                        y0:y1,
                        frame_index * stride_px:frame_index * stride_px + frame_size,
                    ]
                    for frame_index in range(frames_per_sequence)
                ]
                target_visible_ok = (
                    defect_type == "clean"
                    or planned_target_frame_index is not None
                    and np.any(mask_crops[planned_target_frame_index])
                )
                signal_ok = all(
                    np.all(np.isfinite(crop))
                    and float(np.max(crop) - np.min(crop)) > 1e-12
                    for crop in crops
                )
                if visible_ok and target_visible_ok and signal_ok:
                    break
                failed = root / "failed_attempts"
                failed.mkdir(exist_ok=True)
                np.savez_compressed(failed / f"{sequence_id}-attempt-{attempt+1}.npz",
                                    topo=result.channels.topo, mask=result.defect_mask)
                (failed / f"{sequence_id}-attempt-{attempt+1}.json").write_text(
                    json.dumps(dict(seed=sequence_seed, geometric_intersection=visible_ok,
                                    target_intersection=bool(target_visible_ok), finite_dynamic_signal=bool(signal_ok))))
            else:
                raise RuntimeError(
                    f"Unable to generate a non-degenerate surface-visible sequence for "
                    f"'{defect_type}' after 25 attempts from base seed {base_sequence_seed}."
                )
            print(f"{Path(output_dir).name}: {sequence_id} {defect_type} attempt={attempt+1}", flush=True)
            generation_attempt = attempt + 1
            maximum_generation_attempt = max(maximum_generation_attempt, generation_attempt)
            if generation_attempt > 1:
                resampled_degenerate_count += 1
            defect_frame_count = 0
            b_mode, b_value = _emission_fields(result.metadata)
            for frame_index in range(frames_per_sequence):
                x0 = int(frame_index * stride_px)
                x1 = x0 + frame_size
                topo_crop = crops[frame_index]
                mask_crop = mask_crops[frame_index]
                frame_types = [
                    defect.defect_type
                    for defect in result.defects
                    if np.any(defect.mask[y0:y1, x0:x1])
                ]
                if frame_types:
                    defect_frame_count += 1
                sample_id = f"{sequence_id}-frame-{frame_index:04d}"
                raw_topo_path, image_topo_path = _save_raw_topo(
                    root, split, sample_id, topo_crop
                )
                mask_path = save_mask(root, split, sample_id, mask_crop)
                records.append(
                    {
                        "contract_version": CONTRACT_VERSION,
                        "sample_id": sample_id,
                        "sequence_id": sequence_id,
                        "frame_index": frame_index,
                        "split": split,
                        "channels": {"topo": _relative(image_topo_path, root)},
                        "raw_channels": {"topo": _relative(raw_topo_path, root)},
                        "mask_path": _relative(mask_path, root),
                        "defect_type": (
                            "clean"
                            if not frame_types
                            else frame_types[0]
                            if len(frame_types) == 1
                            else "multiple"
                        ),
                        "defect_types": frame_types,
                        "is_defective": bool(frame_types),
                        "seed": sequence_seed,
                        "base_seed": base_sequence_seed,
                        "generation_attempt": generation_attempt,
                        "raw_topo_dynamic_range": float(
                            np.max(topo_crop) - np.min(topo_crop)
                        ),
                        "beam_regime": regime,
                        "emission_b_mode": b_mode,
                        "emission_b_value": b_value,
                        "emission_b_source_path": (result.metadata.get("emission_b") or {}).get("source_path"),
                        "emission_b_source_sha256": (result.metadata.get("emission_b") or {}).get("source_sha256"),
                        "calibration_source": result.metadata.get("calibration_source"),
                        "calibration_sha256": result.metadata.get("calibration_sha256"),
                        "render_view_mode": "top_surface",
                        "pixel_size_mm": float(result.metadata["effective_pixel_size_mm"]),
                        "noise_enabled": noise_profile != "none",
                        "noise_profile": noise_profile,
                        "planned_target_frame_index": planned_target_frame_index,
                        "planned_defect_parameters": list(sampled.defects),
                    "realized_non_noise_parameters": {key: value for key, value in asdict(sampled).items() if key != "noise"},
                    "renderer_parameters": {"image_size": engine.image_size, "pixel_size_mm": engine.pixel_size_mm,
                                            "weld_layout": asdict(engine.weld_layout), "render_model_mode": engine.render_model_mode,
                                            "emission_b_mode": engine.emission_b_mode, "render_view_mode": engine.render_view_mode},
                        "source_scene_size_px": scene_size,
                        "crop_x0_px": x0,
                        "crop_y0_px": y0,
                    }
                )
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "planned_class": defect_type,
                    "planned_defect_types": planned_defect_types,
                    "seed": sequence_seed,
                    "base_seed": base_sequence_seed,
                    "generation_attempt": generation_attempt,
                    "beam_regime": regime,
                    "frames": frames_per_sequence,
                    "defect_frames": defect_frame_count,
                    "planned_target_frame_index": planned_target_frame_index,
                    "planned_target_x_mm": float(planned_target_x_mm),
                }
            )
            sequence_records = [
                record for record in records if record["sequence_id"] == sequence_id
            ]
            sequence_records.sort(key=lambda record: int(record["frame_index"]))
            for center in range(context_radius, frames_per_sequence - context_radius):
                previous = sequence_records[center - context_radius:center]
                future = sequence_records[center + 1:center + context_radius + 1]
                target = sequence_records[center]
                window_rows.append(
                    {
                        "window_id": f"{sequence_id}-window-{center:04d}",
                        "sequence_id": sequence_id,
                        "split": split,
                        "target_sample_id": target["sample_id"],
                        "target_frame_index": center,
                        "previous_sample_ids": [record["sample_id"] for record in previous],
                        "future_sample_ids": [record["sample_id"] for record in future],
                        "target_is_defective": bool(target["is_defective"]),
                    }
                )

    normalization = _apply_training_global_topo_normalisation(
        root, records, frozen_normalization_path=frozen_normalization
    )
    artifacts = _write_contract_bundle(
        root,
        records,
        dataset_kind="sequential",
        extra_descriptor={
            "classes": list(DATASET_CLASSES),
            "sequences_per_class": int(sequences_per_class),
            "frames_per_sequence": int(frames_per_sequence),
            "prediction_task": "unseen_region",
            "frame_size_px": int(frame_size),
            "stride_px": int(stride_px),
            "context_radius": int(context_radius),
            "noise_profile": noise_profile,
            "noise_policy": noise_profile,
            "provenance": "generation_provenance.json",
            "windows": "windows.jsonl",
            "sequence_manifest": "sequence_manifest.json",
            "source_scene_size_px": scene_size,
            "sequence_semantics": (
                "non-overlapping adjacent crops from one coherent rendered weld surface"
                if stride_px >= frame_size
                else "overlapping crops from one coherent rendered weld surface"
            ),
            "context_overlap_policy": (
                "non_overlapping" if stride_px >= frame_size else "overlapping_target_leakage_risk"
            ),
            "weld_layout_mode": str(weld_layout_mode),
            "normalization": "topo_normalization.json",
        },
    )
    sequence_manifest_path = root / "sequence_manifest.json"
    sequence_manifest_path.write_text(json.dumps(sequence_rows, indent=2), encoding="utf-8")
    windows_path = root / "windows.jsonl"
    windows_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in window_rows),
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "output_dir": str(root),
        "record_count": len(records),
        "sequence_count": len(sequence_rows),
        "frames_per_sequence": frames_per_sequence,
        "sequence_manifest": str(sequence_manifest_path),
        "windows": str(windows_path),
        "window_count": len(window_rows),
        "defect_frame_count": int(sum(row["defect_frames"] for row in sequence_rows)),
        "resampled_degenerate_count": resampled_degenerate_count,
        "maximum_generation_attempt": maximum_generation_attempt,
        "normalization": normalization,
        "provenance": provenance,
        **artifacts,
    }
    (root / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
