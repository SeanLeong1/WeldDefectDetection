from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from ..comparison_metrics import compute_topography_delta_metrics
from ..config_loader import load_config_bundle
from ..domain_randomisation.batch_generator import BatchGenerator, BatchOptions
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..geometry.defect_injection import parse_defect_plan, validate_defect_plan
from ..materials import material_from_average_atomic_number
from ..models import SliceRuntime, WeldLayoutRuntime
from ..output.image_writer import save_channel_images, save_mask
from ..renderer.calibration import (
    RENDER_MODEL_FORMULA_CASINO,
    RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
    RENDER_MODEL_FORMULA_ONLY,
)
from ..renderer.scan_engine import ScanEngine


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _build_weld_layout(
    weld_layout_mode: str,
    seam_rotation_deg: float,
    seam_offset_mm: float,
    circle_radius_mm: float,
    circle_position_deg: float,
) -> WeldLayoutRuntime:
    return WeldLayoutRuntime(
        mode=str(weld_layout_mode),
        seam_rotation_deg=float(seam_rotation_deg),
        seam_offset_mm=float(seam_offset_mm),
        circle_radius_mm=float(circle_radius_mm),
        circle_position_deg=float(circle_position_deg),
    )


def _build_sampled_parameters(
    config_dir: str | Path,
    seed: int,
    image_size: int,
    material_name: str | None,
    average_atomic_number: float | None,
    defect_plan: str | None,
    max_defects: int,
    voxel_depth_mm: float | None,
    slice_offset_mm: float | None,
    slice_tilt_deg: float | None,
    slice_rotation_deg: float | None,
    randomize_defect_location: bool,
):
    bundle = load_config_bundle(config_dir)
    sampled = sample_parameters(
        bundle,
        seed=seed,
        material_name=material_name,
        max_defects=max_defects,
        randomize_defect_location=randomize_defect_location,
    )
    if average_atomic_number is not None:
        sampled = replace(
            sampled,
            material=material_from_average_atomic_number(
                average_z=average_atomic_number,
                known_presets=bundle.material.presets.values(),
            ),
        )
    if defect_plan is not None:
        parsed = parse_defect_plan(defect_plan)
        validate_defect_plan(bundle.defect, parsed)
        sampled = replace(sampled, defects=parsed)
    if any(v is not None for v in (voxel_depth_mm, slice_offset_mm, slice_tilt_deg, slice_rotation_deg)):
        sampled = replace(
            sampled,
            slice=SliceRuntime(
                voxel_depth_mm=sampled.slice.voxel_depth_mm if voxel_depth_mm is None else float(voxel_depth_mm),
                slice_offset_mm=sampled.slice.slice_offset_mm if slice_offset_mm is None else float(slice_offset_mm),
                slice_tilt_deg=sampled.slice.slice_tilt_deg if slice_tilt_deg is None else float(slice_tilt_deg),
                slice_rotation_deg=sampled.slice.slice_rotation_deg if slice_rotation_deg is None else float(slice_rotation_deg),
            ),
        )
    return sampled


def _result_delta(reference, candidate) -> dict:
    compo_delta = abs(candidate.channels.compo - reference.channels.compo)
    metrics = {
        "mean_abs_compo_delta": float(compo_delta.mean()),
        "max_abs_compo_delta": float(compo_delta.max()),
    }
    metrics.update(
        compute_topography_delta_metrics(
            reference.channels.topo,
            candidate.channels.topo,
            defect_mask=candidate.defect_mask,
        )
    )
    return metrics


def generate_single_sample(
    config_dir: str | Path,
    seed: int = 42,
    image_size: int = 256,
    material_name: str | None = None,
    average_atomic_number: float | None = None,
    defect_plan: str | None = None,
    max_defects: int = 3,
    pixel_size_mm: float = 0.05,
    weld_layout_mode: str = "local_patch",
    seam_rotation_deg: float = 0.0,
    seam_offset_mm: float = 0.0,
    circle_radius_mm: float = 80.0,
    circle_position_deg: float = 90.0,
    voxel_depth_mm: float | None = None,
    slice_offset_mm: float | None = None,
    slice_tilt_deg: float | None = None,
    slice_rotation_deg: float | None = None,
    randomize_defect_location: bool = True,
    calibration_path: str | Path | None = None,
    render_model_mode: str | None = None,
    emission_b_mode: str | None = None,
    emission_b_path: str | Path | None = None,
    render_view_mode: str = "slice",
) -> dict:
    sampled = _build_sampled_parameters(
        config_dir=config_dir,
        seed=seed,
        image_size=image_size,
        material_name=material_name,
        average_atomic_number=average_atomic_number,
        defect_plan=defect_plan,
        max_defects=max_defects,
        voxel_depth_mm=voxel_depth_mm,
        slice_offset_mm=slice_offset_mm,
        slice_tilt_deg=slice_tilt_deg,
        slice_rotation_deg=slice_rotation_deg,
        randomize_defect_location=randomize_defect_location,
    )

    engine = ScanEngine(
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        calibration_path=None if calibration_path is None else str(calibration_path),
        render_model_mode=render_model_mode,
        emission_b_mode=emission_b_mode,
        emission_b_path=None if emission_b_path is None else str(emission_b_path),
        weld_layout=_build_weld_layout(
            weld_layout_mode=weld_layout_mode,
            seam_rotation_deg=seam_rotation_deg,
            seam_offset_mm=seam_offset_mm,
            circle_radius_mm=circle_radius_mm,
            circle_position_deg=circle_position_deg,
        ),
        render_view_mode=render_view_mode,
    )
    result = engine.run(sampled)
    return {
        "channels": result.channels,
        "defect_mask": result.defect_mask,
        "defects": result.defects,
        "metadata": result.metadata,
    }


def generate_batch_dataset(
    config_dir: str | Path,
    output_dir: str | Path,
    num_images: int,
    image_size: int = 256,
    seed: int = 42,
    workers: int = 1,
    pixel_size_mm: float = 0.05,
    weld_layout_mode: str = "local_patch",
    seam_rotation_deg: float = 0.0,
    seam_offset_mm: float = 0.0,
    circle_radius_mm: float = 80.0,
    circle_position_deg: float = 90.0,
    voxel_depth_mm: float | None = None,
    slice_offset_mm: float | None = None,
    slice_tilt_deg: float | None = None,
    slice_rotation_deg: float | None = None,
    randomize_slice: bool = True,
    randomize_defect_location: bool = True,
    calibration_path: str | Path | None = None,
    render_model_mode: str | None = None,
    emission_b_mode: str | None = None,
    emission_b_path: str | Path | None = None,
    disable_noise: bool = False,
    render_view_mode: str = "slice",
) -> dict:
    bundle = load_config_bundle(config_dir)
    options = BatchOptions(
        num_images=num_images,
        output_dir=Path(output_dir),
        image_size=image_size,
        seed=seed,
        workers=workers,
        pixel_size_mm=pixel_size_mm,
        weld_layout_mode=weld_layout_mode,
        seam_rotation_deg=seam_rotation_deg,
        seam_offset_mm=seam_offset_mm,
        circle_radius_mm=circle_radius_mm,
        circle_position_deg=circle_position_deg,
        voxel_depth_mm=voxel_depth_mm,
        slice_offset_mm=slice_offset_mm,
        slice_tilt_deg=slice_tilt_deg,
        slice_rotation_deg=slice_rotation_deg,
        randomize_slice=randomize_slice,
        randomize_defect_location=randomize_defect_location,
        calibration_path=None if calibration_path is None else str(calibration_path),
        render_model_mode=render_model_mode,
        emission_b_mode=emission_b_mode,
        emission_b_path=None if emission_b_path is None else str(emission_b_path),
        disable_noise=disable_noise,
        render_view_mode=render_view_mode,
    )
    return BatchGenerator(bundle, options).run()


def render_comparison_case(
    *,
    config_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    image_size: int = 256,
    material_name: str | None = None,
    average_atomic_number: float | None = None,
    defect_plan: str | None = None,
    max_defects: int = 3,
    pixel_size_mm: float = 0.05,
    weld_layout_mode: str = "local_patch",
    seam_rotation_deg: float = 0.0,
    seam_offset_mm: float = 0.0,
    circle_radius_mm: float = 80.0,
    circle_position_deg: float = 90.0,
    voxel_depth_mm: float | None = None,
    slice_offset_mm: float | None = None,
    slice_tilt_deg: float | None = None,
    slice_rotation_deg: float | None = None,
    randomize_defect_location: bool = True,
    calibration_path: str | Path | None = None,
    render_model_mode: str | None = None,
    emission_b_mode: str | None = None,
    emission_b_path: str | Path | None = None,
    stem: str | None = None,
) -> dict:
    sampled = _build_sampled_parameters(
        config_dir=config_dir,
        seed=seed,
        image_size=image_size,
        material_name=material_name,
        average_atomic_number=average_atomic_number,
        defect_plan=defect_plan,
        max_defects=max_defects,
        voxel_depth_mm=voxel_depth_mm,
        slice_offset_mm=slice_offset_mm,
        slice_tilt_deg=slice_tilt_deg,
        slice_rotation_deg=slice_rotation_deg,
        randomize_defect_location=randomize_defect_location,
    )
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    case_stem = stem or f"seed_{seed}"
    weld_layout = _build_weld_layout(
        weld_layout_mode=weld_layout_mode,
        seam_rotation_deg=seam_rotation_deg,
        seam_offset_mm=seam_offset_mm,
        circle_radius_mm=circle_radius_mm,
        circle_position_deg=circle_position_deg,
    )

    selected_calibration_path = (
        None if calibration_path is None else str(Path(calibration_path).resolve())
    )
    if render_model_mode is None:
        mode_names = [RENDER_MODEL_FORMULA_ONLY]
        if selected_calibration_path is not None:
            mode_names.extend(
                [
                    RENDER_MODEL_FORMULA_CASINO,
                    RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
                ]
            )
    else:
        mode_names = [str(render_model_mode)]

    modes: list[tuple[str, ScanEngine]] = [
        (
            mode_name,
            ScanEngine(
                image_size=image_size,
                pixel_size_mm=pixel_size_mm,
                calibration_path=selected_calibration_path,
                render_model_mode=mode_name,
                emission_b_mode=emission_b_mode,
                emission_b_path=None if emission_b_path is None else str(Path(emission_b_path).resolve()),
                weld_layout=weld_layout,
            ),
        )
        for mode_name in mode_names
    ]

    rendered: dict[str, object] = {}
    rendered_results = {}
    for mode_name, engine in modes:
        result = engine.run(sampled)
        rendered_results[mode_name] = result
        image_paths = save_channel_images(
            out_root,
            split=mode_name,
            stem=case_stem,
            compo=result.channels.compo,
            topo=result.channels.topo,
            tx=result.channels.tx,
            ty=result.channels.ty,
            display_mapping=result.metadata.get("display_mapping"),
            save_u8_preview=True,
        )
        mask_path = save_mask(out_root, mode_name, case_stem, result.defect_mask)
        rendered[mode_name] = {
            "images": image_paths,
            "mask": mask_path,
            "metadata": _json_ready(result.metadata),
            "defects": [d.defect_type for d in result.defects],
        }
    baseline = rendered_results.get(RENDER_MODEL_FORMULA_ONLY)
    for mode_name, result in rendered_results.items():
        if baseline is not None and mode_name != RENDER_MODEL_FORMULA_ONLY:
            rendered[mode_name]["delta_vs_formula_only"] = _result_delta(baseline, result)
        prior_mode = (
            RENDER_MODEL_FORMULA_CASINO
            if mode_name == RENDER_MODEL_FORMULA_CASINO_GEOMETRY
            else None
        )
        if prior_mode is not None and prior_mode in rendered_results:
            rendered[mode_name]["delta_vs_formula_casino"] = _result_delta(
                rendered_results[prior_mode],
                result,
            )

    summary = {
        "status": "ok",
        "case_stem": case_stem,
        "seed": seed,
        "image_size": image_size,
        "pixel_size_mm": pixel_size_mm,
        "weld_layout": asdict(weld_layout),
        "calibration_path": selected_calibration_path,
        "comparison_definition": {
            RENDER_MODEL_FORMULA_ONLY: "formula-only surrogate with no CASINO-derived calibration applied",
            RENDER_MODEL_FORMULA_CASINO: "formula surrogate with CASINO-derived yield, blur, and depth calibration only",
            RENDER_MODEL_FORMULA_CASINO_GEOMETRY: "formula surrogate plus CASINO-derived yield, blur, depth, and defect-geometry response profiles",
        },
        "modes": rendered,
    }
    summary_path = out_root / f"{case_stem}_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
