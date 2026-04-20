from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config_loader import load_config_bundle
from ..domain_randomisation.batch_generator import BatchGenerator, BatchOptions
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..geometry.defect_injection import parse_defect_plan, validate_defect_plan
from ..materials import material_from_average_atomic_number
from ..models import SliceRuntime
from ..renderer.scan_engine import ScanEngine


def generate_single_sample(
    config_dir: str | Path,
    seed: int = 42,
    image_size: int = 256,
    material_name: str | None = None,
    average_atomic_number: float | None = None,
    defect_plan: str | None = None,
    max_defects: int = 3,
    voxel_depth_mm: float | None = None,
    slice_offset_mm: float | None = None,
    slice_tilt_deg: float | None = None,
    slice_rotation_deg: float | None = None,
) -> dict:
    bundle = load_config_bundle(config_dir)
    sampled = sample_parameters(
        bundle,
        seed=seed,
        material_name=material_name,
        max_defects=max_defects,
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

    engine = ScanEngine(image_size=image_size)
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
    voxel_depth_mm: float | None = None,
    slice_offset_mm: float | None = None,
    slice_tilt_deg: float | None = None,
    slice_rotation_deg: float | None = None,
    randomize_slice: bool = True,
) -> dict:
    bundle = load_config_bundle(config_dir)
    options = BatchOptions(
        num_images=num_images,
        output_dir=Path(output_dir),
        image_size=image_size,
        seed=seed,
        workers=workers,
        voxel_depth_mm=voxel_depth_mm,
        slice_offset_mm=slice_offset_mm,
        slice_tilt_deg=slice_tilt_deg,
        slice_rotation_deg=slice_rotation_deg,
        randomize_slice=randomize_slice,
    )
    return BatchGenerator(bundle, options).run()
