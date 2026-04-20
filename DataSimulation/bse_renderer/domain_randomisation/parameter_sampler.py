from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config_loader import ConfigBundle
from ..geometry.defect_injection import sample_defect_plan
from ..models import (
    BeamRuntime,
    DetectorRuntime,
    MaterialPreset,
    NoiseRuntime,
    RangeSpec,
    SliceRuntime,
)


@dataclass(frozen=True)
class SampledParameters:
    seed: int
    beam: BeamRuntime
    material: MaterialPreset
    detector: DetectorRuntime
    noise: NoiseRuntime
    slice: SliceRuntime
    defects: list[dict]


def _sample(spec: RangeSpec, rng: np.random.Generator, log: bool = False) -> float:
    if log:
        lo = math.log(max(spec.minimum, 1e-12))
        hi = math.log(max(spec.maximum, 1e-12))
        return float(np.exp(rng.uniform(lo, hi)))
    return float(rng.uniform(spec.minimum, spec.maximum))


def sample_parameters(
    bundle: ConfigBundle,
    seed: int,
    material_name: str | None = None,
    max_defects: int = 3,
) -> SampledParameters:
    rng = np.random.default_rng(seed)
    beam_spec = bundle.beam

    va = _sample(beam_spec.accelerating_voltage_kv, rng)
    ib = _sample(beam_spec.beam_current_ma, rng, log=True)
    if_norm = _sample(beam_spec.focusing_current_norm, rng)
    travel = _sample(beam_spec.travel_speed_mm_min, rng)
    dwell = _sample(beam_spec.dwell_time_us, rng)
    focal = _sample(beam_spec.focal_spot_mm, rng)
    beam = BeamRuntime(
        accelerating_voltage_kv=va,
        beam_current_ma=ib,
        focusing_current_norm=if_norm,
        travel_speed_mm_min=travel,
        dwell_time_us=dwell,
        focal_spot_mm=focal,
        e0_kev=va,
        power_w=(va * 1000.0) * (ib / 1000.0),
    )

    m_name = material_name or bundle.material.default_material
    material = bundle.material.presets[m_name]

    dspec = bundle.detector
    h = _sample(dspec.height_mm, rng)
    r = _sample(dspec.radius_mm, rng)
    inner = np.clip(_sample(dspec.inner_radius_mm, rng), dspec.inner_radius_mm.minimum, dspec.outer_radius_mm.maximum)
    outer = np.clip(_sample(dspec.outer_radius_mm, rng), inner + 0.1, dspec.outer_radius_mm.maximum)
    sensitivity = {}
    for quad in ("A", "B", "C", "D"):
        offset = _sample(dspec.sensitivity_offset, rng)
        sensitivity[quad] = float(1.0 + offset)
    detector = DetectorRuntime(
        height_mm=h,
        radius_mm=r,
        inner_radius_mm=float(inner),
        outer_radius_mm=float(outer),
        quadrants_deg=dict(dspec.quadrants_deg),
        sensitivity=sensitivity,
        psi0_rad=float(np.arctan2(h, r)),
    )

    nspec = bundle.noise
    q_offsets = {
        q: _sample(nspec.quadrant_offset, rng) for q in ("A", "B", "C", "D")
    }
    noise = NoiseRuntime(
        poisson_enabled=nspec.poisson_enabled,
        thermal_enabled=nspec.thermal_enabled,
        contamination_enabled=nspec.contamination_enabled,
        temperature_k=_sample(nspec.temperature_k, rng),
        bandwidth_hz=_sample(nspec.bandwidth_hz, rng, log=True),
        contamination_rate=_sample(nspec.contamination_rate, rng),
        quadrant_offsets=q_offsets,
    )

    sspec = bundle.slice
    slice_runtime = SliceRuntime(
        voxel_depth_mm=_sample(sspec.voxel_depth_mm, rng),
        slice_offset_mm=_sample(sspec.slice_offset_mm, rng),
        slice_tilt_deg=_sample(sspec.slice_tilt_deg, rng),
        slice_rotation_deg=_sample(sspec.slice_rotation_deg, rng),
    )

    defects = sample_defect_plan(bundle.defect, rng, max_defects=max_defects)
    return SampledParameters(
        seed=seed,
        beam=beam,
        material=material,
        detector=detector,
        noise=noise,
        slice=slice_runtime,
        defects=defects,
    )
