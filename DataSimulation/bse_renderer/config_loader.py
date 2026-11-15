from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import (
    BeamRuntime,
    BeamSpec,
    DefectSpec,
    DetectorRuntime,
    DetectorSpec,
    MaterialPreset,
    MaterialSpec,
    NoiseRuntime,
    NoiseSpec,
    RangeSpec,
    SliceRuntime,
    SliceSpec,
)


class ConfigError(ValueError):
    """Raised when config files are missing or invalid."""


@dataclass(frozen=True)
class ConfigBundle:
    config_dir: Path
    beam: BeamSpec
    material: MaterialSpec
    detector: DetectorSpec
    defect: DefectSpec
    noise: NoiseSpec
    slice: SliceSpec


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return data


def _range_spec(node: dict[str, Any], name: str) -> RangeSpec:
    try:
        minimum = float(node["min"])
        maximum = float(node["max"])
        default = float(node["default"])
        unit = str(node.get("unit", ""))
    except KeyError as exc:
        raise ConfigError(f"Range '{name}' missing key: {exc}") from exc
    if minimum > maximum:
        raise ConfigError(f"Invalid range for {name}: min > max")
    if default < minimum or default > maximum:
        raise ConfigError(f"Default for {name} must be inside [min, max]")
    return RangeSpec(minimum=minimum, maximum=maximum, default=default, unit=unit)


def _load_beam(path: Path) -> BeamSpec:
    data = _read_yaml(path)
    return BeamSpec(
        accelerating_voltage_kv=_range_spec(data["accelerating_voltage_kv"], "accelerating_voltage_kv"),
        beam_current_ma=_range_spec(data["beam_current_ma"], "beam_current_ma"),
        focusing_current_norm=_range_spec(data["focusing_current_norm"], "focusing_current_norm"),
        travel_speed_mm_min=_range_spec(data["travel_speed_mm_min"], "travel_speed_mm_min"),
        dwell_time_us=_range_spec(data["dwell_time_us"], "dwell_time_us"),
        focal_spot_mm=_range_spec(data["focal_spot_mm"], "focal_spot_mm"),
    )


def _load_material(path: Path) -> MaterialSpec:
    data = _read_yaml(path)
    default_material = str(data["default_material"])
    raw_presets = data.get("presets", {})
    if not isinstance(raw_presets, dict) or not raw_presets:
        raise ConfigError("material_params.yaml must define non-empty presets")
    presets: dict[str, MaterialPreset] = {}
    for name, preset in raw_presets.items():
        presets[name] = MaterialPreset(
            name=name,
            z=float(preset["z"]),
            a=float(preset["a"]),
            rho=float(preset["rho"]),
            b=float(preset["b"]),
            n=float(preset["n"]),
        )
    if default_material not in presets:
        raise ConfigError(f"default_material='{default_material}' not found in presets")
    return MaterialSpec(default_material=default_material, presets=presets)


def _load_detector(path: Path) -> DetectorSpec:
    data = _read_yaml(path)
    quadrants = data.get("quadrants_deg", {"A": 0.0, "B": 90.0, "C": 180.0, "D": 270.0})
    if set(quadrants) != {"A", "B", "C", "D"}:
        raise ConfigError("detector quadrants must contain keys A,B,C,D")
    return DetectorSpec(
        height_mm=_range_spec(data["height_mm"], "height_mm"),
        radius_mm=_range_spec(data["radius_mm"], "radius_mm"),
        inner_radius_mm=_range_spec(data["inner_radius_mm"], "inner_radius_mm"),
        outer_radius_mm=_range_spec(data["outer_radius_mm"], "outer_radius_mm"),
        quadrants_deg={k: float(v) for k, v in quadrants.items()},
        sensitivity_offset=_range_spec(data["sensitivity_offset"], "sensitivity_offset"),
    )


def _load_defect(path: Path) -> DefectSpec:
    data = _read_yaml(path)
    probs = {str(k): float(v) for k, v in data.get("probabilities", {}).items()}
    if not probs:
        raise ConfigError("defect_params.yaml must contain probabilities")
    for k, value in probs.items():
        if value < 0:
            raise ConfigError(f"Defect probability for {k} must be non-negative")
    total = sum(probs.values())
    if total <= 0:
        raise ConfigError("Defect probabilities must sum to > 0")
    params: dict[str, dict[str, tuple[float, float]]] = {}
    raw_ranges = data.get("ranges", {})
    for defect_name, param_map in raw_ranges.items():
        typed: dict[str, tuple[float, float]] = {}
        for param_name, minmax in param_map.items():
            if not isinstance(minmax, list) or len(minmax) != 2:
                raise ConfigError(
                    f"Defect range {defect_name}.{param_name} must be [min, max]"
                )
            lo, hi = float(minmax[0]), float(minmax[1])
            if lo > hi:
                raise ConfigError(f"Invalid range {defect_name}.{param_name}: min > max")
            typed[param_name] = (lo, hi)
        params[defect_name] = typed
    return DefectSpec(probabilities=probs, parameter_ranges=params)


def _load_noise(path: Path) -> NoiseSpec:
    data = _read_yaml(path)
    return NoiseSpec(
        poisson_enabled=bool(data.get("poisson_enabled", True)),
        thermal_enabled=bool(data.get("thermal_enabled", True)),
        contamination_enabled=bool(data.get("contamination_enabled", True)),
        temperature_k=_range_spec(data["temperature_k"], "temperature_k"),
        bandwidth_hz=_range_spec(data["bandwidth_hz"], "bandwidth_hz"),
        contamination_rate=_range_spec(data["contamination_rate"], "contamination_rate"),
        quadrant_offset=_range_spec(data["quadrant_offset"], "quadrant_offset"),
    )


def _load_slice(path: Path) -> SliceSpec:
    data = _read_yaml(path)
    return SliceSpec(
        voxel_depth_mm=_range_spec(data["voxel_depth_mm"], "voxel_depth_mm"),
        slice_offset_mm=_range_spec(data["slice_offset_mm"], "slice_offset_mm"),
        slice_tilt_deg=_range_spec(data["slice_tilt_deg"], "slice_tilt_deg"),
        slice_rotation_deg=_range_spec(data["slice_rotation_deg"], "slice_rotation_deg"),
    )


def load_config_bundle(config_dir: str | Path) -> ConfigBundle:
    requested = Path(config_dir)
    # Only the built-in default is relative to the package. Explicit paths
    # retain their usual meaning, including useful missing-file errors.
    base = (Path(__file__).resolve().parents[1] / "config"
            if requested == Path("config") else requested.resolve())
    return ConfigBundle(
        config_dir=base,
        beam=_load_beam(base / "beam_params.yaml"),
        material=_load_material(base / "material_params.yaml"),
        detector=_load_detector(base / "detector_params.yaml"),
        defect=_load_defect(base / "defect_params.yaml"),
        noise=_load_noise(base / "noise_params.yaml"),
        slice=_load_slice(base / "slice_params.yaml"),
    )


def default_beam_runtime(spec: BeamSpec) -> BeamRuntime:
    va = spec.accelerating_voltage_kv.default
    ib = spec.beam_current_ma.default
    if_norm = spec.focusing_current_norm.default
    travel = spec.travel_speed_mm_min.default
    dwell = spec.dwell_time_us.default
    focal = spec.focal_spot_mm.default
    return BeamRuntime(
        accelerating_voltage_kv=va,
        beam_current_ma=ib,
        focusing_current_norm=if_norm,
        travel_speed_mm_min=travel,
        dwell_time_us=dwell,
        focal_spot_mm=focal,
        e0_kev=va,
        power_w=(va * 1_000.0) * (ib / 1_000.0),
    )


def default_material(spec: MaterialSpec) -> MaterialPreset:
    return spec.presets[spec.default_material]


def default_detector_runtime(spec: DetectorSpec) -> DetectorRuntime:
    h = spec.height_mm.default
    r = spec.radius_mm.default
    psi0 = math.atan2(h, r)
    return DetectorRuntime(
        height_mm=h,
        radius_mm=r,
        inner_radius_mm=spec.inner_radius_mm.default,
        outer_radius_mm=spec.outer_radius_mm.default,
        quadrants_deg=dict(spec.quadrants_deg),
        sensitivity={q: 1.0 for q in ("A", "B", "C", "D")},
        psi0_rad=psi0,
    )


def default_noise_runtime(spec: NoiseSpec) -> NoiseRuntime:
    return NoiseRuntime(
        poisson_enabled=spec.poisson_enabled,
        thermal_enabled=spec.thermal_enabled,
        contamination_enabled=spec.contamination_enabled,
        temperature_k=spec.temperature_k.default,
        bandwidth_hz=spec.bandwidth_hz.default,
        contamination_rate=spec.contamination_rate.default,
        quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
    )


def default_slice_runtime(spec: SliceSpec) -> SliceRuntime:
    return SliceRuntime(
        voxel_depth_mm=spec.voxel_depth_mm.default,
        slice_offset_mm=spec.slice_offset_mm.default,
        slice_tilt_deg=spec.slice_tilt_deg.default,
        slice_rotation_deg=spec.slice_rotation_deg.default,
    )
