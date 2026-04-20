from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RangeSpec:
    minimum: float
    maximum: float
    default: float
    unit: str

    def validate(self, value: float, name: str) -> float:
        if value < self.minimum or value > self.maximum:
            raise ValueError(
                f"{name}={value} is outside [{self.minimum}, {self.maximum}] {self.unit}"
            )
        return value


@dataclass(frozen=True)
class BeamSpec:
    accelerating_voltage_kv: RangeSpec
    beam_current_ma: RangeSpec
    focusing_current_norm: RangeSpec
    travel_speed_mm_min: RangeSpec
    dwell_time_us: RangeSpec
    focal_spot_mm: RangeSpec


@dataclass(frozen=True)
class MaterialPreset:
    name: str
    z: float
    a: float
    rho: float
    b: float
    n: float


@dataclass(frozen=True)
class MaterialSpec:
    default_material: str
    presets: dict[str, MaterialPreset]


@dataclass(frozen=True)
class DetectorSpec:
    height_mm: RangeSpec
    radius_mm: RangeSpec
    inner_radius_mm: RangeSpec
    outer_radius_mm: RangeSpec
    quadrants_deg: dict[str, float]
    sensitivity_offset: RangeSpec


@dataclass(frozen=True)
class DefectSpec:
    probabilities: dict[str, float]
    parameter_ranges: dict[str, dict[str, tuple[float, float]]]


@dataclass(frozen=True)
class NoiseSpec:
    poisson_enabled: bool
    thermal_enabled: bool
    contamination_enabled: bool
    temperature_k: RangeSpec
    bandwidth_hz: RangeSpec
    contamination_rate: RangeSpec
    quadrant_offset: RangeSpec


@dataclass(frozen=True)
class SliceSpec:
    voxel_depth_mm: RangeSpec
    slice_offset_mm: RangeSpec
    slice_tilt_deg: RangeSpec
    slice_rotation_deg: RangeSpec


@dataclass(frozen=True)
class BeamRuntime:
    accelerating_voltage_kv: float
    beam_current_ma: float
    focusing_current_norm: float
    travel_speed_mm_min: float
    dwell_time_us: float
    focal_spot_mm: float
    e0_kev: float
    power_w: float


@dataclass(frozen=True)
class DetectorRuntime:
    height_mm: float
    radius_mm: float
    inner_radius_mm: float
    outer_radius_mm: float
    quadrants_deg: dict[str, float]
    sensitivity: dict[str, float]
    psi0_rad: float


@dataclass(frozen=True)
class NoiseRuntime:
    poisson_enabled: bool
    thermal_enabled: bool
    contamination_enabled: bool
    temperature_k: float
    bandwidth_hz: float
    contamination_rate: float
    quadrant_offsets: dict[str, float]


@dataclass(frozen=True)
class SliceRuntime:
    voxel_depth_mm: float
    slice_offset_mm: float
    slice_tilt_deg: float
    slice_rotation_deg: float


@dataclass(frozen=True)
class SectionPlaneSpec:
    origin_mm: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    u_axis_xyz: tuple[float, float, float]
    v_axis_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class DefectInstance:
    defect_type: str
    params: dict[str, float]
    mask: Any
    bbox_xyxy: tuple[int, int, int, int]
    category_id: int


@dataclass(frozen=True)
class RenderChannels:
    compo: Any
    topo: Any
    tx: Any
    ty: Any
    tmag: Any
    rx: Any


@dataclass
class VoxelVolumeState:
    material_z: Any
    yield_modifier: Any
    void_fraction: Any
    relief_mm: Any
    defect_mask: Any
    fusion_mask: Any
    x_mm: Any
    y_mm: Any
    z_mm: Any
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    defect_channels: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionSliceState:
    material_z_map: Any
    yield_modifier_map: Any
    void_fraction_map: Any
    relief_mm_map: Any
    defect_mask: Any
    plane: SectionPlaneSpec
    sampling_extent_mm: tuple[float, float]
    defect_intersections: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    channels: RenderChannels
    defect_mask: Any
    defects: list[DefectInstance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
