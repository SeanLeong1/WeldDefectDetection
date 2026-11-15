from __future__ import annotations

from dataclasses import asdict

import numpy as np

try:
    from scipy.ndimage import map_coordinates
except Exception:  # pragma: no cover - optional dependency fallback
    map_coordinates = None

from ..models import (
    BeamRuntime,
    DefectInstance,
    DetectorRuntime,
    MaterialPreset,
    SectionPlaneSpec,
    SectionSliceState,
    SliceRuntime,
    VoxelVolumeState,
    WeldLayoutRuntime,
)
from .defect_injection import CATEGORY_IDS
from .weld_profile import (
    WELD_GEOMETRY_RECESSED,
    build_weld_layout_maps,
    generate_weld_profile,
    normalize_weld_layout,
    resolve_layout_sampling_grid,
    validate_weld_geometry_mode,
)


def _mm_coords(count: int, spacing_mm: float) -> np.ndarray:
    center = 0.5 * (count - 1)
    return ((np.arange(count, dtype=np.float32) - center) * float(spacing_mm)).astype(
        np.float32
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _xy_grids(volume: VoxelVolumeState) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(volume.x_mm, volume.y_mm, indexing="xy")


def _ellipse_profile_mm(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    cx_mm: float,
    cy_mm: float,
    rx_mm: float,
    ry_mm: float,
    theta_rad: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    ct = float(np.cos(theta_rad))
    st = float(np.sin(theta_rad))
    xr = (x_mm - cx_mm) * ct + (y_mm - cy_mm) * st
    yr = -(x_mm - cx_mm) * st + (y_mm - cy_mm) * ct
    r2 = (xr / max(float(rx_mm), 1e-6)) ** 2 + (yr / max(float(ry_mm), 1e-6)) ** 2
    return np.clip(1.0 - r2, 0.0, None).astype(np.float32) ** float(max(power, 0.2))


def _segment_profile_mm(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    x0_mm: float,
    y0_mm: float,
    x1_mm: float,
    y1_mm: float,
    width0_mm: float,
    width1_mm: float,
    power: float = 1.0,
) -> np.ndarray:
    dx = float(x1_mm - x0_mm)
    dy = float(y1_mm - y0_mm)
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-12:
        return _ellipse_profile_mm(
            x_mm,
            y_mm,
            cx_mm=x0_mm,
            cy_mm=y0_mm,
            rx_mm=max(width0_mm, 1e-6),
            ry_mm=max(width0_mm, 1e-6),
            power=power,
        )
    t = np.clip(((x_mm - x0_mm) * dx + (y_mm - y0_mm) * dy) / seg_len2, 0.0, 1.0)
    px = x0_mm + t * dx
    py = y0_mm + t * dy
    width_mm = max(float(width0_mm), 1e-6) + t * (float(width1_mm) - float(width0_mm))
    dist = np.sqrt((x_mm - px) ** 2 + (y_mm - py) ** 2)
    radial = np.clip(
        1.0 - (dist / np.maximum(width_mm, 1e-6)) ** 2,
        0.0,
        None,
    ) ** float(max(power, 0.2))
    taper = np.clip(np.sin(np.pi * t), 0.0, None) ** 0.7
    return (radial * taper).astype(np.float32)


def _z_profile(
    z_mm: np.ndarray,
    *,
    center_mm: float,
    half_extent_mm: float,
    power: float = 1.0,
) -> np.ndarray:
    norm = np.abs(z_mm - float(center_mm)) / max(float(half_extent_mm), 1e-6)
    return np.clip(1.0 - norm**2, 0.0, None).astype(np.float32) ** float(max(power, 0.2))


def _pick_mm_from_mask(
    rng: np.random.Generator,
    mask: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        ix = int(rng.integers(0, len(x_mm)))
        iy = int(rng.integers(0, len(y_mm)))
        return float(x_mm[ix]), float(y_mm[iy])
    idx = int(rng.integers(0, len(xs)))
    return float(x_mm[xs[idx]]), float(y_mm[ys[idx]])


def _pick_mm_from_mask_by_score(
    mask: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    score: np.ndarray,
) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        iy = int(score.shape[0] // 2)
        ix = int(score.shape[1] // 2)
        return float(x_grid[iy, ix]), float(y_grid[iy, ix])
    local_scores = np.asarray(score[ys, xs], dtype=np.float64)
    best = int(np.argmin(local_scores))
    return float(x_grid[ys[best], xs[best]]), float(y_grid[ys[best], xs[best]])


def _select_fixed_occurrence_target(
    values: np.ndarray,
    occurrence_index: int,
    occurrence_count: int,
) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or occurrence_count <= 1:
        return 0.0
    lo = float(np.quantile(finite, 0.15))
    hi = float(np.quantile(finite, 0.85))
    if hi <= lo + 1e-9:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo + 1e-9:
        return float(finite.mean())
    positions = np.linspace(lo, hi, int(max(occurrence_count, 1)), dtype=np.float64)
    idx = int(np.clip(occurrence_index, 0, len(positions) - 1))
    return float(positions[idx])


def _bias_mask_near_slice(
    mask: np.ndarray,
    x_grid: np.ndarray,
    *,
    target_x_mm: float | None,
    window_mm: float,
) -> np.ndarray:
    if target_x_mm is None:
        return mask
    narrowed = mask & (np.abs(x_grid - float(target_x_mm)) <= float(max(window_mm, 1e-6)))
    return narrowed if narrowed.any() else mask


def _surface_z_reference_mm(volume: VoxelVolumeState) -> float:
    surface_index = int(
        np.clip(
            int(volume.metadata.get("surface_plane_index", len(volume.z_mm) // 2)),
            0,
            len(volume.z_mm) - 1,
        )
    )
    return float(volume.z_mm[surface_index])


def _volume_weld_layout(volume: VoxelVolumeState) -> WeldLayoutRuntime:
    meta = dict(volume.metadata.get("weld_layout") or {})
    return normalize_weld_layout(
        WeldLayoutRuntime(
            mode=str(meta.get("mode", "local_patch")),
            seam_rotation_deg=float(meta.get("seam_rotation_deg", 0.0)),
            seam_offset_mm=float(meta.get("seam_offset_mm", 0.0)),
            circle_radius_mm=float(meta.get("circle_radius_mm", 80.0)),
            circle_position_deg=float(meta.get("circle_position_deg", 90.0)),
        )
    )


def _anchored_center_x_mm(
    volume: VoxelVolumeState,
    target_slice_x_mm: float | None,
) -> float:
    if target_slice_x_mm is None:
        return 0.0
    return float(np.clip(float(target_slice_x_mm), volume.x_mm[0] + 0.25, volume.x_mm[-1] - 0.25))


def _fusion_edge_y_mm(fusion_2d: np.ndarray, volume: VoxelVolumeState) -> float:
    if fusion_2d.any(axis=1).any():
        return float(max(np.abs(volume.y_mm[fusion_2d.any(axis=1)]).max(), 0.8))
    return float(max(np.abs(volume.y_mm).max() * 0.5, 0.8))


def _anchored_weld_point_mm(
    volume: VoxelVolumeState,
    mask: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    signed_distance_mm: np.ndarray,
    axial_mm: np.ndarray,
    *,
    target_slice_x_mm: float | None,
) -> tuple[float, float]:
    layout = _volume_weld_layout(volume)
    if layout.mode == "circular_weld":
        score = np.abs(axial_mm) + 0.35 * np.abs(signed_distance_mm)
        return _pick_mm_from_mask_by_score(mask, x_grid, y_grid, score)
    target_x = 0.0 if target_slice_x_mm is None else float(target_slice_x_mm)
    score = (
        np.square(x_grid - target_x)
        + 0.30 * np.square(y_grid)
        + 0.15 * np.square(signed_distance_mm)
    )
    return _pick_mm_from_mask_by_score(mask, x_grid, y_grid, score)


def _fixed_occurrence_anchor_point_mm(
    volume: VoxelVolumeState,
    mask: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    signed_distance_mm: np.ndarray,
    axial_mm: np.ndarray,
    *,
    target_slice_x_mm: float | None,
    occurrence_index: int,
    occurrence_count: int,
) -> tuple[float, float]:
    if occurrence_count <= 1:
        return _anchored_weld_point_mm(
            volume,
            mask,
            x_grid,
            y_grid,
            signed_distance_mm,
            axial_mm,
            target_slice_x_mm=target_slice_x_mm,
        )
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return _anchored_weld_point_mm(
            volume,
            mask,
            x_grid,
            y_grid,
            signed_distance_mm,
            axial_mm,
            target_slice_x_mm=target_slice_x_mm,
        )
    target_axial = _select_fixed_occurrence_target(
        axial_mm[ys, xs],
        occurrence_index,
        occurrence_count,
    )
    score = (
        np.abs(axial_mm - target_axial)
        + 0.35 * np.abs(signed_distance_mm)
        + 1e-4 * (np.abs(x_grid) + np.abs(y_grid))
    )
    return _pick_mm_from_mask_by_score(mask, x_grid, y_grid, score)


def _local_weld_tangent_theta_rad(
    volume: VoxelVolumeState,
    *,
    x_mm: float,
    y_mm: float,
) -> float:
    layout = _volume_weld_layout(volume)
    if layout.mode == "circular_weld":
        return float(np.arctan2(y_mm, x_mm) + (0.5 * np.pi))
    return float(np.radians(layout.seam_rotation_deg))


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    mag = float(np.linalg.norm(vec))
    if mag <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (vec / mag).astype(np.float32)


def build_section_plane(slice_runtime: SliceRuntime) -> SectionPlaneSpec:
    tilt = np.radians(float(slice_runtime.slice_tilt_deg))
    rotation = np.radians(float(slice_runtime.slice_rotation_deg))
    normal = np.array(
        [
            np.cos(tilt),
            np.sin(tilt) * np.cos(rotation),
            np.sin(tilt) * np.sin(rotation),
        ],
        dtype=np.float32,
    )
    normal = _safe_normalize(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u_axis = ref - np.dot(ref, normal) * normal
    if float(np.linalg.norm(u_axis)) <= 1e-6:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u_axis = ref - np.dot(ref, normal) * normal
    u_axis = _safe_normalize(u_axis)
    v_axis = _safe_normalize(np.cross(normal, u_axis))
    return SectionPlaneSpec(
        origin_mm=(float(slice_runtime.slice_offset_mm), 0.0, 0.0),
        normal_xyz=tuple(float(v) for v in normal),
        u_axis_xyz=tuple(float(v) for v in u_axis),
        v_axis_xyz=tuple(float(v) for v in v_axis),
    )


def build_voxel_volume(
    *,
    beam: BeamRuntime,
    material: MaterialPreset,
    image_size: int,
    pixel_size_mm: float,
    voxel_depth_mm: float,
    crown_height_mm: float,
    seam_depth_mm: float,
    bead_width_mm: float,
    haz_width_mm: float,
    beam_focus_quality: float,
    asymmetry_offset_mm: float,
    weld_geometry_mode: str = WELD_GEOMETRY_RECESSED,
    weld_layout: WeldLayoutRuntime | None = None,
    seed: int,
) -> VoxelVolumeState:
    weld_geometry_mode = validate_weld_geometry_mode(weld_geometry_mode)
    weld_layout = normalize_weld_layout(weld_layout)
    _, _, grid_meta = resolve_layout_sampling_grid(
        width_px=image_size,
        height_px=image_size,
        pixel_size_mm=pixel_size_mm,
        layout=weld_layout,
    )
    layout_pixel_size_mm = float(grid_meta["layout_pixel_size_mm"])
    relief_2d = generate_weld_profile(
        width_px=image_size,
        height_px=image_size,
        crown_height_mm=crown_height_mm,
        seam_depth_mm=seam_depth_mm,
        bead_width_mm=bead_width_mm,
        haz_width_mm=haz_width_mm,
        pixel_size_mm=layout_pixel_size_mm,
        travel_speed_mm_min=beam.travel_speed_mm_min,
        beam_power_kw=beam.power_w / 1000.0,
        beam_focus_quality=beam_focus_quality,
        asymmetry_offset_mm=asymmetry_offset_mm,
        weld_geometry_mode=weld_geometry_mode,
        layout=weld_layout,
        seed=seed,
    ).astype(np.float32)

    x_mm = _mm_coords(image_size, layout_pixel_size_mm)
    y_mm = _mm_coords(image_size, layout_pixel_size_mm)
    lateral_extent_mm = float(max(image_size - 1, 1) * layout_pixel_size_mm)
    depth_extent_mm = float(
        max(
            0.55 * lateral_extent_mm,
            3.2 * max(bead_width_mm, layout_pixel_size_mm),
            2.4,
        )
    )
    nz = max(16, int(np.ceil(depth_extent_mm / max(voxel_depth_mm, 1e-6))) + 1)
    z_mm = _mm_coords(nz, voxel_depth_mm)

    signed_distance_mm, _, layout_meta = build_weld_layout_maps(
        width_px=image_size,
        height_px=image_size,
        pixel_size_mm=layout_pixel_size_mm,
        layout=weld_layout,
        asymmetry_offset_mm=asymmetry_offset_mm,
    )
    fusion_2d = np.abs(signed_distance_mm) <= max(1.5 * layout_pixel_size_mm, 0.5 * bead_width_mm)
    z_phase = 2.0 * np.pi * (z_mm / max(depth_extent_mm, 1e-6))
    z_mod = (
        1.0
        + 0.025 * np.sin(z_phase + 0.7)
        + 0.015 * np.cos(1.7 * z_phase - 0.5)
    ).astype(np.float32)

    if weld_geometry_mode == WELD_GEOMETRY_RECESSED:
        material_z_2d = np.where(fusion_2d, material.z + 0.2, material.z).astype(
            np.float32
        )
        material_z = np.repeat(material_z_2d[None, :, :], nz, axis=0).astype(
            np.float32
        )
        fusion_mask = np.repeat(fusion_2d[None, :, :], nz, axis=0)
        relief_mm = (relief_2d[None, :, :] * z_mod[:, None, None]).astype(np.float32)
        relief_mm += (
            0.0035
            * seam_depth_mm
            * np.sin(0.65 * z_phase)[:, None, None]
        ).astype(np.float32)
        fusion_body_half_depth_mm = float(depth_extent_mm * 0.5)
        surface_plane_index = int(np.argmin(np.abs(z_mm - float(z_mm.max()))))
    else:
        half_depth_mm = float(max(0.55, 1.35 * max(seam_depth_mm, 0.10)))
        z_norm = np.clip(np.abs(z_mm) / max(half_depth_mm, 1e-6), 0.0, 1.0)
        surface_half_width_mm = float(max(0.18, 0.22 * bead_width_mm))
        mid_half_width_mm = float(
            max(surface_half_width_mm + 0.12, 0.54 * bead_width_mm)
        )
        fusion_half_width_mm = surface_half_width_mm + (
            mid_half_width_mm - surface_half_width_mm
        ) * np.clip(1.0 - z_norm**1.55, 0.0, None) ** 0.82
        fusion_mask = np.abs(signed_distance_mm)[None, :, :] <= fusion_half_width_mm[:, None, None]
        material_z = np.where(fusion_mask, material.z + 0.2, material.z).astype(
            np.float32
        )

        surface_z_mm = 0.72 * half_depth_mm
        surface_sigma_mm = float(max(0.12, 0.18 * half_depth_mm))
        surface_envelope = np.exp(
            -0.5 * ((z_mm - surface_z_mm) / max(surface_sigma_mm, 1e-6)) ** 2
        ).astype(np.float32)
        relief_mm = (
            relief_2d[None, :, :]
            * surface_envelope[:, None, None]
            * z_mod[:, None, None]
        ).astype(np.float32)
        body_crown = (
            0.010
            * max(crown_height_mm, 0.04)
            * np.exp(-0.5 * (z_mm / max(0.45 * half_depth_mm, 1e-6)) ** 2)
        ).astype(np.float32)
        relief_mm += (
            body_crown[:, None, None] * fusion_mask.astype(np.float32)
        ).astype(np.float32)
        relief_mm += (
            0.0025
            * seam_depth_mm
            * np.sin(0.65 * z_phase)[:, None, None]
            * fusion_mask.astype(np.float32)
        ).astype(np.float32)
        fusion_body_half_depth_mm = half_depth_mm
        surface_plane_index = int(np.argmin(np.abs(z_mm - surface_z_mm)))

    shape = (nz, image_size, image_size)
    volume = VoxelVolumeState(
        material_z=material_z,
        yield_modifier=np.ones(shape, dtype=np.float32),
        void_fraction=np.zeros(shape, dtype=np.float32),
        relief_mm=relief_mm,
        defect_mask=np.zeros(shape, dtype=bool),
        fusion_mask=fusion_mask,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        spacing_mm=(float(layout_pixel_size_mm), float(layout_pixel_size_mm), float(voxel_depth_mm)),
        origin_mm=(float(x_mm[0]), float(y_mm[0]), float(z_mm[0])),
        metadata={
            "volume_shape": list(shape),
            "depth_extent_mm": depth_extent_mm,
            "bead_width_mm": float(bead_width_mm),
            "haz_width_mm": float(haz_width_mm),
            "pixel_size_mm": float(layout_pixel_size_mm),
            "nominal_pixel_size_mm": float(pixel_size_mm),
            "weld_geometry_mode": weld_geometry_mode,
            "weld_layout": layout_meta,
            "fusion_body_half_depth_mm": fusion_body_half_depth_mm,
            "surface_plane_index": surface_plane_index,
        },
    )
    return volume


def _apply_body_to_volume(
    volume: VoxelVolumeState,
    occupancy: np.ndarray,
    *,
    yield_scale: float,
    void_fraction: float,
    relief_delta_mm: float,
    z_scale: float = 1.0,
) -> np.ndarray:
    occ = np.clip(np.asarray(occupancy, dtype=np.float32), 0.0, 1.0)
    mask = occ > 0.08
    volume.yield_modifier = np.minimum(
        volume.yield_modifier,
        1.0 - (1.0 - float(yield_scale)) * occ,
    )
    volume.void_fraction = np.maximum(
        volume.void_fraction,
        float(void_fraction) * occ,
    )
    volume.relief_mm = (volume.relief_mm + float(relief_delta_mm) * occ).astype(np.float32)
    if z_scale != 1.0:
        volume.material_z = (
            volume.material_z * (1.0 - (1.0 - float(z_scale)) * occ)
        ).astype(np.float32)
    volume.defect_mask |= mask
    return mask


def apply_defect_bodies(
    volume: VoxelVolumeState,
    defect_plan: list[dict[str, object]],
    rng: np.random.Generator,
    *,
    target_slice_x_mm: float | None = None,
    randomize_position: bool = True,
    anchor_to_surface: bool = False,
) -> None:
    x_grid, y_grid = _xy_grids(volume)
    weld_layout = _volume_weld_layout(volume)
    fusion_2d = volume.fusion_mask[volume.fusion_mask.shape[0] // 2]
    signed_distance_mm, axial_mm, _ = build_weld_layout_maps(
        width_px=len(volume.x_mm),
        height_px=len(volume.y_mm),
        pixel_size_mm=float(volume.metadata.get("pixel_size_mm", volume.spacing_mm[0])),
        layout=weld_layout,
    )
    fusion_half_width_mm = max(
        float(np.max(np.abs(signed_distance_mm[fusion_2d]))) if fusion_2d.any() else 0.0,
        1.5 * float(volume.spacing_mm[0]),
    )
    slice_window_mm = max(0.45, 9.0 * volume.spacing_mm[0])
    weld_2d = fusion_2d.copy()
    centerline_half_width_mm = max(2.0 * volume.spacing_mm[0], 0.35 * fusion_half_width_mm)
    centerline_2d = fusion_2d & (np.abs(signed_distance_mm) <= centerline_half_width_mm)
    haz_band_mm = max(0.35, 4.0 * volume.spacing_mm[1])
    haz_2d = np.abs(np.abs(signed_distance_mm) - fusion_half_width_mm) <= haz_band_mm
    haz_2d &= ~centerline_2d
    if weld_layout.mode != "circular_weld":
        weld_2d = _bias_mask_near_slice(
            weld_2d,
            x_grid,
            target_x_mm=target_slice_x_mm,
            window_mm=slice_window_mm,
        )
        centerline_2d = _bias_mask_near_slice(
            centerline_2d,
            x_grid,
            target_x_mm=target_slice_x_mm,
            window_mm=slice_window_mm,
        )
        haz_2d = _bias_mask_near_slice(
            haz_2d,
            x_grid,
            target_x_mm=target_slice_x_mm,
            window_mm=slice_window_mm,
        )
    anchored_center_x_mm, anchored_center_y_mm = _anchored_weld_point_mm(
        volume,
        centerline_2d if centerline_2d.any() else weld_2d,
        x_grid,
        y_grid,
        signed_distance_mm,
        axial_mm,
        target_slice_x_mm=target_slice_x_mm,
    )
    surface_z_mm = _surface_z_reference_mm(volume)
    weld_edge_y_mm = _fusion_edge_y_mm(fusion_2d, volume)
    edge_mask_2d = haz_2d & (signed_distance_mm >= 0.0)
    if not edge_mask_2d.any():
        edge_mask_2d = haz_2d if haz_2d.any() else weld_2d
    anchored_edge_x_mm, anchored_edge_y_mm = _anchored_weld_point_mm(
        volume,
        edge_mask_2d,
        x_grid,
        y_grid,
        signed_distance_mm,
        axial_mm,
        target_slice_x_mm=target_slice_x_mm,
    )
    centerline_theta_rad = _local_weld_tangent_theta_rad(
        volume,
        x_mm=anchored_center_x_mm,
        y_mm=anchored_center_y_mm,
    )
    edge_theta_rad = _local_weld_tangent_theta_rad(
        volume,
        x_mm=anchored_edge_x_mm,
        y_mm=anchored_edge_y_mm,
    )
    type_totals: dict[str, int] = {}
    for item in defect_plan:
        dtype = str(item.get("type", ""))
        type_totals[dtype] = type_totals.get(dtype, 0) + 1
    type_seen: dict[str, int] = {}

    for item in defect_plan:
        defect_type = str(item["type"])
        params = dict(item.get("params", {}))
        occurrence_index = int(type_seen.get(defect_type, 0))
        occurrence_count = int(type_totals.get(defect_type, 1))
        type_seen[defect_type] = occurrence_index + 1
        if randomize_position:
            z_center = (
                float(surface_z_mm)
                if anchor_to_surface
                else float(rng.normal(0.0, 0.10))
            )
            z_span = float(max(4.0 * volume.spacing_mm[2], rng.uniform(0.45, 1.10)))
        else:
            z_center = float(surface_z_mm)
            z_span = float(max(4.0 * volume.spacing_mm[2], 0.30))
        mask3d = np.zeros_like(volume.defect_mask, dtype=bool)

        if defect_type == "keyhole_pore":
            diameter = float(params.get("diameter_mm", 0.8))
            depth = float(params.get("depth_mm", 0.5))
            if randomize_position:
                cx_mm, cy_mm = _pick_mm_from_mask(rng, centerline_2d, volume.x_mm, volume.y_mm)
                theta = float(rng.normal(0.0, np.radians(8.0)))
            else:
                cx_mm, cy_mm = _fixed_occurrence_anchor_point_mm(
                    volume,
                    centerline_2d if centerline_2d.any() else weld_2d,
                    x_grid,
                    y_grid,
                    signed_distance_mm,
                    axial_mm,
                    target_slice_x_mm=target_slice_x_mm,
                    occurrence_index=occurrence_index,
                    occurrence_count=occurrence_count,
                )
                theta = 0.0
            main = _ellipse_profile_mm(
                x_grid,
                y_grid,
                cx_mm=cx_mm,
                cy_mm=cy_mm,
                rx_mm=0.38 * diameter,
                ry_mm=0.78 * diameter,
                theta_rad=theta,
                power=1.0,
            )
            tail = _segment_profile_mm(
                x_grid,
                y_grid,
                x0_mm=cx_mm,
                y0_mm=cy_mm - 0.12 * diameter,
                x1_mm=cx_mm + float(rng.normal(0.0, 0.10 * diameter)),
                y1_mm=cy_mm + 0.95 * diameter,
                width0_mm=0.18 * diameter,
                width1_mm=0.08 * diameter,
                power=1.1,
            )
            xy = np.clip(0.88 * main + 0.45 * tail, 0.0, 1.0)
            z_prof = _z_profile(
                volume.z_mm,
                center_mm=z_center,
                half_extent_mm=max(z_span, 0.60),
                power=1.2,
            )
            occupancy = z_prof[:, None, None] * xy[None, :, :]
            mask3d = _apply_body_to_volume(
                volume,
                occupancy,
                yield_scale=0.76,
                void_fraction=0.80,
                relief_delta_mm=-0.045 * max(depth, 0.08),
            )
        elif defect_type == "intrinsic_pore":
            diameter = float(params.get("diameter_mm", 0.8))
            depth = float(params.get("depth_mm", 0.5))
            if randomize_position:
                cx_mm, cy_mm = _pick_mm_from_mask(rng, weld_2d, volume.x_mm, volume.y_mm)
                theta = float(rng.uniform(-np.pi / 2.0, np.pi / 2.0))
            else:
                cx_mm, cy_mm = _fixed_occurrence_anchor_point_mm(
                    volume,
                    weld_2d,
                    x_grid,
                    y_grid,
                    signed_distance_mm,
                    axial_mm,
                    target_slice_x_mm=target_slice_x_mm,
                    occurrence_index=occurrence_index,
                    occurrence_count=occurrence_count,
                )
                theta = 0.0
            xy = _ellipse_profile_mm(
                x_grid,
                y_grid,
                cx_mm=cx_mm,
                cy_mm=cy_mm,
                rx_mm=0.42 * diameter,
                ry_mm=0.36 * diameter,
                theta_rad=theta,
                power=1.4,
            )
            z_prof = _z_profile(
                volume.z_mm,
                center_mm=z_center + float(rng.normal(0.0, 0.08 * diameter)),
                half_extent_mm=max(0.24, 0.45 * diameter),
                power=1.5,
            )
            occupancy = z_prof[:, None, None] * xy[None, :, :]
            mask3d = _apply_body_to_volume(
                volume,
                occupancy,
                yield_scale=0.86,
                void_fraction=0.52,
                relief_delta_mm=-0.012 * max(depth, 0.05),
                z_scale=0.985,
            )
        elif defect_type in {"crack_solidification", "crack_liquidation"}:
            length = float(params.get("length_mm", 2.0))
            width = float(params.get("width_mm", 0.1))
            depth = float(params.get("depth_mm", 0.4))
            target_mask = centerline_2d if defect_type == "crack_solidification" else haz_2d
            if randomize_position:
                x0_mm, y0_mm = _pick_mm_from_mask(rng, target_mask, volume.x_mm, volume.y_mm)
                theta = float(
                    rng.normal(0.0, np.radians(10.0 if defect_type == "crack_solidification" else 18.0))
                )
                x1_mm = float(np.clip(x0_mm + length * np.cos(theta), volume.x_mm[0], volume.x_mm[-1]))
                y1_mm = float(np.clip(y0_mm + length * np.sin(theta), volume.y_mm[0], volume.y_mm[-1]))
            else:
                if defect_type == "crack_solidification":
                    cx_mm, cy_mm = _fixed_occurrence_anchor_point_mm(
                        volume,
                        centerline_2d if centerline_2d.any() else weld_2d,
                        x_grid,
                        y_grid,
                        signed_distance_mm,
                        axial_mm,
                        target_slice_x_mm=target_slice_x_mm,
                        occurrence_index=occurrence_index,
                        occurrence_count=occurrence_count,
                    )
                    theta = _local_weld_tangent_theta_rad(volume, x_mm=cx_mm, y_mm=cy_mm)
                else:
                    cx_mm, cy_mm = _fixed_occurrence_anchor_point_mm(
                        volume,
                        edge_mask_2d,
                        x_grid,
                        y_grid,
                        signed_distance_mm,
                        axial_mm,
                        target_slice_x_mm=target_slice_x_mm,
                        occurrence_index=occurrence_index,
                        occurrence_count=occurrence_count,
                    )
                    theta = _local_weld_tangent_theta_rad(volume, x_mm=cx_mm, y_mm=cy_mm)
                dx = 0.5 * length * np.cos(theta)
                dy = 0.5 * length * np.sin(theta)
                x0_mm = float(np.clip(cx_mm - dx, volume.x_mm[0], volume.x_mm[-1]))
                x1_mm = float(np.clip(cx_mm + dx, volume.x_mm[0], volume.x_mm[-1]))
                y0_mm = float(np.clip(cy_mm - dy, volume.y_mm[0], volume.y_mm[-1]))
                y1_mm = float(np.clip(cy_mm + dy, volume.y_mm[0], volume.y_mm[-1]))
            main = _segment_profile_mm(
                x_grid,
                y_grid,
                x0_mm=x0_mm,
                y0_mm=y0_mm,
                x1_mm=x1_mm,
                y1_mm=y1_mm,
                width0_mm=max(width, 0.04),
                width1_mm=max(0.45 * width, 0.02),
                power=1.2,
            )
            branch = np.zeros_like(main)
            if defect_type == "crack_solidification" and (
                (randomize_position and rng.random() < 0.8) or not randomize_position
            ):
                bx = x0_mm + 0.56 * (x1_mm - x0_mm)
                by = y0_mm + 0.56 * (y1_mm - y0_mm)
                if randomize_position:
                    btheta = theta + float(rng.choice([-1.0, 1.0])) * float(
                        np.radians(rng.uniform(18.0, 34.0))
                    )
                else:
                    btheta = theta + float(np.radians(24.0))
                branch = _segment_profile_mm(
                    x_grid,
                    y_grid,
                    x0_mm=bx,
                    y0_mm=by,
                    x1_mm=float(np.clip(bx + 0.32 * length * np.cos(btheta), volume.x_mm[0], volume.x_mm[-1])),
                    y1_mm=float(np.clip(by + 0.32 * length * np.sin(btheta), volume.y_mm[0], volume.y_mm[-1])),
                    width0_mm=max(0.6 * width, 0.02),
                    width1_mm=max(0.18 * width, 0.01),
                    power=1.25,
                )
            xy = np.clip(main + 0.42 * branch, 0.0, 1.0)
            z_prof = _z_profile(
                volume.z_mm,
                center_mm=z_center,
                half_extent_mm=max(0.30, 0.22 * length),
                power=1.0,
            )
            occupancy = z_prof[:, None, None] * xy[None, :, :]
            mask3d = _apply_body_to_volume(
                volume,
                occupancy,
                yield_scale=0.72 if defect_type == "crack_liquidation" else 0.79,
                void_fraction=0.66 if defect_type == "crack_liquidation" else 0.56,
                relief_delta_mm=-0.016 * max(depth, 0.05),
            )
        elif defect_type in {"lack_of_fusion", "cold_shut"}:
            extent = float(params.get("extent_mm", 2.0))
            depth = float(params.get("depth_mm", 0.5))
            irregularity = float(params.get("irregularity", 0.4))
            if randomize_position:
                side = float(rng.choice([-1.0, 1.0]))
                cx_mm = float(
                    np.clip(
                        (
                            0.0 if target_slice_x_mm is None else target_slice_x_mm
                        )
                        + rng.normal(0.0, 0.22 * max(extent, 0.4)),
                        volume.x_mm[0] + 0.30,
                        volume.x_mm[-1] - 0.30,
                    )
                )
                cy_mm = float(side * (0.58 * weld_edge_y_mm))
                theta = float(rng.normal(0.0, np.radians(7.0)))
            else:
                cx_mm, cy_mm = _fixed_occurrence_anchor_point_mm(
                    volume,
                    edge_mask_2d,
                    x_grid,
                    y_grid,
                    signed_distance_mm,
                    axial_mm,
                    target_slice_x_mm=target_slice_x_mm,
                    occurrence_index=occurrence_index,
                    occurrence_count=occurrence_count,
                )
                theta = _local_weld_tangent_theta_rad(volume, x_mm=cx_mm, y_mm=cy_mm)
                side = 1.0
            lens = _ellipse_profile_mm(
                x_grid,
                y_grid,
                cx_mm=cx_mm,
                cy_mm=cy_mm,
                rx_mm=0.95 * extent,
                ry_mm=max(0.08, 0.22 * extent),
                theta_rad=theta,
                power=1.15,
            )
            edge = _segment_profile_mm(
                x_grid,
                y_grid,
                x0_mm=cx_mm - 0.85 * extent,
                y0_mm=cy_mm + side * 0.18 * extent,
                x1_mm=cx_mm + 0.85 * extent,
                y1_mm=cy_mm - side * 0.04 * extent,
                width0_mm=max(0.10, 0.12 * extent),
                width1_mm=max(0.08, 0.09 * extent),
                power=1.2,
            )
            xy = np.clip(0.88 * lens + 0.28 * edge, 0.0, 1.0)
            if randomize_position:
                xy = np.clip(
                    xy
                    * rng.normal(1.0, 0.06 * max(irregularity, 0.1), size=xy.shape).astype(np.float32),
                    0.0,
                    1.0,
                )
            if defect_type == "cold_shut":
                lip = _segment_profile_mm(
                    x_grid,
                    y_grid,
                    x0_mm=cx_mm - 0.55 * extent,
                    y0_mm=cy_mm - side * 0.10 * extent,
                    x1_mm=cx_mm + 0.55 * extent,
                    y1_mm=cy_mm - side * 0.18 * extent,
                    width0_mm=max(0.08, 0.10 * extent),
                    width1_mm=max(0.05, 0.05 * extent),
                    power=1.1,
                )
                xy = np.clip(xy + 0.20 * lip, 0.0, 1.0)
                volume.relief_mm += (0.008 * depth * lip)[None, :, :].astype(np.float32)
            z_prof = _z_profile(
                volume.z_mm,
                center_mm=z_center,
                half_extent_mm=max(0.34, 0.36 * extent),
                power=1.1,
            )
            occupancy = z_prof[:, None, None] * xy[None, :, :]
            mask3d = _apply_body_to_volume(
                volume,
                occupancy,
                yield_scale=0.81 if defect_type == "lack_of_fusion" else 0.89,
                void_fraction=0.48 if defect_type == "lack_of_fusion" else 0.16,
                relief_delta_mm=-0.014 * max(depth, 0.05),
                z_scale=0.95 if defect_type == "lack_of_fusion" else 0.985,
            )
        elif defect_type == "undercut":
            groove_depth = float(params.get("groove_depth_mm", 0.2))
            if randomize_position:
                side = float(rng.choice([-1.0, 1.0]))
                y_line = float(side * (0.72 * weld_edge_y_mm))
                x_mid = float(
                    np.clip(
                        (
                            0.0 if target_slice_x_mm is None else target_slice_x_mm
                        )
                        + rng.normal(0.0, 0.30),
                        volume.x_mm[0] + 0.45,
                        volume.x_mm[-1] - 0.45,
                    )
                )
            else:
                x_mid, y_line = _fixed_occurrence_anchor_point_mm(
                    volume,
                    edge_mask_2d,
                    x_grid,
                    y_grid,
                    signed_distance_mm,
                    axial_mm,
                    target_slice_x_mm=target_slice_x_mm,
                    occurrence_index=occurrence_index,
                    occurrence_count=occurrence_count,
                )
                side = 1.0
            x0_mm = float(np.clip(x_mid - 0.95, volume.x_mm[0], volume.x_mm[-1]))
            x1_mm = float(np.clip(x_mid + 0.95, volume.x_mm[0], volume.x_mm[-1]))
            groove = _segment_profile_mm(
                x_grid,
                y_grid,
                x0_mm=x0_mm,
                y0_mm=y_line,
                x1_mm=x1_mm,
                y1_mm=y_line + float(rng.normal(0.0, 0.08)),
                width0_mm=0.14,
                width1_mm=0.11,
                power=1.0,
            )
            shoulder = _ellipse_profile_mm(
                x_grid,
                y_grid,
                cx_mm=0.5 * (x0_mm + x1_mm),
                cy_mm=y_line - side * 0.10,
                rx_mm=0.26 * max(x1_mm - x0_mm, 0.4),
                ry_mm=0.12,
                power=1.5,
            )
            xy = np.clip(groove + 0.22 * shoulder, 0.0, 1.0)
            z_prof = _z_profile(
                volume.z_mm,
                center_mm=z_center,
                half_extent_mm=max(0.42, 0.45 * max(x1_mm - x0_mm, 0.3)),
                power=1.0,
            )
            occupancy = z_prof[:, None, None] * xy[None, :, :]
            mask3d = _apply_body_to_volume(
                volume,
                occupancy,
                yield_scale=0.91,
                void_fraction=0.18,
                relief_delta_mm=-0.022 * max(groove_depth, 0.05),
                z_scale=0.99,
            )
        elif defect_type == "spiking":
            length = float(params.get("spike_length_mm", 1.0))
            width = float(params.get("spike_width_mm", 0.1))
            if randomize_position:
                cx_mm, cy_mm = _pick_mm_from_mask(rng, centerline_2d, volume.x_mm, volume.y_mm)
                theta = float(rng.normal(np.radians(90.0), np.radians(18.0)))
            else:
                cx_mm, cy_mm = _fixed_occurrence_anchor_point_mm(
                    volume,
                    centerline_2d if centerline_2d.any() else weld_2d,
                    x_grid,
                    y_grid,
                    signed_distance_mm,
                    axial_mm,
                    target_slice_x_mm=target_slice_x_mm,
                    occurrence_index=occurrence_index,
                    occurrence_count=occurrence_count,
                )
                theta = _local_weld_tangent_theta_rad(volume, x_mm=cx_mm, y_mm=cy_mm)
            xy = _segment_profile_mm(
                x_grid,
                y_grid,
                x0_mm=cx_mm,
                y0_mm=cy_mm,
                x1_mm=float(np.clip(cx_mm + length * np.cos(theta), volume.x_mm[0], volume.x_mm[-1])),
                y1_mm=float(np.clip(cy_mm + length * np.sin(theta), volume.y_mm[0], volume.y_mm[-1])),
                width0_mm=max(width, 0.05),
                width1_mm=max(0.20 * width, 0.02),
                power=1.25,
            )
            cap = _ellipse_profile_mm(
                x_grid,
                y_grid,
                cx_mm=cx_mm,
                cy_mm=cy_mm,
                rx_mm=max(0.05, 0.60 * width),
                ry_mm=max(0.08, 1.20 * width),
                theta_rad=theta,
                power=1.1,
            )
            xy = np.clip(xy + 0.28 * cap, 0.0, 1.0)
            z_prof = _z_profile(
                volume.z_mm,
                center_mm=z_center,
                half_extent_mm=max(0.26, 0.28 * length),
                power=1.1,
            )
            occupancy = z_prof[:, None, None] * xy[None, :, :]
            mask3d = _apply_body_to_volume(
                volume,
                occupancy,
                yield_scale=1.02,
                void_fraction=0.02,
                relief_delta_mm=0.012,
                z_scale=1.01,
            )
        else:
            continue

        volume.defect_channels.append(
            {
                "type": defect_type,
                "params": params,
                "category_id": CATEGORY_IDS[defect_type],
                "mask": mask3d.astype(np.uint8),
            }
        )


def _sample_field(
    field: np.ndarray,
    *,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    order: int,
    cval: float,
) -> np.ndarray:
    if map_coordinates is None:  # pragma: no cover - fallback path
        iz, iy, ix = coords
        zi = np.clip(np.rint(iz).astype(int), 0, field.shape[0] - 1)
        yi = np.clip(np.rint(iy).astype(int), 0, field.shape[1] - 1)
        xi = np.clip(np.rint(ix).astype(int), 0, field.shape[2] - 1)
        return field[zi, yi, xi]
    return map_coordinates(
        field.astype(np.float32),
        coords,
        order=order,
        mode="constant",
        cval=float(cval),
    )


def slice_volume(
    volume: VoxelVolumeState,
    plane: SectionPlaneSpec,
    *,
    image_size: int,
    pixel_size_mm: float,
    min_visible_area_px: int = 6,
) -> tuple[SectionSliceState, list[DefectInstance]]:
    half_extent = 0.5 * (image_size - 1) * float(pixel_size_mm)
    u_coords = _mm_coords(image_size, pixel_size_mm)
    v_coords = _mm_coords(image_size, pixel_size_mm)
    uu, vv = np.meshgrid(u_coords, v_coords, indexing="xy")
    origin = np.asarray(plane.origin_mm, dtype=np.float32)
    u_axis = np.asarray(plane.u_axis_xyz, dtype=np.float32)
    v_axis = np.asarray(plane.v_axis_xyz, dtype=np.float32)
    points = origin + uu[..., None] * u_axis + vv[..., None] * v_axis

    ix = (points[..., 0] - float(volume.origin_mm[0])) / max(volume.spacing_mm[0], 1e-6)
    iy = (points[..., 1] - float(volume.origin_mm[1])) / max(volume.spacing_mm[1], 1e-6)
    iz = (points[..., 2] - float(volume.origin_mm[2])) / max(volume.spacing_mm[2], 1e-6)
    valid = (
        (ix >= 0.0)
        & (ix <= volume.material_z.shape[2] - 1)
        & (iy >= 0.0)
        & (iy <= volume.material_z.shape[1] - 1)
        & (iz >= 0.0)
        & (iz <= volume.material_z.shape[0] - 1)
    )
    coords = (iz.astype(np.float32), iy.astype(np.float32), ix.astype(np.float32))

    material_z_map = _sample_field(
        volume.material_z,
        coords=coords,
        order=1,
        cval=float(volume.material_z.mean()),
    ).reshape((image_size, image_size))
    yield_modifier_map = _sample_field(
        volume.yield_modifier,
        coords=coords,
        order=1,
        cval=1.0,
    ).reshape((image_size, image_size))
    void_fraction_map = _sample_field(
        volume.void_fraction,
        coords=coords,
        order=1,
        cval=0.0,
    ).reshape((image_size, image_size))
    relief_mm_map = _sample_field(
        volume.relief_mm,
        coords=coords,
        order=1,
        cval=0.0,
    ).reshape((image_size, image_size))
    defect_mask = np.zeros((image_size, image_size), dtype=bool)

    intersections: list[DefectInstance] = []
    for channel in volume.defect_channels:
        mask = (
            _sample_field(
                channel["mask"].astype(np.float32),
                coords=coords,
                order=0,
                cval=0.0,
            ).reshape((image_size, image_size))
            > 0.5
        )
        mask &= valid
        if int(mask.sum()) < int(min_visible_area_px):
            continue
        defect_mask |= mask
        intersections.append(
            DefectInstance(
                defect_type=str(channel["type"]),
                params=dict(channel["params"]),
                mask=mask.astype(bool),
                bbox_xyxy=_bbox_from_mask(mask),
                category_id=int(channel["category_id"]),
            )
        )

    defect_intersections = [
        {
            "type": defect.defect_type,
            "bbox_xyxy": list(defect.bbox_xyxy),
            "area_px": int(defect.mask.sum()),
        }
        for defect in intersections
    ]
    slice_state = SectionSliceState(
        material_z_map=material_z_map.astype(np.float32),
        yield_modifier_map=np.clip(yield_modifier_map, 0.15, 1.20).astype(np.float32),
        void_fraction_map=np.clip(void_fraction_map, 0.0, 1.0).astype(np.float32),
        relief_mm_map=relief_mm_map.astype(np.float32),
        defect_mask=defect_mask.astype(np.uint8),
        plane=plane,
        sampling_extent_mm=(float(half_extent), float(half_extent)),
        defect_intersections=defect_intersections,
        metadata={
            "valid_fraction": float(valid.mean()),
            "volume_shape": list(volume.material_z.shape),
        },
    )
    return slice_state, intersections


def slice_metadata(
    slice_runtime: SliceRuntime,
    plane: SectionPlaneSpec,
    volume: VoxelVolumeState,
    section: SectionSliceState,
) -> dict[str, object]:
    return {
        "slice": asdict(slice_runtime),
        "slice_plane": {
            "origin_mm": list(plane.origin_mm),
            "normal_xyz": list(plane.normal_xyz),
            "u_axis_xyz": list(plane.u_axis_xyz),
            "v_axis_xyz": list(plane.v_axis_xyz),
        },
        "voxel_spacing_mm": list(volume.spacing_mm),
        "volume_shape": list(volume.material_z.shape),
        "sampling_extent_mm": list(section.sampling_extent_mm),
        "slice_intersections": list(section.defect_intersections),
    }
