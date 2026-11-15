from __future__ import annotations

from typing import Optional

import numpy as np

from ..models import WeldLayoutRuntime

WELD_GEOMETRY_RECESSED = "recessed_seam"
WELD_GEOMETRY_CLOSED = "closed_fusion"
WELD_GEOMETRY_MODES = {
    WELD_GEOMETRY_RECESSED,
    WELD_GEOMETRY_CLOSED,
}
WELD_LAYOUT_LOCAL_PATCH = "local_patch"
WELD_LAYOUT_STRAIGHT_SEAM = "straight_seam"
WELD_LAYOUT_CIRCULAR_WELD = "circular_weld"
WELD_LAYOUT_MODES = {
    WELD_LAYOUT_LOCAL_PATCH,
    WELD_LAYOUT_STRAIGHT_SEAM,
    WELD_LAYOUT_CIRCULAR_WELD,
}


def validate_weld_geometry_mode(mode: str) -> str:
    normalized = str(mode or WELD_GEOMETRY_RECESSED).strip().lower()
    if normalized not in WELD_GEOMETRY_MODES:
        raise ValueError(
            f"Unknown weld geometry mode '{mode}'. "
            f"Expected one of {sorted(WELD_GEOMETRY_MODES)}."
        )
    return normalized


def validate_weld_layout_mode(mode: str) -> str:
    normalized = str(mode or WELD_LAYOUT_LOCAL_PATCH).strip().lower()
    if normalized not in WELD_LAYOUT_MODES:
        raise ValueError(
            f"Unknown weld layout mode '{mode}'. "
            f"Expected one of {sorted(WELD_LAYOUT_MODES)}."
        )
    return normalized


def normalize_weld_layout(layout: WeldLayoutRuntime | None) -> WeldLayoutRuntime:
    if layout is None:
        return WeldLayoutRuntime()
    return WeldLayoutRuntime(
        mode=validate_weld_layout_mode(layout.mode),
        seam_rotation_deg=float(layout.seam_rotation_deg),
        seam_offset_mm=float(layout.seam_offset_mm),
        circle_radius_mm=max(float(layout.circle_radius_mm), 1e-3),
        circle_position_deg=float(layout.circle_position_deg),
    )


def _coord_grid_mm(
    width_px: int,
    height_px: int,
    pixel_size_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(width_px, dtype=np.float32) - 0.5 * (width_px - 1)) * float(pixel_size_mm)
    y = (np.arange(height_px, dtype=np.float32) - 0.5 * (height_px - 1)) * float(pixel_size_mm)
    return np.meshgrid(x, y, indexing="xy")


def resolve_layout_sampling_grid(
    *,
    width_px: int,
    height_px: int,
    pixel_size_mm: float,
    layout: WeldLayoutRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | bool]]:
    layout = normalize_weld_layout(layout)
    nominal_pixel_size_mm = float(max(pixel_size_mm, 1e-6))
    effective_pixel_size_mm = nominal_pixel_size_mm
    fit_full_shape = False
    min_dim = max(min(width_px, height_px) - 1, 1)
    frame_half_extent_mm = 0.5 * min_dim * effective_pixel_size_mm

    if layout.mode == WELD_LAYOUT_CIRCULAR_WELD:
        target_radius_mm = max(
            float(layout.circle_radius_mm + abs(layout.seam_offset_mm)),
            6.0 * nominal_pixel_size_mm,
        )
        target_half_extent_mm = max(
            frame_half_extent_mm,
            1.18 * target_radius_mm + 4.0 * nominal_pixel_size_mm,
        )
        effective_pixel_size_mm = (2.0 * target_half_extent_mm) / float(min_dim)
        frame_half_extent_mm = target_half_extent_mm
        fit_full_shape = effective_pixel_size_mm > nominal_pixel_size_mm * 1.0001

    x_mm, y_mm = _coord_grid_mm(width_px, height_px, effective_pixel_size_mm)
    return x_mm, y_mm, {
        "nominal_pixel_size_mm": nominal_pixel_size_mm,
        "layout_pixel_size_mm": float(effective_pixel_size_mm),
        "frame_half_extent_mm": float(frame_half_extent_mm),
        "fit_full_shape": bool(fit_full_shape),
    }


def build_weld_layout_maps(
    *,
    width_px: int,
    height_px: int,
    pixel_size_mm: float,
    layout: WeldLayoutRuntime | None = None,
    asymmetry_offset_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    layout = normalize_weld_layout(layout)
    x_mm, y_mm, grid_meta = resolve_layout_sampling_grid(
        width_px=width_px,
        height_px=height_px,
        pixel_size_mm=pixel_size_mm,
        layout=layout,
    )
    offset_mm = float(layout.seam_offset_mm + asymmetry_offset_mm)

    if layout.mode == WELD_LAYOUT_CIRCULAR_WELD:
        radius_mm = max(float(layout.circle_radius_mm), float(pixel_size_mm) * 2.0)
        theta0 = np.radians(float(layout.circle_position_deg))
        center_x_mm = 0.0
        center_y_mm = 0.0
        dx = x_mm
        dy = y_mm
        radial_mm = np.sqrt(dx**2 + dy**2)
        signed_distance_mm = (radial_mm - radius_mm - offset_mm).astype(np.float32)
        theta = np.arctan2(dy, dx)
        theta_ref = float(theta0)
        delta_theta = np.arctan2(np.sin(theta - theta_ref), np.cos(theta - theta_ref))
        axial_mm = (radius_mm * delta_theta).astype(np.float32)
        meta = {
            "mode": layout.mode,
            "seam_rotation_deg": float(layout.seam_rotation_deg),
            "seam_offset_mm": offset_mm,
            "circle_radius_mm": radius_mm,
            "circle_position_deg": float(layout.circle_position_deg),
            **grid_meta,
        }
        return signed_distance_mm, axial_mm, meta

    rotation_rad = np.radians(float(layout.seam_rotation_deg))
    tangent = np.array([np.cos(rotation_rad), np.sin(rotation_rad)], dtype=np.float32)
    normal = np.array([-np.sin(rotation_rad), np.cos(rotation_rad)], dtype=np.float32)
    signed_distance_mm = (x_mm * normal[0] + y_mm * normal[1] - offset_mm).astype(np.float32)
    axial_mm = (x_mm * tangent[0] + y_mm * tangent[1]).astype(np.float32)
    meta = {
        "mode": layout.mode,
        "seam_rotation_deg": float(layout.seam_rotation_deg),
        "seam_offset_mm": offset_mm,
        "circle_radius_mm": float(layout.circle_radius_mm),
        "circle_position_deg": float(layout.circle_position_deg),
        **grid_meta,
    }
    return signed_distance_mm, axial_mm, meta


def _sample_axial_line(
    axial_mm: np.ndarray,
    line_values: np.ndarray,
    axis_support_mm: np.ndarray,
) -> np.ndarray:
    sampled = np.interp(
        axial_mm.ravel(),
        axis_support_mm.astype(np.float64),
        line_values.astype(np.float64),
    )
    return sampled.reshape(axial_mm.shape).astype(np.float32)


def generate_ebw_butt_weld_profile(
    width_px: int,
    height_px: int,
    crown_height_mm: float,
    seam_depth_mm: float,
    bead_width_mm: float,
    haz_width_mm: float,
    pixel_size_mm: float,
    travel_speed_mm_min: float,
    beam_power_kw: float,
    beam_focus_quality: float = 1.0,
    asymmetry_offset_mm: float = 0.0,
    weld_geometry_mode: str = WELD_GEOMETRY_RECESSED,
    layout: WeldLayoutRuntime | None = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate EBW butt-joint surface profile on a flat plate baseline.

    Baseline behavior is a recessed centerline seam (underfill/keyhole track).
    Positive reinforcement is supported but intentionally small and optional.
    """
    rng = np.random.default_rng(seed)
    weld_geometry_mode = validate_weld_geometry_mode(weld_geometry_mode)
    layout = normalize_weld_layout(layout)
    signed_distance_mm, axial_mm, _ = build_weld_layout_maps(
        width_px=width_px,
        height_px=height_px,
        pixel_size_mm=pixel_size_mm,
        layout=layout,
        asymmetry_offset_mm=asymmetry_offset_mm,
    )
    axial_span_mm = max(
        float(np.max(np.abs(axial_mm))),
        float(pixel_size_mm),
    )

    half_width_mm = max(3.0 * pixel_size_mm, bead_width_mm / 2.0)
    haz_half_width_mm = max(
        half_width_mm + pixel_size_mm,
        (bead_width_mm + haz_width_mm) / 2.0,
    )

    dist_mm = np.abs(signed_distance_mm)
    norm = dist_mm / max(half_width_mm, 1e-6)

    # 1) Recessed seam geometry from keyhole underfill.
    effective_depth = float(np.clip(seam_depth_mm, 0.01, 0.8))
    keyhole_width_mm = max(0.8 * pixel_size_mm, 0.15 * half_width_mm)
    keyhole_profile = np.exp(-0.5 * (dist_mm / max(keyhole_width_mm, 1e-6)) ** 2)
    seam_profile = np.exp(-0.5 * (dist_mm / max(0.65 * half_width_mm, 1e-6)) ** 2)
    from scipy.ndimage import gaussian_filter1d

    axis_min = float(axial_mm.min()) - pixel_size_mm
    axis_max = float(axial_mm.max()) + pixel_size_mm
    axis_span_mm = max(axis_max - axis_min, pixel_size_mm)
    axis_samples = max(width_px, int(np.ceil(axis_span_mm / max(pixel_size_mm, 1e-6))) + 3)
    axis_support_mm = np.linspace(axis_min, axis_max, axis_samples, dtype=np.float32)

    if weld_geometry_mode == WELD_GEOMETRY_RECESSED:
        base_underfill = -effective_depth * seam_profile
        collapse_frequency = 0.02 * (1.0 - beam_focus_quality)
        collapse_events = rng.random(axis_samples) < collapse_frequency
        collapse_depth = np.zeros(axis_samples, dtype=np.float32)
        collapse_depth[collapse_events] = -rng.uniform(
            0.08, 0.28, size=np.sum(collapse_events)
        ) * max(effective_depth, 0.04)

        collapse_depth_smooth = gaussian_filter1d(collapse_depth, sigma=2.2)
        collapse_2d = _sample_axial_line(axial_mm, collapse_depth_smooth, axis_support_mm)
        collapse_2d *= keyhole_profile
        keyhole_trace = base_underfill * keyhole_profile + collapse_2d

        # 2) Optional reinforcement (kept small by design).
        effective_crown = float(np.clip(crown_height_mm, 0.0, 0.12))
        crown_shape = np.where(norm < 1.0, (1.0 - norm**2.0), 0.0)
        cap = effective_crown * crown_shape

        # 3) Fusion zone boundary notch/ridge transitions around the recessed seam.
        fusion_boundary = np.where(
            (norm >= 0.90) & (norm <= 1.05),
            -0.05 * max(effective_depth, 0.03),
            0.0,
        )
    else:
        effective_crown = float(
            np.clip(max(crown_height_mm, 0.20 * effective_depth), 0.012, 0.12)
        )
        crown_shape = np.where(norm < 1.18, (1.0 - (norm / 1.18) ** 2.0), 0.0)
        cap = effective_crown * (crown_shape**1.2)
        root_sink = -0.09 * effective_depth * np.exp(
            -0.5 * (dist_mm / max(0.22 * half_width_mm, 1e-6)) ** 2
        )
        closure_variation = gaussian_filter1d(
            rng.standard_normal(axis_samples),
            sigma=max(5.0, 0.95 * half_width_mm / max(pixel_size_mm, 1e-6)),
        ).astype(np.float32)
        closure_variation -= float(closure_variation.mean())
        closure_variation /= max(float(np.std(closure_variation)), 1e-6)
        closure_variation_2d = _sample_axial_line(axial_mm, closure_variation, axis_support_mm)
        closure_variation_2d *= 0.006 * max(effective_depth, 0.04) * np.exp(-2.4 * (norm**2))
        toe_fill = 0.11 * effective_crown * np.exp(
            -0.5 * ((norm - 0.88) / 0.17) ** 2
        )
        fusion_boundary = 0.025 * effective_crown * np.exp(
            -0.5 * ((norm - 1.02) / 0.11) ** 2
        )
        keyhole_trace = root_sink + closure_variation_2d + 0.05 * toe_fill * keyhole_profile

    # 4) Weld-region roughness: favor low-amplitude stochastic relief over periodic striping.
    base_freq_hz = 50.0 + 30.0 * (beam_power_kw / 10.0)
    travel_speed_mm_s = travel_speed_mm_min / 60.0
    texture_wavelength_mm = travel_speed_mm_s / max(base_freq_hz, 1e-6)
    texture_wavelength_px = max(3.0, texture_wavelength_mm / max(pixel_size_mm, 1e-6))

    stochastic_line = gaussian_filter1d(
        rng.standard_normal(axis_samples),
        sigma=max(2.0, texture_wavelength_px * 0.9),
    )
    stochastic_line -= float(stochastic_line.mean())
    stochastic_line /= max(float(np.std(stochastic_line)), 1e-6)
    line_wander = gaussian_filter1d(
        rng.standard_normal(axis_samples),
        sigma=max(4.0, texture_wavelength_px * 1.8),
    )
    line_wander -= float(line_wander.mean())
    stochastic_profile = (
        0.75 * stochastic_line + 0.25 * np.tanh(line_wander)
    ).astype(np.float32)

    roughness_scale = 1.0 if weld_geometry_mode == WELD_GEOMETRY_RECESSED else 0.45
    roughness_amplitude = (
        roughness_scale
        * 0.025
        * max(effective_depth, 0.03)
        * (1.35 - 0.45 * beam_focus_quality)
    )
    roughness_envelope = np.exp(-1.5 * (norm**2))
    roughness_2d = (
        roughness_amplitude
        * _sample_axial_line(axial_mm, stochastic_profile, axis_support_mm)
        * roughness_envelope
    )

    # 5) HAZ micro-texture.
    haz_norm = dist_mm / max(haz_half_width_mm, 1e-6)
    haz_noise_amplitude = 0.005 if weld_geometry_mode == WELD_GEOMETRY_RECESSED else 0.002
    haz_roughness = np.where(
        (haz_norm >= 0.95) & (haz_norm <= 1.15),
        haz_noise_amplitude * rng.standard_normal((height_px, width_px)),
        0.0,
    )

    # 6) Surface micro-roughness.
    micro_field = gaussian_filter1d(
        rng.standard_normal((height_px, width_px)),
        sigma=max(1.2, 0.18 * texture_wavelength_px),
        axis=1,
    )
    micro_field = gaussian_filter1d(
        micro_field,
        sigma=max(1.0, 0.12 * texture_wavelength_px),
        axis=0,
    )
    micro_field -= float(micro_field.mean())
    micro_field /= max(float(np.std(micro_field)), 1e-6)
    micro_scale = 0.006 if weld_geometry_mode == WELD_GEOMETRY_RECESSED else 0.0025
    micro_roughness = (
        micro_scale
        * max(effective_depth, 0.03)
        * micro_field
        * np.exp(-0.7 * (norm**2))
    )

    # 7) Clean baseline: manufactured defects should come from the defect system,
    # not from hidden profile-side pores.
    surface_pores = np.zeros_like(cap)

    if layout.mode == WELD_LAYOUT_LOCAL_PATCH:
        patch_half_length_mm = max(
            2.8 * bead_width_mm,
            0.24 * axial_span_mm,
        )
        patch_profile = np.clip(
            1.0 - (np.abs(axial_mm) / max(patch_half_length_mm, 1e-6)) ** 2,
            0.0,
            None,
        ).astype(np.float32) ** 1.35
        cap *= patch_profile
        fusion_boundary *= patch_profile
        keyhole_trace *= patch_profile
        roughness_2d *= patch_profile
        haz_roughness *= patch_profile
        micro_roughness *= patch_profile

    profile = (
        cap
        + fusion_boundary
        + keyhole_trace
        + roughness_2d
        + haz_roughness
        + micro_roughness
        + surface_pores
    )
    return profile.astype(np.float32)


def generate_weld_profile(
    width_px: int,
    height_px: int,
    crown_height_mm: float,
    seam_depth_mm: float,
    bead_width_mm: float,
    haz_width_mm: float,
    pixel_size_mm: float,
    travel_speed_mm_min: float,
    beam_power_kw: float,
    beam_focus_quality: float = 1.0,
    asymmetry_offset_mm: float = 0.0,
    weld_geometry_mode: str = WELD_GEOMETRY_RECESSED,
    layout: WeldLayoutRuntime | None = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Backward-compatible name used by surface builders.
    """
    return generate_ebw_butt_weld_profile(
        width_px=width_px,
        height_px=height_px,
        crown_height_mm=crown_height_mm,
        seam_depth_mm=seam_depth_mm,
        bead_width_mm=bead_width_mm,
        haz_width_mm=haz_width_mm,
        pixel_size_mm=pixel_size_mm,
        travel_speed_mm_min=travel_speed_mm_min,
        beam_power_kw=beam_power_kw,
        beam_focus_quality=beam_focus_quality,
        asymmetry_offset_mm=asymmetry_offset_mm,
        weld_geometry_mode=weld_geometry_mode,
        layout=layout,
        seed=seed,
    )
