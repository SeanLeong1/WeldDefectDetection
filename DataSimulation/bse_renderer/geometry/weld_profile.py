from __future__ import annotations

from typing import Optional

import numpy as np


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
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate EBW butt-joint surface profile on a flat plate baseline.

    Baseline behavior is a recessed centerline seam (underfill/keyhole track).
    Positive reinforcement is supported but intentionally small and optional.
    """
    if seed is not None:
        np.random.seed(seed)

    y, x = np.mgrid[0:height_px, 0:width_px]
    cy = height_px / 2.0 + (asymmetry_offset_mm / pixel_size_mm)

    # Convert geometry to pixels
    half_width_px = max(3.0, (bead_width_mm / max(pixel_size_mm, 1e-6)) / 2.0)
    haz_half_width_px = max(
        half_width_px + 1.0,
        ((bead_width_mm + haz_width_mm) / max(pixel_size_mm, 1e-6)) / 2.0,
    )

    dist = np.abs(y - cy)
    norm = dist / half_width_px

    # 1) Recessed seam geometry from keyhole underfill.
    effective_depth = float(np.clip(seam_depth_mm, 0.01, 0.8))
    keyhole_width_px = max(0.8, 0.15 * half_width_px)
    keyhole_profile = np.exp(-0.5 * (dist / keyhole_width_px) ** 2)
    seam_profile = np.exp(-0.5 * (dist / max(0.65 * half_width_px, 1e-6)) ** 2)
    base_underfill = -effective_depth * seam_profile

    collapse_frequency = 0.02 * (1.0 - beam_focus_quality)
    collapse_events = np.random.rand(width_px) < collapse_frequency
    collapse_depth = np.zeros(width_px, dtype=np.float32)
    collapse_depth[collapse_events] = -np.random.uniform(
        0.08, 0.28, size=np.sum(collapse_events)
    ) * max(effective_depth, 0.04)

    from scipy.ndimage import gaussian_filter1d

    collapse_depth_smooth = gaussian_filter1d(collapse_depth, sigma=2.2)
    collapse_2d = (
        collapse_depth_smooth[np.newaxis, :]
        * keyhole_profile[:, :, np.newaxis].squeeze()
    )
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

    # 4) Weld-region roughness: favor low-amplitude stochastic relief over periodic striping.
    base_freq_hz = 50.0 + 30.0 * (beam_power_kw / 10.0)
    travel_speed_mm_s = travel_speed_mm_min / 60.0
    texture_wavelength_mm = travel_speed_mm_s / max(base_freq_hz, 1e-6)
    texture_wavelength_px = max(3.0, texture_wavelength_mm / max(pixel_size_mm, 1e-6))

    stochastic_line = gaussian_filter1d(
        np.random.randn(width_px),
        sigma=max(2.0, texture_wavelength_px * 0.9),
    )
    stochastic_line -= float(stochastic_line.mean())
    stochastic_line /= max(float(np.std(stochastic_line)), 1e-6)
    line_wander = gaussian_filter1d(
        np.random.randn(width_px),
        sigma=max(4.0, texture_wavelength_px * 1.8),
    )
    line_wander -= float(line_wander.mean())
    stochastic_profile = (
        0.75 * stochastic_line + 0.25 * np.tanh(line_wander)
    ).astype(np.float32)

    roughness_amplitude = 0.025 * max(effective_depth, 0.03) * (1.35 - 0.45 * beam_focus_quality)
    roughness_envelope = np.exp(-1.5 * (norm**2))
    roughness_2d = roughness_amplitude * stochastic_profile[np.newaxis, :] * roughness_envelope

    # 5) HAZ micro-texture.
    haz_norm = dist / haz_half_width_px
    haz_roughness = np.where(
        (haz_norm >= 0.95) & (haz_norm <= 1.15),
        0.005 * np.random.randn(height_px, width_px),
        0.0,
    )

    # 6) Surface micro-roughness.
    micro_field = gaussian_filter1d(
        np.random.randn(height_px, width_px),
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
    micro_roughness = (
        0.006
        * max(effective_depth, 0.03)
        * micro_field
        * np.exp(-0.7 * (norm**2))
    )

    # 7) Optional surface-breaking pores.
    surface_pores = np.zeros_like(cap)
    if np.random.rand() < 0.15:
        n_pores = np.random.randint(1, 4)
        for _ in range(n_pores):
            pore_x = np.random.randint(0, width_px)
            pore_y = cy + np.random.randn() * half_width_px * 0.6
            pore_radius_px = np.random.uniform(0.8, 2.5)

            pore_dist = np.sqrt((x - pore_x) ** 2 + (y - pore_y) ** 2)
            pore_depth = -0.18 * max(effective_depth, 0.04)
            surface_pores += pore_depth * np.exp(
                -0.5 * (pore_dist / max(pore_radius_px, 1e-6)) ** 2
            )

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
        seed=seed,
    )
