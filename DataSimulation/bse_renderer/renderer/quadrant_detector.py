from __future__ import annotations

import numpy as np

from ..models import DetectorRuntime, MaterialPreset
from ..physics.bse_yield import eta_oblique, eta_z
from ..physics.emission_model import emission_distribution


def _wrap_pi(angle_rad: np.ndarray) -> np.ndarray:
    return ((angle_rad + np.pi) % (2.0 * np.pi)) - np.pi


def compute_quadrant_signals(
    normals: np.ndarray,
    detector: DetectorRuntime,
    material: MaterialPreset,
    material_z_map: np.ndarray,
    yield_modifier_map: np.ndarray | None = None,
    openness_map: np.ndarray | None = None,
    directional_gain: float = 1.0,
) -> np.ndarray:
    """Return raw quadrant channels with shape (H, W, 4)."""
    nz = np.clip(normals[..., 2], -1.0, 1.0)
    alpha = np.arccos(np.clip(nz, 0.0, 1.0))
    tilt_azimuth = np.arctan2(normals[..., 1], normals[..., 0] + 1e-12)
    eta0 = eta_z(material_z_map)
    eta = eta_oblique(eta0, alpha, np.maximum(material_z_map, 1e-3))
    yield_modifier = (
        np.ones_like(eta, dtype=np.float32)
        if yield_modifier_map is None
        else np.clip(yield_modifier_map.astype(np.float32), 0.15, 1.20)
    )
    openness = (
        np.zeros_like(eta, dtype=np.float32)
        if openness_map is None
        else np.clip(openness_map.astype(np.float32), 0.0, 1.0)
    )
    eta = eta * yield_modifier * (1.0 - 0.35 * openness)

    phi_center = np.pi / 2.0 - detector.psi0_rad
    phi_inner = np.arctan2(detector.inner_radius_mm, max(detector.height_mm, 1e-6))
    phi_outer = np.arctan2(detector.outer_radius_mm, max(detector.height_mm, 1e-6))
    sector_solid_angle = (np.pi / 2.0) * max(
        np.cos(phi_inner) - np.cos(phi_outer),
        1e-6,
    )
    sin_phi = float(np.sin(phi_center))
    cos_phi = float(np.cos(phi_center))
    gain = float(np.clip(directional_gain, 0.2, 2.5))

    quad_order = ["A", "B", "C", "D"]
    channels = []
    for q in quad_order:
        center = np.radians(detector.quadrants_deg[q])
        rel_azimuth = _wrap_pi(center - tilt_azimuth)
        emission = emission_distribution(
            phi_center,
            rel_azimuth,
            alpha_rad=alpha,
            b_coeff=material.b,
        )
        detector_dir = np.array(
            [np.cos(center) * sin_phi, np.sin(center) * sin_phi, cos_phi],
            dtype=np.float32,
        )
        view_term = np.clip(
            (normals[..., 0] * detector_dir[0])
            + (normals[..., 1] * detector_dir[1])
            + (normals[..., 2] * detector_dir[2]),
            0.0,
            1.0,
        )
        directional_term = (1.0 - gain) + gain * (
            0.58 * emission + 0.42 * np.power(view_term, max(material.n, 1.0))
        )
        channel = eta * sector_solid_angle * np.clip(directional_term, 0.05, None)
        channels.append(channel.astype(np.float32))

    return np.stack(channels, axis=-1)
