from __future__ import annotations

import numpy as np


def emission_distribution(
    phi_rad: np.ndarray | float,
    varphi_rad: np.ndarray | float,
    alpha_rad: np.ndarray | float,
    b_coeff: np.ndarray | float,
) -> np.ndarray:
    """Reimer-Riepenhausen-inspired angular emission law."""
    phi = np.asarray(phi_rad, dtype=np.float64)
    varphi = np.asarray(varphi_rad, dtype=np.float64)
    alpha = np.asarray(alpha_rad, dtype=np.float64)
    b = np.asarray(b_coeff, dtype=np.float64)
    lambert = np.cos(phi)
    asym = 1.0 + b * np.sin(alpha) * np.sin(phi) * np.cos(varphi)
    return np.clip(lambert * asym, 0.0, None)


def solid_angle_integral(
    phi_min: float,
    phi_max: float,
    varphi_min: float,
    varphi_max: float,
    alpha_rad: np.ndarray | float,
    b_coeff: np.ndarray | float,
    steps_phi: int = 8,
    steps_varphi: int = 16,
) -> np.ndarray:
    """Numerical integral of emission distribution over solid angle patch."""
    phi = np.linspace(phi_min, phi_max, max(2, steps_phi))
    varphi = np.linspace(varphi_min, varphi_max, max(2, steps_varphi))
    p_grid, v_grid = np.meshgrid(phi, varphi, indexing="ij")

    alpha = np.asarray(alpha_rad, dtype=np.float64)
    b = np.asarray(b_coeff, dtype=np.float64)

    dist = emission_distribution(
        p_grid[..., np.newaxis, np.newaxis],
        v_grid[..., np.newaxis, np.newaxis],
        alpha[np.newaxis, np.newaxis, ...],
        b[np.newaxis, np.newaxis, ...],
    )
    jacobian = np.sin(p_grid)[..., np.newaxis, np.newaxis]
    weighted = dist * jacobian
    dphi = (phi_max - phi_min) / max(1, len(phi) - 1)
    dvarphi = (varphi_max - varphi_min) / max(1, len(varphi) - 1)
    integral = weighted.sum(axis=(0, 1)) * dphi * dvarphi
    return integral.astype(np.float32)

