from __future__ import annotations

import numpy as np


def emission_distribution(
    phi_rad: np.ndarray | float,
    varphi_rad: np.ndarray | float,
    alpha_rad: np.ndarray | float,
    b_coeff: np.ndarray | float,
) -> np.ndarray:
    """Reimer-Riepenhausen-inspired angular emission law.

    Invalid fitted densities raise instead of being silently clipped.
    """
    phi = np.asarray(phi_rad, dtype=np.float64)
    varphi = np.asarray(varphi_rad, dtype=np.float64)
    alpha = np.asarray(alpha_rad, dtype=np.float64)
    b = np.asarray(b_coeff, dtype=np.float64)
    if any(not np.all(np.isfinite(value)) for value in (phi, varphi, alpha, b)):
        raise ValueError("Emission angles and B must be finite.")
    if np.any((alpha < 0.0) | (alpha > np.pi / 2.0)):
        raise ValueError("alpha_rad must lie in [0, pi/2].")
    if np.any(np.abs(b * np.sin(alpha)) > 1.0 + 1e-12):
        raise ValueError("Emission B violates global density positivity: abs(B sin(alpha)) <= 1.")
    if np.any((phi < 0.0) | (phi > np.pi / 2.0)):
        raise ValueError("phi_rad must lie in the outward hemisphere [0, pi/2].")
    lambert = np.cos(phi)
    asym = 1.0 + b * np.sin(alpha) * np.sin(phi) * np.cos(varphi)
    minimum = float(np.min(asym))
    if minimum < -1e-12:
        raise ValueError(
            "Emission B produces a negative angular density factor "
            f"(minimum={minimum:.6g}); reject or constrain this B model."
        )
    return lambert * np.maximum(asym, 0.0)


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
    if not 0 <= phi_min < phi_max <= np.pi / 2 or not varphi_min < varphi_max:
        raise ValueError("Integration limits must define an outward solid-angle patch.")
    pn, pw = np.polynomial.legendre.leggauss(max(2, steps_phi))
    vn, vw = np.polynomial.legendre.leggauss(max(2, steps_varphi))
    phi = (phi_max + phi_min) / 2 + pn * (phi_max - phi_min) / 2
    varphi = (varphi_max + varphi_min) / 2 + vn * (varphi_max - varphi_min) / 2
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
    weights = pw[:, None] * vw[None, :] * (phi_max - phi_min) * (varphi_max - varphi_min) / 4
    integral = (weighted * weights[..., None, None]).sum(axis=(0, 1))
    return integral.astype(np.float32)
