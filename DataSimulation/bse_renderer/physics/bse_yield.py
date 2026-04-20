from __future__ import annotations

import numpy as np


def eta_z(z: np.ndarray | float) -> np.ndarray:
    """Backscatter coefficient from empirical cubic fit."""
    z_arr = np.asarray(z, dtype=np.float64)
    eta = -0.0254 + 0.016 * z_arr - 1.86e-4 * (z_arr**2) + 8.3e-7 * (z_arr**3)
    return np.clip(eta, 0.0, 1.0)


def eta_oblique(eta_0: np.ndarray | float, alpha_rad: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
    """Oblique incidence correction eta(alpha) = eta0 / cos(alpha)^n, n=9/sqrt(Z)."""
    eta0 = np.asarray(eta_0, dtype=np.float64)
    alpha = np.asarray(alpha_rad, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    n = 9.0 / np.sqrt(np.maximum(z_arr, 1e-6))
    cos_term = np.clip(np.cos(alpha), 1e-4, 1.0)
    eta = eta0 / (cos_term**n)
    return np.clip(eta, 0.0, 1.0)


def eta_void(shape: tuple[int, ...] | None = None) -> np.ndarray | float:
    if shape is None:
        return 0.0
    return np.zeros(shape, dtype=np.float32)

