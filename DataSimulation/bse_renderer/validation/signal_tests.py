from __future__ import annotations

import numpy as np

from ..physics.bse_yield import eta_oblique, eta_void, eta_z
from ..physics.emission_model import emission_distribution


def run_signal_tests() -> dict:
    z_vals = np.array([6, 13, 26, 29, 79], dtype=np.float64)
    eta_vals = eta_z(z_vals)
    eta_ok = bool(np.all((eta_vals >= 0.0) & (eta_vals <= 1.0)))

    alpha = np.radians(20.0)
    eta_ob = eta_oblique(eta_vals, alpha, z_vals)
    oblique_ok = bool(np.all(eta_ob >= eta_vals))

    phi = np.linspace(0, np.pi / 2, 32)
    varphi = np.linspace(0, 2 * np.pi, 64)
    p_grid, v_grid = np.meshgrid(phi, varphi, indexing="ij")
    dist = emission_distribution(p_grid, v_grid, alpha, 0.0)
    hemi_int = float((dist * np.sin(p_grid)).mean())

    return {
        "eta_bounds_ok": eta_ok,
        "oblique_non_decreasing_ok": oblique_ok,
        "lambert_hemisphere_mean": hemi_int,
        "eta_void": float(eta_void()),
    }

