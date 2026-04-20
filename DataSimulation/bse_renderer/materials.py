from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import MaterialPreset


# Reference values spanning light-to-heavy engineering materials.
_REFERENCE_PRESETS: tuple[MaterialPreset, ...] = (
    MaterialPreset(name="Al", z=13.0, a=26.982, rho=2.70, b=0.11, n=2.20),
    MaterialPreset(name="Ti-6Al-4V", z=22.0, a=47.90, rho=4.43, b=0.15, n=1.92),
    MaterialPreset(name="Fe", z=26.0, a=55.845, rho=7.87, b=0.18, n=1.77),
    MaterialPreset(name="Ni", z=28.0, a=58.693, rho=8.90, b=0.22, n=1.70),
    MaterialPreset(name="Cu", z=29.0, a=63.546, rho=8.96, b=0.23, n=1.68),
    MaterialPreset(name="Mo", z=42.0, a=95.95, rho=10.28, b=0.30, n=1.50),
    MaterialPreset(name="Ag", z=47.0, a=107.868, rho=10.49, b=0.32, n=1.44),
    MaterialPreset(name="W", z=74.0, a=183.84, rho=19.25, b=0.42, n=1.26),
)


def _interp_with_extrapolation(x: float, xp: np.ndarray, yp: np.ndarray) -> float:
    if len(xp) == 1:
        return float(yp[0])

    if x <= xp[0]:
        slope = (yp[1] - yp[0]) / max(xp[1] - xp[0], 1e-12)
        return float(yp[0] + slope * (x - xp[0]))
    if x >= xp[-1]:
        slope = (yp[-1] - yp[-2]) / max(xp[-1] - xp[-2], 1e-12)
        return float(yp[-1] + slope * (x - xp[-1]))
    return float(np.interp(x, xp, yp))


def _build_reference_table(
    known_presets: Iterable[MaterialPreset] | None = None,
) -> list[MaterialPreset]:
    merged: dict[str, MaterialPreset] = {p.name: p for p in _REFERENCE_PRESETS}
    if known_presets is not None:
        for preset in known_presets:
            merged[preset.name] = preset
    return sorted(merged.values(), key=lambda p: p.z)


def material_from_average_atomic_number(
    average_z: float,
    known_presets: Iterable[MaterialPreset] | None = None,
    name: str | None = None,
) -> MaterialPreset:
    """
    Estimate material parameters from average atomic number using interpolation
    over an embedded reference table plus configured presets.
    """
    z = float(average_z)
    if z <= 0:
        raise ValueError("Average atomic number must be > 0.")

    refs = _build_reference_table(known_presets)
    zs = np.array([p.z for p in refs], dtype=np.float64)
    a_vals = np.array([p.a for p in refs], dtype=np.float64)
    rho_vals = np.array([p.rho for p in refs], dtype=np.float64)
    b_vals = np.array([p.b for p in refs], dtype=np.float64)
    n_vals = np.array([p.n for p in refs], dtype=np.float64)

    a = _interp_with_extrapolation(z, zs, a_vals)
    rho = max(0.1, _interp_with_extrapolation(z, zs, rho_vals))
    b = float(np.clip(_interp_with_extrapolation(z, zs, b_vals), 0.05, 0.65))
    n = float(np.clip(_interp_with_extrapolation(z, zs, n_vals), 1.1, 2.6))

    return MaterialPreset(
        name=name or f"Zavg_{z:.2f}",
        z=z,
        a=a,
        rho=rho,
        b=b,
        n=n,
    )
