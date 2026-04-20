from __future__ import annotations

import numpy as np


def kanaya_okayama_range(a: float, e0_kev: float, z: float, rho: float) -> float:
    """Electron range in micrometers."""
    if z <= 0 or rho <= 0 or e0_kev <= 0:
        return 0.0
    return float((0.0276 * a * (e0_kev**1.67)) / ((z**0.89) * rho))


def defect_visible(depth_um: np.ndarray | float, range_um: float) -> np.ndarray:
    depth = np.asarray(depth_um, dtype=np.float64)
    if range_um <= 0:
        return np.zeros_like(depth, dtype=bool)
    return depth <= range_um


def depth_decay(depth_um: np.ndarray | float, range_um: float) -> np.ndarray:
    depth = np.asarray(depth_um, dtype=np.float64)
    if range_um <= 0:
        return np.zeros_like(depth, dtype=np.float32)
    return np.exp(-depth / max(range_um, 1e-6)).astype(np.float32)

