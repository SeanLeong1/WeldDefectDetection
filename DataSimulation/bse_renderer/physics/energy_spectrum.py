from __future__ import annotations

import numpy as np


def depth_attenuation(depth_um: np.ndarray | float, range_um: float) -> np.ndarray:
    """Proxy attenuation from depth-induced BSE energy loss."""
    depth = np.asarray(depth_um, dtype=np.float64)
    if range_um <= 0:
        return np.zeros_like(depth, dtype=np.float32)
    ratio = np.clip(depth / max(range_um, 1e-6), 0.0, 1.0)
    plateau = np.exp(-1.25 * ratio)
    low_loss = np.sqrt(np.clip(1.0 - ratio, 0.0, 1.0))
    return (0.7 * plateau + 0.3 * low_loss).astype(np.float32)

