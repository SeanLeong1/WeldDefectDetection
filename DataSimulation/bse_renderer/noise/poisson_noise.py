from __future__ import annotations

import numpy as np


def apply_poisson_noise(
    signal: np.ndarray,
    beam_current_ma: float,
    dwell_time_us: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal = np.clip(signal.astype(np.float64), 0.0, None)
    scale = max(1e2, beam_current_ma * dwell_time_us * 10.0)
    expected = np.clip(signal * scale, 0.0, 1e7)
    measured = rng.poisson(expected)
    return (measured / max(scale, 1e-12)).astype(np.float32)

