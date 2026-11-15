from __future__ import annotations

import numpy as np


K_B = 1.380649e-23


def _smooth_1d(arr: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k == 1:
        return arr
    window = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(arr, window, mode="same")


def apply_thermal_drift(
    signal: np.ndarray,
    temperature_k: float,
    bandwidth_hz: float,
    drift_rate: float,
    rng: np.random.Generator,
    axis: int = 0,
    session_baseline: float = 0.0,
    session_progress: float = 1.0,
) -> np.ndarray:
    """
    Inject low-frequency thermal drift plus weak readout noise.
    The previous implementation used large per-pixel white noise, which could
    dominate defect contrast. Here we model line-correlated drift instead.
    """
    arr = signal.astype(np.float64)
    thermal_power = max(K_B * temperature_k * bandwidth_hz, 0.0)
    sigma_line = np.sqrt(thermal_power) * 7.0e5
    sigma_line = float(np.clip(sigma_line, 2e-4, 7e-3))
    sigma_pixel = 0.25 * sigma_line

    n = arr.shape[axis]
    line_steps = rng.normal(0.0, sigma_line, size=n)
    line_drift = np.cumsum(line_steps)
    line_drift = _smooth_1d(line_drift, kernel=max(5, n // 32))
    line_drift -= float(line_drift.mean())
    if axis == 0:
        line_drift = line_drift[:, np.newaxis]
    else:
        line_drift = line_drift[np.newaxis, :]

    ramp = np.linspace(0.0, 1.0, n, dtype=np.float64) * drift_rate
    if axis == 0:
        ramp = ramp[:, np.newaxis]
    else:
        ramp = ramp[np.newaxis, :]

    session_shift = float(session_baseline) * float(np.clip(session_progress, 0.0, 1.0))
    pixel_noise = rng.normal(0.0, sigma_pixel, size=arr.shape)
    return (arr + line_drift + ramp + session_shift + pixel_noise).astype(np.float32)
