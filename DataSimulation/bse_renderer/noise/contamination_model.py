from __future__ import annotations

import numpy as np


def apply_contamination(
    quadrants: np.ndarray,
    contamination_rate: float,
    quadrant_offsets: dict[str, float],
    session_progress: float = 1.0,
    session_mismatch: dict[str, float] | None = None,
) -> np.ndarray:
    q = quadrants.astype(np.float64).copy()
    h = q.shape[0]
    progress = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, np.newaxis]
    session_scale = float(np.clip(session_progress, 0.0, 1.0))
    depth_profile = 0.25 + 0.75 * progress
    q *= (1.0 - contamination_rate * session_scale * depth_profile)[..., np.newaxis]

    names = ["A", "B", "C", "D"]
    mismatch = session_mismatch or {}
    for idx, name in enumerate(names):
        quad_gain = 1.0 + quadrant_offsets.get(name, 0.0)
        quad_gain *= 1.0 + mismatch.get(name, 0.0)
        q[..., idx] *= quad_gain
    return q.astype(np.float32)
