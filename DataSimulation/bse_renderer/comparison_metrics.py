from __future__ import annotations

import numpy as np


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(values[mask].mean())


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(values[mask].max())


def compute_topography_delta_metrics(
    reference_topo: np.ndarray,
    candidate_topo: np.ndarray,
    defect_mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    delta = candidate_topo - reference_topo
    abs_delta = np.abs(delta)
    if defect_mask is None:
        defect_mask_bool = np.zeros_like(reference_topo, dtype=bool)
    else:
        defect_mask_bool = defect_mask.astype(bool)
    background_mask = ~defect_mask_bool

    defect_pixels = int(defect_mask_bool.sum())
    background_pixels = int(background_mask.sum())
    defect_mean_abs = _masked_mean(abs_delta, defect_mask_bool)
    background_mean_abs = _masked_mean(abs_delta, background_mask)

    return {
        "mean_abs_topo_delta": float(abs_delta.mean()),
        "max_abs_topo_delta": float(abs_delta.max()),
        "mean_topo_delta": float(delta.mean()),
        "defect_pixels": defect_pixels,
        "background_pixels": background_pixels,
        "defect_fraction": float(defect_pixels / abs_delta.size) if abs_delta.size else 0.0,
        "defect_mean_abs_topo_delta": defect_mean_abs,
        "defect_max_abs_topo_delta": _masked_max(abs_delta, defect_mask_bool),
        "defect_mean_topo_delta": _masked_mean(delta, defect_mask_bool),
        "background_mean_abs_topo_delta": background_mean_abs,
        "background_max_abs_topo_delta": _masked_max(abs_delta, background_mask),
        "background_mean_topo_delta": _masked_mean(delta, background_mask),
        "defect_to_background_mean_ratio": (
            None
            if defect_mean_abs is None
            or background_mean_abs is None
            or background_mean_abs <= 1e-12
            else float(defect_mean_abs / background_mean_abs)
        ),
    }
