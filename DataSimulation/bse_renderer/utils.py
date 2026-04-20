from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def normalize_with_window(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("high must be greater than low")
    out = (np.asarray(arr, dtype=np.float32) - float(low)) / float(high - low)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_linear_display_map(
    arr: np.ndarray,
    *,
    low_q: float = 0.02,
    high_q: float = 0.98,
    pad_fraction: float = 0.08,
    clamp_low_to_zero: bool = False,
) -> dict[str, float | str]:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"mode": "linear", "low": 0.0, "high": 1.0}
    lo = float(np.quantile(finite, np.clip(low_q, 0.0, 1.0)))
    hi = float(np.quantile(finite, np.clip(high_q, 0.0, 1.0)))
    if hi <= lo:
        center = float(finite.mean())
        span = max(float(finite.std()) * 2.0, 1e-6)
        lo = center - span
        hi = center + span
    pad = max((hi - lo) * float(max(pad_fraction, 0.0)), 1e-6)
    lo -= pad
    hi += pad
    if clamp_low_to_zero:
        lo = max(0.0, lo)
        hi = max(lo + 1e-6, hi)
    return {"mode": "linear", "low": float(lo), "high": float(hi)}


def build_signed_display_map(
    arr: np.ndarray,
    *,
    quantile: float = 0.985,
    pad_fraction: float = 0.08,
    center: float = 0.0,
) -> dict[str, float | str]:
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"mode": "signed", "center": float(center), "span": 1.0}
    centered = finite - float(center)
    span = float(np.quantile(np.abs(centered), np.clip(quantile, 0.0, 1.0)))
    if span <= 1e-6:
        span = max(float(finite.std()) * 2.0, 1e-6)
    span *= 1.0 + float(max(pad_fraction, 0.0))
    return {"mode": "signed", "center": float(center), "span": float(span)}


def apply_display_map(arr: np.ndarray, mapping: dict[str, float | str]) -> np.ndarray:
    mode = str(mapping.get("mode", "linear"))
    data = np.asarray(arr, dtype=np.float32)
    if mode == "signed":
        center = float(mapping.get("center", 0.0))
        span = max(float(mapping.get("span", 1.0)), 1e-6)
        out = 0.5 + 0.5 * ((data - center) / span)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    low = float(mapping.get("low", 0.0))
    high = float(mapping.get("high", 1.0))
    return normalize_with_window(data, low=low, high=high)


def apply_sem_tone_curve(
    arr: np.ndarray,
    *,
    black: float = 0.14,
    white: float = 0.86,
    gamma: float = 1.05,
) -> np.ndarray:
    data = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    toned = np.power(data, float(max(gamma, 1e-6)))
    toned = float(black) + (float(white) - float(black)) * toned
    return np.clip(toned, 0.0, 1.0).astype(np.float32)


def resize_image(arr: np.ndarray, image_size: int) -> np.ndarray:
    if arr.shape == (image_size, image_size):
        return arr.astype(np.float32)
    img = Image.fromarray((normalize_01(arr) * 65535).astype(np.uint16), mode="I;16")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0


def seed_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
