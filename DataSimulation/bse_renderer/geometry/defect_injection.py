from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..models import DefectInstance, DefectSpec
from .surface_mesh import SurfaceState


CATEGORY_IDS = {
    "keyhole_pore": 1,
    "intrinsic_pore": 2,
    "crack_solidification": 3,
    "crack_liquidation": 4,
    "undercut": 5,
    "lack_of_fusion": 6,
    "cold_shut": 7,
    "spiking": 8,
}


_DEFECT_EXPR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s*$")


def _sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def sample_defect_plan(
    spec: DefectSpec,
    rng: np.random.Generator,
    max_defects: int = 3,
) -> list[dict[str, Any]]:
    defect_types = list(spec.probabilities.keys())
    weights = np.array([spec.probabilities[d] for d in defect_types], dtype=np.float64)
    weights = weights / weights.sum()
    count = int(rng.integers(0, max_defects + 1))
    plan: list[dict[str, Any]] = []
    for _ in range(count):
        dtype = str(rng.choice(defect_types, p=weights))
        params = {}
        for pname, bounds in spec.parameter_ranges.get(dtype, {}).items():
            params[pname] = _sample_range(rng, bounds)
        plan.append({"type": dtype, "params": params})
    return plan


def parse_defect_plan(plan_text: str) -> list[dict[str, Any]]:
    """
    Parse defect plan syntax:
      keyhole_pore(diameter_mm=1.2, depth_mm=0.8);
      crack_solidification(length_mm=4.0, width_mm=0.1, depth_mm=0.4)

    Optional parameter:
      count=<int> expands one entry into repeated defect entries.
    """
    if not plan_text or not plan_text.strip():
        return []

    plan: list[dict[str, Any]] = []
    entries = [chunk.strip() for chunk in plan_text.split(";") if chunk.strip()]
    for entry in entries:
        match = _DEFECT_EXPR.match(entry)
        if not match:
            raise ValueError(
                f"Invalid defect entry '{entry}'. Use type(param=value, ...)."
            )
        defect_type = match.group(1)
        args_text = (match.group(2) or "").strip()
        params: dict[str, float] = {}
        if args_text:
            for token in args_text.split(","):
                if "=" not in token:
                    raise ValueError(
                        f"Invalid token '{token.strip()}' in '{entry}'. Expected key=value."
                    )
                key, value = token.split("=", 1)
                key = key.strip()
                try:
                    params[key] = float(value.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value for '{key}' in '{entry}'."
                    ) from exc

        count = int(params.pop("count", 1))
        if count < 1 or count > 50:
            raise ValueError("count must be between 1 and 50.")
        for _ in range(count):
            plan.append({"type": defect_type, "params": dict(params)})
    return plan


def validate_defect_plan(spec: DefectSpec, defect_plan: list[dict[str, Any]]) -> None:
    known = set(spec.probabilities)
    ranges = spec.parameter_ranges
    for item in defect_plan:
        defect_type = str(item.get("type", ""))
        if defect_type not in known:
            raise ValueError(f"Unknown defect type '{defect_type}'.")
        allowed = ranges.get(defect_type, {})
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Defect '{defect_type}' params must be a mapping.")
        for key, raw_value in params.items():
            if key not in allowed:
                raise ValueError(
                    f"Parameter '{key}' is not valid for defect '{defect_type}'."
                )
            lo, hi = allowed[key]
            value = float(raw_value)
            if value < lo or value > hi:
                raise ValueError(
                    f"{defect_type}.{key}={value} outside allowed range [{lo}, {hi}]."
                )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _draw_disk(shape: tuple[int, int], cx: int, cy: int, radius_px: int) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= max(1, radius_px) ** 2


def _draw_line(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int, width: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    n = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    for x, y in zip(xs, ys):
        out |= _draw_disk(shape, int(round(x)), int(round(y)), max(1, width))
    return out


def _rotated_coords(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    xr = (x - cx) * ct + (y - cy) * st
    yr = -(x - cx) * st + (y - cy) * ct
    return xr.astype(np.float32), yr.astype(np.float32)


def _ellipse_profile(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    theta: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    xr, yr = _rotated_coords(shape, cx, cy, theta)
    r2 = (xr / max(float(rx), 1e-6)) ** 2 + (yr / max(float(ry), 1e-6)) ** 2
    profile = np.clip(1.0 - r2, 0.0, None) ** float(max(power, 0.2))
    return profile.astype(np.float32)


def _segment_profile(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width_start: float,
    width_end: float,
    power: float = 1.0,
) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-6:
        return _ellipse_profile(
            shape,
            x0,
            y0,
            rx=max(width_start, 1.0),
            ry=max(width_start, 1.0),
            power=power,
        )
    t = np.clip(((x - x0) * dx + (y - y0) * dy) / seg_len2, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    width = max(float(width_start), 1e-6) + t * (float(width_end) - float(width_start))
    dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    radial = np.clip(1.0 - (dist / np.maximum(width, 1e-6)) ** 2, 0.0, None) ** float(max(power, 0.2))
    taper = np.sin(np.pi * t) ** 0.7
    return (radial * taper).astype(np.float32)


def _profile_mask(profile: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    return np.asarray(profile > float(threshold), dtype=bool)


def _fusion_centerline(state: SurfaceState) -> tuple[float, float]:
    cy, half_w = _fusion_geometry(state)
    return cy, half_w


def _fusion_geometry(state: SurfaceState) -> tuple[float, float]:
    ys, _ = np.where(state.fusion_boundary)
    if len(ys) == 0:
        h = state.heightmap_mm.shape[0]
        return float(h / 2.0), float(max(2.0, 0.15 * h))
    y_min = float(ys.min())
    y_max = float(ys.max())
    return (0.5 * (y_min + y_max), max(2.0, 0.5 * (y_max - y_min)))


def _region_masks(state: SurfaceState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = state.heightmap_mm.shape
    y, _ = np.mgrid[0:h, 0:w]
    cy, half_w = _fusion_geometry(state)
    dist = np.abs(y - cy)
    weld = dist <= half_w
    keyhole = dist <= max(2.0, 0.25 * half_w)
    haz = np.abs(dist - half_w) <= max(2.0, 0.18 * half_w)
    return weld, keyhole, haz


def _sample_from_mask(
    rng: np.random.Generator,
    mask: np.ndarray,
    radius_px: int = 1,
) -> tuple[int, int]:
    h, w = mask.shape
    if radius_px > 0:
        valid = mask.copy()
        valid[:radius_px, :] = False
        valid[h - radius_px :, :] = False
        valid[:, :radius_px] = False
        valid[:, w - radius_px :] = False
    else:
        valid = mask
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return int(rng.integers(0, w)), int(rng.integers(0, h))
    idx = int(rng.integers(0, len(xs)))
    return int(xs[idx]), int(ys[idx])


def _apply_defect_response(
    state: SurfaceState,
    mask: np.ndarray,
    *,
    yield_scale: float,
    openness: float,
    z_scale: float = 1.0,
) -> None:
    state.yield_modifier[mask] = np.minimum(
        state.yield_modifier[mask], float(yield_scale)
    )
    state.openness[mask] = np.maximum(state.openness[mask], float(openness))
    if z_scale != 1.0:
        state.material_z[mask] = state.material_z[mask] * float(z_scale)


def _apply_section_body(
    state: SurfaceState,
    profile: np.ndarray,
    *,
    yield_scale: float,
    void_fraction: float,
    relief_mm: float,
    z_scale: float = 1.0,
) -> np.ndarray:
    prof = np.clip(np.asarray(profile, dtype=np.float32), 0.0, 1.0)
    mask = prof > 0.08
    state.section_yield_modifier = np.minimum(
        state.section_yield_modifier,
        1.0 - (1.0 - float(yield_scale)) * prof,
    )
    state.section_void_fraction = np.maximum(
        state.section_void_fraction,
        float(void_fraction) * prof,
    )
    state.section_relief_mm += (float(relief_mm) * prof).astype(np.float32)
    if z_scale != 1.0:
        state.section_material_z *= (
            1.0 - (1.0 - float(z_scale)) * prof
        ).astype(np.float32)
    return mask


def _apply_keyhole_or_intrinsic(
    state: SurfaceState,
    rng: np.random.Generator,
    defect_type: str,
    params: dict[str, float],
) -> np.ndarray:
    h, w = state.heightmap_mm.shape
    diameter = params.get("diameter_mm", 0.8)
    depth = params.get("depth_mm", 0.5)
    radius_px = float(max(1.0, (diameter / state.pixel_size_mm) / 2.0))
    radius_px = min(radius_px, float(max(1, min(h, w) // 4)))
    weld_mask, keyhole_mask, _ = _region_masks(state)
    target = keyhole_mask if defect_type == "keyhole_pore" else weld_mask
    cx, cy = _sample_from_mask(rng, target, radius_px=int(np.ceil(radius_px)))

    if defect_type == "keyhole_pore":
        theta = float(rng.normal(0.0, np.radians(10.0)))
        main = _ellipse_profile(
            (h, w),
            cx=cx,
            cy=cy,
            rx=0.72 * radius_px,
            ry=1.45 * radius_px,
            theta=theta,
            power=1.0,
        )
        tail = _segment_profile(
            (h, w),
            x0=cx,
            y0=cy - 0.25 * radius_px,
            x1=cx + float(rng.normal(0.0, 0.18 * radius_px)),
            y1=cy + 1.55 * radius_px,
            width_start=0.32 * radius_px,
            width_end=0.12 * radius_px,
            power=1.1,
        )
        shoulder = _ellipse_profile(
            (h, w),
            cx=cx + float(rng.normal(0.0, 0.10 * radius_px)),
            cy=cy - 0.35 * radius_px,
            rx=0.40 * radius_px,
            ry=0.55 * radius_px,
            theta=theta,
            power=1.2,
        )
        profile = np.clip(0.84 * main + 0.40 * tail + 0.18 * shoulder, 0.0, 1.0)
        state.heightmap_mm -= (depth * profile).astype(np.float32)
        mask = _profile_mask(profile, threshold=0.10)
        body = _ellipse_profile(
            (h, w),
            cx=cx,
            cy=cy + 0.18 * radius_px,
            rx=0.70 * radius_px,
            ry=1.80 * radius_px,
            theta=theta,
            power=1.0,
        )
        core = _ellipse_profile(
            (h, w),
            cx=cx + float(rng.normal(0.0, 0.12 * radius_px)),
            cy=cy + 0.48 * radius_px,
            rx=0.42 * radius_px,
            ry=0.88 * radius_px,
            theta=theta,
            power=1.3,
        )
        section_profile = np.clip(0.74 * body + 0.32 * core, 0.0, 1.0)
        mask |= _apply_section_body(
            state,
            section_profile,
            yield_scale=0.74,
            void_fraction=0.72,
            relief_mm=-0.035 * max(depth, 0.08),
        )
        severe = float(np.clip(depth / max(diameter, 1e-6), 0.0, 2.0))
        _apply_defect_response(
            state,
            mask,
            yield_scale=float(np.clip(0.84 - 0.06 * severe, 0.68, 0.90)),
            openness=float(np.clip(0.24 + 0.10 * severe, 0.18, 0.48)),
        )
        state.defect_depth_um[mask] = np.minimum(
            state.defect_depth_um[mask],
            np.maximum(15.0, depth * 160.0),
        )
    else:
        theta = float(rng.uniform(-np.pi / 2.0, np.pi / 2.0))
        profile = _ellipse_profile(
            (h, w),
            cx=cx,
            cy=cy,
            rx=0.85 * radius_px,
            ry=0.78 * radius_px,
            theta=theta,
            power=1.4,
        )
        state.heightmap_mm -= (0.16 * depth * profile).astype(np.float32)
        mask = _profile_mask(profile, threshold=0.14)
        section_profile = _ellipse_profile(
            (h, w),
            cx=cx + float(rng.normal(0.0, 0.18 * radius_px)),
            cy=cy + float(rng.normal(0.0, 0.22 * radius_px)),
            rx=0.92 * radius_px,
            ry=0.80 * radius_px,
            theta=theta,
            power=1.4,
        )
        mask |= _apply_section_body(
            state,
            section_profile,
            yield_scale=0.84,
            void_fraction=0.48,
            relief_mm=-0.010 * max(depth, 0.08),
            z_scale=0.985,
        )
        depth_um = np.maximum(0.0, depth * 1000.0) + 150.0
        _apply_defect_response(
            state,
            mask,
            yield_scale=0.92,
            openness=float(np.clip(0.05 + 0.04 * (depth / max(diameter, 1e-6)), 0.04, 0.14)),
            z_scale=0.985,
        )
        state.defect_depth_um[mask] = np.minimum(state.defect_depth_um[mask], depth_um)
    state.defect_mask[mask] = True
    return mask


def _apply_crack(
    state: SurfaceState,
    rng: np.random.Generator,
    defect_type: str,
    params: dict[str, float],
) -> np.ndarray:
    h, w = state.heightmap_mm.shape
    length_px = float(max(2.0, params.get("length_mm", 2.0) / state.pixel_size_mm))
    length_px = min(length_px, float(max(2, min(h, w) - 2)))
    width_px = float(max(1.0, params.get("width_mm", 0.1) / state.pixel_size_mm))
    depth_mm = float(params.get("depth_mm", 0.3))
    _, keyhole_mask, haz_mask = _region_masks(state)
    if defect_type == "crack_solidification":
        x0, y0 = _sample_from_mask(rng, keyhole_mask, radius_px=int(np.ceil(width_px + 1)))
        theta = float(rng.normal(0.0, np.radians(8.0)))
    else:
        x0, y0 = _sample_from_mask(rng, haz_mask, radius_px=int(np.ceil(width_px + 1)))
        theta = float(rng.normal(0.0, np.radians(16.0)))
    x1 = float(np.clip(x0 + length_px * np.cos(theta), 0, w - 1))
    y1 = float(np.clip(y0 + length_px * np.sin(theta), 0, h - 1))
    main = _segment_profile(
        (h, w),
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        width_start=1.15 * width_px,
        width_end=0.45 * width_px,
        power=1.2,
    )
    branch = np.zeros((h, w), dtype=np.float32)
    if defect_type == "crack_solidification" and rng.random() < 0.8:
        bx = x0 + 0.55 * (x1 - x0)
        by = y0 + 0.55 * (y1 - y0)
        btheta = theta + float(rng.choice([-1.0, 1.0])) * float(np.radians(rng.uniform(18.0, 35.0)))
        blen = 0.30 * length_px
        branch = _segment_profile(
            (h, w),
            x0=bx,
            y0=by,
            x1=float(np.clip(bx + blen * np.cos(btheta), 0, w - 1)),
            y1=float(np.clip(by + blen * np.sin(btheta), 0, h - 1)),
            width_start=0.65 * width_px,
            width_end=0.22 * width_px,
            power=1.25,
        )
    profile = np.clip(main + 0.45 * branch, 0.0, 1.0)
    state.heightmap_mm -= (depth_mm * profile).astype(np.float32)
    if defect_type == "crack_liquidation":
        state.heightmap_mm -= (0.14 * depth_mm * profile).astype(np.float32)
    mask = _profile_mask(profile, threshold=0.09)
    section_profile = np.clip(0.78 * main + 0.65 * branch, 0.0, 1.0)
    mask |= _apply_section_body(
        state,
        section_profile,
        yield_scale=0.72 if defect_type == "crack_liquidation" else 0.78,
        void_fraction=0.66 if defect_type == "crack_liquidation" else 0.54,
        relief_mm=-0.012 * max(depth_mm, 0.05),
    )
    _apply_defect_response(
        state,
        mask,
        yield_scale=0.76 if defect_type == "crack_liquidation" else 0.81,
        openness=0.36 if defect_type == "crack_liquidation" else 0.28,
    )
    state.defect_depth_um[mask] = np.minimum(
        state.defect_depth_um[mask], np.maximum(10.0, depth_mm * 130.0)
    )
    state.defect_mask[mask] = True
    return mask


def _apply_undercut(state: SurfaceState, rng: np.random.Generator, params: dict[str, float]) -> np.ndarray:
    h, w = state.heightmap_mm.shape
    depth_mm = float(params.get("groove_depth_mm", 0.2))
    toe_angle = np.radians(float(params.get("toe_angle_deg", 25.0)))
    cy, half_w = _fusion_centerline(state)
    side = float(rng.choice([-1.0, 1.0]))
    y_line = float(np.clip(cy + side * (half_w + 1.2 + 1.4 * np.tan(toe_angle)), 1, h - 2))
    x0 = float(rng.integers(int(0.05 * w), int(0.18 * w)))
    x1 = float(rng.integers(int(0.72 * w), int(0.95 * w)))
    y1 = float(np.clip(y_line + rng.normal(0.0, 1.5), 1, h - 2))
    profile = _segment_profile(
        (h, w),
        x0=x0,
        y0=y_line,
        x1=x1,
        y1=y1,
        width_start=2.4,
        width_end=1.6,
        power=1.0,
    )
    shoulder = _ellipse_profile(
        (h, w),
        cx=0.5 * (x0 + x1),
        cy=y_line - side * 1.0,
        rx=0.28 * (x1 - x0),
        ry=2.6,
        theta=0.0,
        power=1.6,
    )
    profile = np.clip(profile + 0.22 * shoulder, 0.0, 1.0)
    state.heightmap_mm -= (depth_mm * profile).astype(np.float32)
    mask = _profile_mask(profile, threshold=0.10)
    section_band = _segment_profile(
        (h, w),
        x0=x0,
        y0=y_line + side * 0.6,
        x1=x1,
        y1=y1 + side * 0.4,
        width_start=2.2,
        width_end=1.8,
        power=1.2,
    )
    mask |= _apply_section_body(
        state,
        section_band,
        yield_scale=0.90,
        void_fraction=0.18,
        relief_mm=-0.016 * max(depth_mm, 0.05),
        z_scale=0.99,
    )
    _apply_defect_response(state, mask, yield_scale=0.93, openness=0.12)
    state.defect_depth_um[mask] = np.minimum(state.defect_depth_um[mask], depth_mm * 1000.0)
    state.defect_mask[mask] = True
    return mask


def _apply_lof_like(
    state: SurfaceState,
    rng: np.random.Generator,
    defect_type: str,
    params: dict[str, float],
) -> np.ndarray:
    h, w = state.heightmap_mm.shape
    extent_px = float(max(2.0, params.get("extent_mm", 2.0) / state.pixel_size_mm))
    extent_px = min(extent_px, float(max(2, min(h, w) // 4)))
    depth_mm = float(params.get("depth_mm", 0.5))
    irregularity = float(params.get("irregularity", 0.4))
    cy, half_w = _fusion_centerline(state)
    side = float(rng.choice([-1.0, 1.0]))
    cx = float(rng.integers(int(0.18 * w), int(0.82 * w)))
    cy_local = float(np.clip(cy + side * (0.86 * half_w + rng.normal(0.0, 1.8)), 1, h - 2))
    theta = float(rng.normal(0.0, np.radians(8.0)))
    lens = _ellipse_profile(
        (h, w),
        cx=cx,
        cy=cy_local,
        rx=1.05 * extent_px,
        ry=max(2.0, 0.34 * extent_px),
        theta=theta,
        power=1.15,
    )
    edge_wear = _segment_profile(
        (h, w),
        x0=cx - 0.95 * extent_px,
        y0=cy_local + side * 0.35 * extent_px,
        x1=cx + 0.95 * extent_px,
        y1=cy_local - side * 0.10 * extent_px,
        width_start=max(1.4, 0.28 * extent_px),
        width_end=max(1.0, 0.22 * extent_px),
        power=1.3,
    )
    profile = np.clip(0.88 * lens + 0.30 * edge_wear, 0.0, 1.0)
    if defect_type == "cold_shut":
        lip = _segment_profile(
            (h, w),
            x0=cx - 0.65 * extent_px,
            y0=cy_local - side * 0.20 * extent_px,
            x1=cx + 0.65 * extent_px,
            y1=cy_local - side * 0.32 * extent_px,
            width_start=max(1.1, 0.18 * extent_px),
            width_end=max(0.9, 0.10 * extent_px),
            power=1.2,
        )
        state.heightmap_mm += (0.10 * depth_mm * lip).astype(np.float32)
        profile = np.clip(profile + 0.16 * lip, 0.0, 1.0)
    noise_field = rng.normal(1.0, 0.10 * max(irregularity, 0.1), size=(h, w)).astype(np.float32)
    profile = np.clip(profile * noise_field, 0.0, 1.0)
    mask = _profile_mask(profile, threshold=0.09)
    section_profile = np.clip(0.82 * lens + 0.42 * edge_wear, 0.0, 1.0)
    if defect_type == "cold_shut":
        section_profile = np.clip(section_profile + 0.24 * lip, 0.0, 1.0)
    mask |= _apply_section_body(
        state,
        section_profile,
        yield_scale=0.80 if defect_type == "lack_of_fusion" else 0.88,
        void_fraction=0.44 if defect_type == "lack_of_fusion" else 0.14,
        relief_mm=-0.014 * max(depth_mm, 0.05),
        z_scale=0.94 if defect_type == "lack_of_fusion" else 0.985,
    )
    state.defect_depth_um[mask] = np.minimum(
        state.defect_depth_um[mask],
        np.maximum(0.0, depth_mm * 1000.0 + 70.0 * rng.normal(0.0, irregularity, size=mask.sum()).astype(np.float32)),
    )
    state.heightmap_mm -= ((0.32 if defect_type == "lack_of_fusion" else 0.22) * depth_mm * profile).astype(np.float32)
    _apply_defect_response(
        state,
        mask,
        yield_scale=0.86 if defect_type == "lack_of_fusion" else 0.89,
        openness=0.18 if defect_type == "lack_of_fusion" else 0.08,
        z_scale=0.96 if defect_type == "lack_of_fusion" else 0.98,
    )
    state.defect_mask[mask] = True
    return mask


def _apply_spiking(state: SurfaceState, rng: np.random.Generator, params: dict[str, float]) -> np.ndarray:
    h, w = state.heightmap_mm.shape
    length_px = float(max(2.0, params.get("spike_length_mm", 1.0) / state.pixel_size_mm))
    width_px = float(max(1.0, params.get("spike_width_mm", 0.1) / state.pixel_size_mm))
    _, keyhole_mask, _ = _region_masks(state)
    cx, cy = _sample_from_mask(rng, keyhole_mask, radius_px=int(np.ceil(width_px + 1)))
    theta = float(rng.normal(np.radians(90.0), np.radians(18.0)))
    profile = _segment_profile(
        (h, w),
        x0=cx,
        y0=cy,
        x1=float(np.clip(cx + length_px * np.cos(theta), 0, w - 1)),
        y1=float(np.clip(cy + length_px * np.sin(theta), 0, h - 1)),
        width_start=1.10 * width_px,
        width_end=0.20 * width_px,
        power=1.25,
    )
    cap = _ellipse_profile(
        (h, w),
        cx=cx,
        cy=cy,
        rx=max(1.0, 0.65 * width_px),
        ry=max(1.0, 1.3 * width_px),
        theta=theta,
        power=1.1,
    )
    profile = np.clip(profile + 0.28 * cap, 0.0, 1.0)
    state.heightmap_mm += (0.18 * profile).astype(np.float32)
    mask = _profile_mask(profile, threshold=0.10)
    section_profile = _segment_profile(
        (h, w),
        x0=cx,
        y0=cy,
        x1=float(np.clip(cx + 0.65 * length_px * np.cos(theta), 0, w - 1)),
        y1=float(np.clip(cy + 0.65 * length_px * np.sin(theta), 0, h - 1)),
        width_start=0.85 * width_px,
        width_end=0.20 * width_px,
        power=1.25,
    )
    mask |= _apply_section_body(
        state,
        section_profile,
        yield_scale=1.02,
        void_fraction=0.02,
        relief_mm=0.010,
        z_scale=1.01,
    )
    _apply_defect_response(state, mask, yield_scale=1.02, openness=0.01)
    state.defect_depth_um[mask] = np.minimum(state.defect_depth_um[mask], 50.0)
    state.defect_mask[mask] = True
    return mask


def apply_defect_plan(
    state: SurfaceState,
    defect_plan: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[DefectInstance]:
    defects: list[DefectInstance] = []
    for item in defect_plan:
        d_type = str(item["type"])
        params = dict(item.get("params", {}))
        if d_type in {"keyhole_pore", "intrinsic_pore"}:
            mask = _apply_keyhole_or_intrinsic(state, rng, d_type, params)
        elif d_type in {"crack_solidification", "crack_liquidation"}:
            mask = _apply_crack(state, rng, d_type, params)
        elif d_type == "undercut":
            mask = _apply_undercut(state, rng, params)
        elif d_type in {"lack_of_fusion", "cold_shut"}:
            mask = _apply_lof_like(state, rng, d_type, params)
        elif d_type == "spiking":
            mask = _apply_spiking(state, rng, params)
        else:
            continue
        bbox = _bbox_from_mask(mask)
        defects.append(
            DefectInstance(
                defect_type=d_type,
                params=params,
                mask=mask.copy(),
                bbox_xyxy=bbox,
                category_id=CATEGORY_IDS[d_type],
            )
        )
    state.recompute_normals()
    return defects
