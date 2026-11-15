from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..models import DefectInstance


_CORE_DEFECTS = {
    "keyhole_pore",
    "intrinsic_pore",
    "crack_solidification",
    "crack_liquidation",
    "lack_of_fusion",
    "undercut",
}


@dataclass(frozen=True)
class StandardsDefaults:
    thickness_t_mm: float
    throat_s_mm: float
    pore_diameter_floor_mm: float
    contrast_detection_threshold: float
    topo_contrast_detection_threshold: float
    confidence_z: float


def load_standards_defaults(config_dir: str | Path) -> StandardsDefaults:
    p = (Path(__file__).resolve().parents[2] / "config" if Path(config_dir) == Path("config") else Path(config_dir).resolve()) / "standards_params.yaml"
    if not p.exists():
        return StandardsDefaults(
            thickness_t_mm=10.0,
            throat_s_mm=5.0,
            pore_diameter_floor_mm=0.5,
            contrast_detection_threshold=0.03,
            topo_contrast_detection_threshold=0.03,
            confidence_z=1.96,
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    iso = raw.get("iso_13919_1", {})
    compo_threshold = float(
        iso.get(
            "compo_contrast_detection_threshold",
            iso.get("contrast_detection_threshold", 0.03),
        )
    )
    return StandardsDefaults(
        thickness_t_mm=float(iso.get("thickness_t_mm", 10.0)),
        throat_s_mm=float(iso.get("throat_s_mm", 5.0)),
        pore_diameter_floor_mm=float(iso.get("pore_diameter_floor_mm", 0.5)),
        contrast_detection_threshold=compo_threshold,
        topo_contrast_detection_threshold=float(
            iso.get("topo_contrast_detection_threshold", compo_threshold)
        ),
        confidence_z=float(iso.get("confidence_z", 1.96)),
    )


def level_b_threshold_mm(
    defect_type: str,
    *,
    t_mm: float,
    s_mm: float,
    pore_floor_mm: float,
) -> float:
    if defect_type in {"crack_solidification", "crack_liquidation", "lack_of_fusion"}:
        return 0.0
    if defect_type in {"keyhole_pore", "intrinsic_pore"}:
        return min(0.4 * s_mm, 3.0, max(pore_floor_mm, 1e-6))
    if defect_type == "undercut":
        return min(0.05 * t_mm, 0.5)
    return 0.0


def defect_size_depth_mm(defect: DefectInstance) -> tuple[float, float]:
    p = defect.params
    if defect.defect_type in {"keyhole_pore", "intrinsic_pore"}:
        return float(p.get("diameter_mm", 0.0)), float(p.get("depth_mm", 0.0))
    if defect.defect_type in {"crack_solidification", "crack_liquidation"}:
        return float(p.get("length_mm", 0.0)), float(p.get("depth_mm", 0.0))
    if defect.defect_type == "lack_of_fusion":
        return float(p.get("extent_mm", 0.0)), float(p.get("depth_mm", 0.0))
    if defect.defect_type == "undercut":
        return float(p.get("groove_depth_mm", 0.0)), float(p.get("groove_depth_mm", 0.0))
    return 0.0, 0.0


def _expand_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(max(0, radius)):
        previous = out.copy()
        out[1:, :] |= previous[:-1, :]
        out[:-1, :] |= previous[1:, :]
        out[:, 1:] |= previous[:, :-1]
        out[:, :-1] |= previous[:, 1:]
    return out


def defect_detected_proxy(
    defect: DefectInstance,
    compo: np.ndarray,
    topo: np.ndarray,
    threshold: float,
    topo_threshold: float | None = None,
) -> tuple[bool, float, float]:
    mask = defect.mask.astype(bool)
    if not np.any(mask):
        return False, 0.0, 0.0
    expanded = _expand_mask(mask, radius=2)
    ring = expanded & (~mask)
    if not np.any(ring):
        ring = ~mask
    in_compo = float(np.mean(compo[mask]))
    out_compo = float(np.mean(compo[ring]))
    in_topo = float(np.mean(topo[mask]))
    out_topo = float(np.mean(topo[ring]))
    c_contrast = abs(in_compo - out_compo)
    t_contrast = abs(in_topo - out_topo)
    topo_thr = float(threshold if topo_threshold is None else topo_threshold)
    detected = bool(c_contrast >= threshold or t_contrast >= topo_thr)
    return detected, c_contrast, t_contrast


def _wilson_interval(k: int, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z / n)
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    lo = float(max(0.0, center - half))
    hi = float(min(1.0, center + half))
    return lo, hi


def _build_bins(threshold_mm: float, sizes_mm: list[float]) -> list[float]:
    thr = float(threshold_mm)
    if thr > 1e-6:
        return [0.0, 0.5 * thr, thr, 1.5 * thr, 2.5 * thr, np.inf]

    # For defect classes without an explicit Level-B size threshold, avoid
    # degenerate [0,0) bins by anchoring bins to observed defect scales.
    sizes = np.array([float(s) for s in sizes_mm if float(s) >= 0.0], dtype=np.float64)
    if sizes.size == 0:
        ref = 0.5
    else:
        ref = float(np.percentile(sizes, 75))
    ref = max(ref, 0.25)
    return [0.0, 0.5 * ref, 0.8 * ref, 1.0 * ref, 1.25 * ref, np.inf]


def _bin_label(lo: float, hi: float) -> str:
    if np.isinf(hi):
        return f">={lo:.3f}"
    return f"[{lo:.3f},{hi:.3f})"


def evaluate_standards_report(
    *,
    defects: list[DefectInstance],
    compo: np.ndarray,
    topo: np.ndarray,
    t_mm: float,
    s_mm: float,
    pore_floor_mm: float,
    contrast_threshold: float,
    topo_contrast_threshold: float | None = None,
    confidence_z: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for defect in defects:
        if defect.defect_type not in _CORE_DEFECTS:
            continue
        size_mm, depth_mm = defect_size_depth_mm(defect)
        threshold_mm = level_b_threshold_mm(
            defect.defect_type,
            t_mm=t_mm,
            s_mm=s_mm,
            pore_floor_mm=pore_floor_mm,
        )
        required = bool(size_mm >= threshold_mm)
        detected, c_contrast, t_contrast = defect_detected_proxy(
            defect,
            compo=compo,
            topo=topo,
            threshold=contrast_threshold,
            topo_threshold=topo_contrast_threshold,
        )
        rows.append(
            {
                "defect_type": defect.defect_type,
                "size_mm": float(size_mm),
                "depth_mm": float(depth_mm),
                "threshold_mm": float(threshold_mm),
                "required_for_level_b_eval": required,
                "detected": detected,
                "pass_detection_requirement": (detected if required else True),
                "compo_contrast": float(c_contrast),
                "topo_contrast": float(t_contrast),
            }
        )

    summary: dict[str, Any] = {}
    pod_bins: dict[str, list[dict[str, Any]]] = {}
    for d_type in sorted({r["defect_type"] for r in rows}):
        d_rows = [r for r in rows if r["defect_type"] == d_type]
        req_rows = [r for r in d_rows if r["required_for_level_b_eval"]]
        req_n = len(req_rows)
        req_k = int(sum(1 for r in req_rows if r["detected"]))
        rate = float(req_k / req_n) if req_n else 0.0
        ci_lo, ci_hi = _wilson_interval(req_k, req_n, confidence_z)
        summary[d_type] = {
            "required_count": req_n,
            "detected_required_count": req_k,
            "detection_rate": rate,
            "detection_rate_ci95": [ci_lo, ci_hi],
        }

        threshold_mm = float(max((r["threshold_mm"] for r in d_rows), default=0.5))
        edges = _build_bins(
            threshold_mm,
            [float(r.get("size_mm", 0.0)) for r in d_rows],
        )
        bins: list[dict[str, Any]] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = [r for r in d_rows if (r["size_mm"] >= lo and r["size_mm"] < hi)]
            n = len(in_bin)
            k = int(sum(1 for r in in_bin if r["detected"]))
            pod = float(k / n) if n else 0.0
            b_lo, b_hi = _wilson_interval(k, n, confidence_z)
            bins.append(
                {
                    "bin": _bin_label(lo, hi),
                    "n": n,
                    "k_detected": k,
                    "pod": pod,
                    "pod_ci95": [b_lo, b_hi],
                }
            )
        pod_bins[d_type] = bins

    return {
        "rows": rows,
        "summary": summary,
        "pod_bins": pod_bins,
        "settings": {
            "t_mm": float(t_mm),
            "s_mm": float(s_mm),
            "pore_floor_mm": float(pore_floor_mm),
            "contrast_threshold": float(contrast_threshold),
            "compo_contrast_threshold": float(contrast_threshold),
            "topo_contrast_threshold": float(
                contrast_threshold
                if topo_contrast_threshold is None
                else topo_contrast_threshold
            ),
            "confidence_z": float(confidence_z),
        },
    }
