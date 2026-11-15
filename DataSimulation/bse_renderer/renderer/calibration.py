from __future__ import annotations

from bisect import bisect_left
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RENDER_MODEL_FORMULA_ONLY = "formula_only"
RENDER_MODEL_FORMULA_CASINO = "formula_casino"
RENDER_MODEL_FORMULA_CASINO_GEOMETRY = "formula_casino_geometry"
RENDER_MODEL_MODES = (
    RENDER_MODEL_FORMULA_ONLY,
    RENDER_MODEL_FORMULA_CASINO,
    RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
)


@dataclass(frozen=True)
class ChannelCalibration:
    scale: float = 1.0
    bias: float = 0.0


@dataclass(frozen=True)
class YieldSurrogateCalibration:
    scale: float = 1.0
    reference_energy_kev: float = 60.0
    energy_exponent: float = 0.0
    obliquity_exponent_scale: float = 1.0
    tilt_deg_knots: tuple[float, ...] = ()
    tilt_scale_knots: tuple[float, ...] = ()


@dataclass(frozen=True)
class BlurSurrogateCalibration:
    scale: float = 1.0
    reference_r50_nm: float | None = None
    reference_r90_nm: float | None = None
    tilt_deg_knots: tuple[float, ...] = ()
    tilt_scale_knots: tuple[float, ...] = ()


@dataclass(frozen=True)
class DepthSurrogateCalibration:
    range_scale: float = 1.0
    reference_z50_nm: float | None = None
    reference_z90_nm: float | None = None
    tilt_deg_knots: tuple[float, ...] = ()
    tilt_scale_knots: tuple[float, ...] = ()


@dataclass(frozen=True)
class DefectSurrogateCalibration:
    pore_visibility_scale: float = 1.0
    groove_topography_scale: float = 1.0
    cluster_visibility_scale: float = 1.0


@dataclass(frozen=True)
class GeometryResponseProfile:
    geometry: str
    yield_scale: float = 1.0
    blur_scale: float = 1.0
    depth_scale: float = 1.0
    topo_gain: float = 1.0
    openness_scale: float = 1.0
    reference_eta_bse: float | None = None
    reference_r50_nm: float | None = None
    reference_z50_nm: float | None = None
    source_sim_id: str | None = None


@dataclass(frozen=True)
class SurrogateCalibration:
    yield_model: YieldSurrogateCalibration
    blur_model: BlurSurrogateCalibration
    depth_model: DepthSurrogateCalibration
    defect_model: DefectSurrogateCalibration
    geometry_profiles: dict[str, GeometryResponseProfile] = field(default_factory=dict)
    source_campaign: str | None = None

    @staticmethod
    def identity(source_campaign: str | None = None) -> "SurrogateCalibration":
        return SurrogateCalibration(
            yield_model=YieldSurrogateCalibration(),
            blur_model=BlurSurrogateCalibration(),
            depth_model=DepthSurrogateCalibration(),
            defect_model=DefectSurrogateCalibration(),
            geometry_profiles={},
            source_campaign=source_campaign,
        )


@dataclass(frozen=True)
class RendererCalibration:
    compo: ChannelCalibration
    topo: ChannelCalibration
    surrogate: SurrogateCalibration
    source: str | None = None

    @staticmethod
    def identity(source: str | None = None) -> "RendererCalibration":
        return RendererCalibration(
            compo=ChannelCalibration(1.0, 0.0),
            topo=ChannelCalibration(1.0, 0.0),
            surrogate=SurrogateCalibration.identity(source_campaign=None),
            source=source,
        )


def interpolate_tilt_scale(
    tilt_deg: float,
    tilt_deg_knots: tuple[float, ...] | list[float],
    tilt_scale_knots: tuple[float, ...] | list[float],
) -> float:
    if len(tilt_deg_knots) != len(tilt_scale_knots) or not tilt_deg_knots:
        return 1.0
    pairs = sorted(
        (
            max(abs(float(deg)), 0.0),
            max(float(scale), 1e-6),
        )
        for deg, scale in zip(tilt_deg_knots, tilt_scale_knots)
    )
    x = max(abs(float(tilt_deg)), 0.0)
    if x <= pairs[0][0]:
        return float(pairs[0][1])
    if x >= pairs[-1][0]:
        return float(pairs[-1][1])
    degrees = [item[0] for item in pairs]
    idx = bisect_left(degrees, x)
    lo_deg, lo_scale = pairs[idx - 1]
    hi_deg, hi_scale = pairs[idx]
    if hi_deg <= lo_deg + 1e-12:
        return float(hi_scale)
    t = (x - lo_deg) / (hi_deg - lo_deg)
    return float(lo_scale + t * (hi_scale - lo_scale))


def interpolate_tilt_scale_map(
    tilt_deg: np.ndarray | float,
    tilt_deg_knots: tuple[float, ...] | list[float],
    tilt_scale_knots: tuple[float, ...] | list[float],
) -> np.ndarray:
    arr = np.asarray(tilt_deg, dtype=np.float64)
    if len(tilt_deg_knots) != len(tilt_scale_knots) or not tilt_deg_knots:
        return np.ones_like(arr, dtype=np.float32)
    pairs = sorted(
        (
            max(abs(float(deg)), 0.0),
            max(float(scale), 1e-6),
        )
        for deg, scale in zip(tilt_deg_knots, tilt_scale_knots)
    )
    degrees = np.asarray([item[0] for item in pairs], dtype=np.float64)
    scales = np.asarray([item[1] for item in pairs], dtype=np.float64)
    flat = np.interp(
        np.abs(arr).reshape(-1),
        degrees,
        scales,
        left=float(scales[0]),
        right=float(scales[-1]),
    )
    return flat.reshape(arr.shape).astype(np.float32)


def parameter_only_calibration(calibration: "RendererCalibration") -> "RendererCalibration":
    return RendererCalibration(
        compo=calibration.compo,
        topo=calibration.topo,
        surrogate=SurrogateCalibration(
            yield_model=calibration.surrogate.yield_model,
            blur_model=calibration.surrogate.blur_model,
            depth_model=calibration.surrogate.depth_model,
            defect_model=DefectSurrogateCalibration(),
            geometry_profiles={},
            source_campaign=calibration.surrogate.source_campaign,
        ),
        source=calibration.source,
    )


def validate_render_model_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    value = str(mode).strip().lower()
    if value not in RENDER_MODEL_MODES:
        raise ValueError(
            f"Unknown render model mode '{mode}'. Expected one of: {', '.join(RENDER_MODEL_MODES)}."
        )
    return value


def infer_render_model_mode(calibration: "RendererCalibration") -> str:
    if calibration.surrogate.geometry_profiles:
        return RENDER_MODEL_FORMULA_CASINO_GEOMETRY
    if calibration == RendererCalibration.identity(source=calibration.source):
        return RENDER_MODEL_FORMULA_ONLY
    return RENDER_MODEL_FORMULA_CASINO


def load_renderer_calibration(path: str | Path | None) -> RendererCalibration:
    if path is None:
        return RendererCalibration.identity(source=None)
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Renderer calibration file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not any(key in raw for key in ("compo", "topo", "surrogate")):
        raise ValueError("Calibration must contain channel or surrogate parameters.")
    metadata = raw.get("metadata") or {}
    if metadata.get("backend") == "mock" or (metadata.get("source") == "casino_calibration" and (
            metadata.get("backend") != "external" or metadata.get("quality_passed") is not True
            or int(metadata.get("num_successful_cases", 0)) < 6 or not metadata.get("case_provenance"))):
        raise ValueError("CASINO calibration lacks passing external-source quality evidence.")
    def check_finite(value):
        if isinstance(value, dict):
            for item in value.values(): check_finite(item)
        elif isinstance(value, list):
            for item in value: check_finite(item)
        elif isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError("Calibration values must be finite.")
    check_finite(raw)
    compo = raw.get("compo", {})
    topo = raw.get("topo", {})
    surrogate = raw.get("surrogate", {})
    yield_model = surrogate.get("yield_model", {})
    blur_model = surrogate.get("blur_model", {})
    depth_model = surrogate.get("depth_model", {})
    defect_model = surrogate.get("defect_model", {})
    geometry_profiles_raw = surrogate.get("geometry_profiles", {})
    geometry_profiles: dict[str, GeometryResponseProfile] = {}
    if isinstance(geometry_profiles_raw, dict):
        for geometry_name, item in geometry_profiles_raw.items():
            if not isinstance(item, dict):
                continue
            geometry_profiles[str(geometry_name)] = GeometryResponseProfile(
                geometry=str(item.get("geometry", geometry_name)),
                yield_scale=float(item.get("yield_scale", 1.0)),
                blur_scale=float(item.get("blur_scale", 1.0)),
                depth_scale=float(item.get("depth_scale", 1.0)),
                topo_gain=float(item.get("topo_gain", 1.0)),
                openness_scale=float(item.get("openness_scale", 1.0)),
                reference_eta_bse=(
                    None
                    if item.get("reference_eta_bse") is None
                    else float(item["reference_eta_bse"])
                ),
                reference_r50_nm=(
                    None
                    if item.get("reference_r50_nm") is None
                    else float(item["reference_r50_nm"])
                ),
                reference_z50_nm=(
                    None
                    if item.get("reference_z50_nm") is None
                    else float(item["reference_z50_nm"])
                ),
                source_sim_id=None if item.get("source_sim_id") is None else str(item["source_sim_id"]),
            )
    return RendererCalibration(
        compo=ChannelCalibration(
            scale=float(compo.get("scale", 1.0)),
            bias=float(compo.get("bias", 0.0)),
        ),
        topo=ChannelCalibration(
            scale=float(topo.get("scale", 1.0)),
            bias=float(topo.get("bias", 0.0)),
        ),
        surrogate=SurrogateCalibration(
            yield_model=YieldSurrogateCalibration(
                scale=float(yield_model.get("scale", 1.0)),
                reference_energy_kev=float(yield_model.get("reference_energy_kev", 60.0)),
                energy_exponent=float(yield_model.get("energy_exponent", 0.0)),
                obliquity_exponent_scale=float(
                    yield_model.get("obliquity_exponent_scale", 1.0)
                ),
                tilt_deg_knots=tuple(float(v) for v in yield_model.get("tilt_deg_knots", []) or []),
                tilt_scale_knots=tuple(
                    float(v) for v in yield_model.get("tilt_scale_knots", []) or []
                ),
            ),
            blur_model=BlurSurrogateCalibration(
                scale=float(blur_model.get("scale", 1.0)),
                reference_r50_nm=(
                    None
                    if blur_model.get("reference_r50_nm") is None
                    else float(blur_model["reference_r50_nm"])
                ),
                reference_r90_nm=(
                    None
                    if blur_model.get("reference_r90_nm") is None
                    else float(blur_model["reference_r90_nm"])
                ),
                tilt_deg_knots=tuple(float(v) for v in blur_model.get("tilt_deg_knots", []) or []),
                tilt_scale_knots=tuple(
                    float(v) for v in blur_model.get("tilt_scale_knots", []) or []
                ),
            ),
            depth_model=DepthSurrogateCalibration(
                range_scale=float(depth_model.get("range_scale", 1.0)),
                reference_z50_nm=(
                    None
                    if depth_model.get("reference_z50_nm") is None
                    else float(depth_model["reference_z50_nm"])
                ),
                reference_z90_nm=(
                    None
                    if depth_model.get("reference_z90_nm") is None
                    else float(depth_model["reference_z90_nm"])
                ),
                tilt_deg_knots=tuple(float(v) for v in depth_model.get("tilt_deg_knots", []) or []),
                tilt_scale_knots=tuple(
                    float(v) for v in depth_model.get("tilt_scale_knots", []) or []
                ),
            ),
            defect_model=DefectSurrogateCalibration(
                pore_visibility_scale=float(defect_model.get("pore_visibility_scale", 1.0)),
                groove_topography_scale=float(
                    defect_model.get("groove_topography_scale", 1.0)
                ),
                cluster_visibility_scale=float(
                    defect_model.get("cluster_visibility_scale", 1.0)
                ),
            ),
            geometry_profiles=geometry_profiles,
            source_campaign=surrogate.get("source_campaign"),
        ),
        source=str(p),
    )


def resolve_renderer_calibration(
    path: str | Path | None,
    *,
    render_model_mode: str | None = None,
) -> RendererCalibration:
    mode = validate_render_model_mode(render_model_mode)
    if mode is None:
        if path is None:
            return RendererCalibration.identity(source=None)
        return load_renderer_calibration(path)
    if mode == RENDER_MODEL_FORMULA_ONLY:
        source = None if path is None else str(Path(path).resolve())
        return RendererCalibration.identity(source=source)
    if path is None:
        raise ValueError(
            f"Render model '{mode}' requires a renderer calibration JSON path."
        )
    loaded = load_renderer_calibration(path)
    if mode == RENDER_MODEL_FORMULA_CASINO:
        return parameter_only_calibration(loaded)
    return loaded


def save_renderer_calibration(
    path: str | Path,
    *,
    compo_scale: float,
    compo_bias: float,
    topo_scale: float,
    topo_bias: float,
    surrogate: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "compo": {"scale": float(compo_scale), "bias": float(compo_bias)},
        "topo": {"scale": float(topo_scale), "bias": float(topo_bias)},
    }
    if surrogate:
        payload["surrogate"] = surrogate
    if metadata:
        payload["metadata"] = metadata
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(p)
