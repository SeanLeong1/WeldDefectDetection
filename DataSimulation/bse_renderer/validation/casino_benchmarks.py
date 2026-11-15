from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import default_beam_runtime, load_config_bundle
from ..physics.bse_yield import eta_z
from ..physics.interaction_volume import kanaya_okayama_range
from ..renderer.calibration import interpolate_tilt_scale, save_renderer_calibration


@dataclass(frozen=True)
class CasinoBenchmarkRecord:
    sim_id: str
    material: str
    geometry: str
    energy_kev: float
    tilt_deg: float
    beam_diameter_nm: float
    electrons: int
    eta_bse: float
    mean_bse_energy_kev: float
    peak_bse_angle_deg: float
    r50_nm: float
    r90_nm: float
    z50_nm: float
    z90_nm: float
    notes: str
    quality_flag: str
    defect_signal_proxy: float | None = None
    source_note: str | None = None
    artifact_root: str | None = None


@dataclass(frozen=True)
class CasinoBenchmarkCampaign:
    schema_version: int
    campaign_id: str
    records: tuple[CasinoBenchmarkRecord, ...]
    source_archive: str | None = None
    source_campaign_note: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class CasinoAssemblyDefaults:
    material: str | None = None
    energy_kev: float | None = None
    beam_diameter_nm: float | None = None
    electrons: int | None = None
    quality_flag: str = "usable"
    notes: str = ""


@dataclass(frozen=True)
class CasinoAssemblyRunSpec:
    run_id: str
    artifact_root: str
    geometry: str
    tilt_deg: float
    material: str | None = None
    energy_kev: float | None = None
    beam_diameter_nm: float | None = None
    electrons: int | None = None
    notes: str = ""
    quality_flag: str = "usable"
    defect_signal_proxy: float | None = None
    source_note: str | None = None
    summary_dat: str | None = None
    bs_angle_dat: str | None = None
    bs_energy_dat: str | None = None
    surface_radius_dat: str | None = None
    zmax_bs_dat: str | None = None


@dataclass(frozen=True)
class CasinoAssemblyRecordSpec:
    sim_id: str
    from_runs: tuple[str, ...]
    geometry: str
    tilt_deg: float
    material: str | None = None
    energy_kev: float | None = None
    beam_diameter_nm: float | None = None
    electrons: int | None = None
    notes: str = ""
    quality_flag: str = "usable"
    defect_signal_proxy: float | None = None
    source_note: str | None = None
    reducer: str = "median"


@dataclass(frozen=True)
class CasinoAssemblyConfig:
    schema_version: int
    campaign_id: str
    runs: tuple[CasinoAssemblyRunSpec, ...]
    records: tuple[CasinoAssemblyRecordSpec, ...]
    auto_single_run_records: bool = True
    defaults: CasinoAssemblyDefaults = CasinoAssemblyDefaults()
    source_archive: str | None = None
    source_campaign_note: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ParsedCasinoRunMetrics:
    run_id: str
    artifact_root: str
    summary_dat_path: str
    bs_angle_dat_path: str
    bs_energy_dat_path: str
    surface_radius_dat_path: str
    zmax_bs_dat_path: str | None
    eta_bse: float
    mean_bse_energy_kev: float
    peak_bse_energy_kev: float
    mean_bse_angle_deg: float
    peak_bse_angle_deg: float
    r50_nm: float
    r90_nm: float
    z50_nm: float
    z90_nm: float
    depth_source: str
    summary_metadata: dict[str, Any]


_SIM_ID_PATTERN = re.compile(
    r"^(?:(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{2}-\d{2})_)?"
    r"(?P<material>[^_]+)_(?P<geometry>.+)_(?P<energy>\d+(?:\.\d+)?)keV_tilt"
    r"(?P<tilt>-?\d+(?:\.\d+)?)(?:_(?P<suffix>.+))?$"
)


def _require_positive(name: str, value: Any) -> float:
    out = float(value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return out


def _require_non_negative(name: str, value: Any) -> float:
    out = float(value)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return out


def _parse_tabbed_key_value(line: str) -> tuple[str, list[str]] | None:
    raw = line.strip()
    if not raw or raw.startswith("=") or raw.startswith("-") or raw.startswith("#"):
        return None
    parts = [part.strip().strip('"') for part in raw.split("\t")]
    parts = [part for part in parts if part != ""]
    if not parts:
        return None
    return parts[0], parts[1:]


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_scale_vector(text: str) -> tuple[float, float, float] | None:
    m = re.match(r"\(([^,]+),([^,]+),([^)]+)\)", text.strip())
    if not m:
        return None
    vals = tuple(float(part.strip()) for part in m.groups())
    return vals


def _load_distribution_file(path: str | Path) -> tuple[str, tuple[str, str], np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Distribution file '{path}' is too short to parse.")
    title = lines[0].strip()
    headers = [h.strip() for h in lines[1].split("\t")]
    if len(headers) < 2:
        raise ValueError(f"Distribution file '{path}' does not contain two tab-separated columns.")
    rows: list[tuple[float, float]] = []
    for line in lines[2:]:
        raw = line.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split("\t")]
        if len(parts) < 2:
            continue
        x = _safe_float(parts[0])
        y = _safe_float(parts[1])
        if x is None or y is None:
            continue
        rows.append((x, y))
    if not rows:
        raise ValueError(f"Distribution file '{path}' did not contain any numeric data rows.")
    return title, (headers[0], headers[1]), np.asarray(rows, dtype=np.float64)


def _weighted_mean_and_peak(xy: np.ndarray) -> tuple[float, float]:
    x = xy[:, 0]
    w = xy[:, 1]
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("Distribution weights sum to zero.")
    mean = float(np.sum(x * w) / total)
    peak = float(x[int(np.argmax(w))])
    return mean, peak


def _quantiles_from_density(xy: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    if xy.shape[0] < 2:
        raise ValueError("Need at least two rows to integrate density distribution.")
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.diff(x)
    areas = y[1:] * dx
    total = float(np.sum(areas))
    if total <= 0.0:
        raise ValueError("Integrated density is zero.")
    cdf = np.cumsum(areas) / total
    out: list[float] = []
    for q in quantiles:
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), len(x) - 1)
        out.append(float(x[idx + 1 if idx + 1 < len(x) else idx]))
    return out


def _quantiles_from_histogram(xy: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    x = xy[:, 0]
    y = xy[:, 1]
    total = float(np.sum(y))
    if total <= 0.0:
        raise ValueError("Histogram weights sum to zero.")
    cdf = np.cumsum(y) / total
    out: list[float] = []
    for q in quantiles:
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), len(x) - 1)
        out.append(float(x[idx]))
    return out


def _parse_casino_summary_dat(path: str | Path) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    meta: dict[str, Any] = {
        "source_path": str(Path(path).resolve()),
    }
    current_section: str | None = None
    current_trajectory: int | None = None
    trajectories: dict[int, list[float]] = {}
    in_collisions = False
    for line in lines:
        stripped = line.strip().strip('"')
        if stripped == "Sample Info :":
            current_section = "sample"
            continue
        if stripped == "Regions Info :":
            current_section = "regions"
            continue
        if stripped == "Element Info :":
            current_section = "elements"
            continue
        kv = _parse_tabbed_key_value(line)
        if kv is None:
            continue
        key, values = kv
        if key == "Trajectory":
            current_trajectory = int(float(values[0]))
            trajectories.setdefault(current_trajectory, [])
            in_collisions = False
            continue
        if key == "Collisions":
            in_collisions = True
            continue
        if in_collisions and current_trajectory is not None:
            first = _safe_float(key)
            if first is not None and len(values) >= 7:
                z_val = _safe_float(values[1])
                if z_val is not None:
                    trajectories[current_trajectory].append(z_val)
                continue
        if key == "Name" and current_section == "regions" and values:
            region_names = [value for value in values if value and "Outside" not in value]
            if region_names and "material" not in meta:
                meta["material"] = region_names[0]
        elif key == "Scale (x, y, z)" and values:
            vec = _parse_scale_vector(values[0])
            if vec is not None:
                meta["scale_nm"] = vec
        elif key == "Rho" and current_section == "regions" and values:
            rho = _safe_float(values[0])
            if rho is not None:
                meta["density_g_cm3"] = rho
        elif key in {"initial energy", "Initial Energy"} and values:
            val = _safe_float(values[0])
            if val is not None:
                meta["energy_kev"] = val
        elif key == "BackScattering Coefficient" and values:
            val = _safe_float(values[0])
            if val is not None:
                meta["eta_bse"] = val
        elif key == "nb backscattered" and values:
            val = _safe_float(values[0])
            if val is not None:
                meta["num_backscattered"] = int(val)
        elif key == "nb transmitted" and values:
            val = _safe_float(values[0])
            if val is not None:
                meta["num_transmitted"] = int(val)
        elif key == "Nb Simulated Trajectories" and values:
            val = _safe_float(values[0])
            if val is not None:
                meta["electrons"] = int(val)
    meta["trajectory_z_values"] = trajectories
    return meta


def _fallback_depth_quantiles_from_summary_dat(path: str | Path) -> tuple[float, float]:
    meta = _parse_casino_summary_dat(path)
    trajectory_z_values = meta.get("trajectory_z_values", {})
    depths: list[float] = []
    for z_values in trajectory_z_values.values():
        if len(z_values) < 2:
            continue
        entry_z = z_values[0]
        max_depth = max(abs(z - entry_z) for z in z_values)
        if max_depth > 0.0:
            depths.append(float(max_depth))
    if not depths:
        raise ValueError("No usable trajectory depth values found in summary .dat fallback.")
    arr = np.asarray(depths, dtype=np.float64)
    return float(np.quantile(arr, 0.5)), float(np.quantile(arr, 0.9))


def _normalize_artifact_root(export_root: str | Path, artifact_root: str) -> Path:
    p = Path(artifact_root)
    if p.is_absolute():
        return p.resolve()
    return (Path(export_root).resolve() / p).resolve()


def _discover_graph_file(run_dir: Path, explicit: str | None, suffix: str) -> Path:
    if explicit:
        p = Path(explicit)
        return p.resolve() if p.is_absolute() else (run_dir / p).resolve()
    suffix_lower = suffix.lower()
    matches = sorted(
        p
        for p in run_dir.glob(f"*{suffix}")
        if p.name.lower() == suffix_lower or p.name.lower().endswith(f"_{suffix_lower}")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '*{suffix}' file in '{run_dir}', found {len(matches)}."
        )
    return matches[0].resolve()


def _discover_optional_graph_file(run_dir: Path, explicit: str | None, suffix: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        resolved = p.resolve() if p.is_absolute() else (run_dir / p).resolve()
        if not resolved.exists():
            raise ValueError(f"Expected optional file '{resolved}' does not exist.")
        return resolved
    suffix_lower = suffix.lower()
    matches = sorted(
        p
        for p in run_dir.glob(f"*{suffix}")
        if p.name.lower() == suffix_lower or p.name.lower().endswith(f"_{suffix_lower}")
    )
    if len(matches) > 1:
        raise ValueError(
            f"Expected at most one '*{suffix}' file in '{run_dir}', found {len(matches)}."
        )
    if not matches:
        return None
    return matches[0].resolve()


def _discover_summary_dat(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return p.resolve() if p.is_absolute() else (run_dir / p).resolve()
    dat_files = sorted(run_dir.glob("*.dat"))
    excluded_suffixes = {
        "bsangle.dat",
        "bsenergy.dat",
        "surfaceradiusofbe.dat",
        "energyofsurfaceradiusofbe.dat",
        "transmittedenergy.dat",
        "zmaxbs.dat",
        "zmax.dat",
    }
    candidates = [p for p in dat_files if not any(p.name.lower().endswith(sfx) for sfx in excluded_suffixes)]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one summary .dat in '{run_dir}', found {len(candidates)}."
        )
    return candidates[0].resolve()


def parse_casino_export_run(
    *,
    run_id: str,
    artifact_root: str | Path,
    summary_dat: str | Path | None = None,
    bs_angle_dat: str | Path | None = None,
    bs_energy_dat: str | Path | None = None,
    surface_radius_dat: str | Path | None = None,
    zmax_bs_dat: str | Path | None = None,
) -> ParsedCasinoRunMetrics:
    run_dir = Path(artifact_root).resolve()
    summary_path = _discover_summary_dat(run_dir, None if summary_dat is None else str(summary_dat))
    angle_path = _discover_graph_file(run_dir, None if bs_angle_dat is None else str(bs_angle_dat), "BSAngle.dat")
    energy_path = _discover_graph_file(run_dir, None if bs_energy_dat is None else str(bs_energy_dat), "BSEnergy.dat")
    radius_path = _discover_graph_file(
        run_dir,
        None if surface_radius_dat is None else str(surface_radius_dat),
        "SurfaceRadiusofBE.dat",
    )
    z_path = _discover_optional_graph_file(
        run_dir,
        None if zmax_bs_dat is None else str(zmax_bs_dat),
        "zmaxBS.dat",
    )

    summary_meta = _parse_casino_summary_dat(summary_path)
    if "eta_bse" not in summary_meta:
        raise ValueError(f"Could not find BackScattering Coefficient in '{summary_path}'.")

    _, _, angle_xy = _load_distribution_file(angle_path)
    mean_angle_deg, peak_angle_deg = _weighted_mean_and_peak(angle_xy)

    _, _, energy_xy = _load_distribution_file(energy_path)
    mean_energy_kev, peak_energy_kev = _weighted_mean_and_peak(energy_xy)

    _, radius_headers, radius_xy = _load_distribution_file(radius_path)
    radius_axis, radius_value = radius_headers
    if "Radius" not in radius_axis or "/ nm" not in radius_value:
        raise ValueError(f"Unexpected surface-radius headers in '{radius_path}': {radius_headers!r}")
    r50_nm, r90_nm = _quantiles_from_density(radius_xy, (0.5, 0.9))

    if z_path is None:
        z50_nm, z90_nm = _fallback_depth_quantiles_from_summary_dat(summary_path)
        depth_source = "summary_dat_fallback_missing_zmax_bs_dat"
    else:
        _, _, z_xy = _load_distribution_file(z_path)
        nonzero_bins = int(np.count_nonzero(z_xy[:, 1] > 0.0))
        if nonzero_bins <= 1:
            z50_nm, z90_nm = _fallback_depth_quantiles_from_summary_dat(summary_path)
            depth_source = "summary_dat_fallback"
        else:
            z50_nm, z90_nm = _quantiles_from_histogram(z_xy, (0.5, 0.9))
            depth_source = "zmax_bs_dat"

    return ParsedCasinoRunMetrics(
        run_id=run_id,
        artifact_root=str(run_dir),
        summary_dat_path=str(summary_path),
        bs_angle_dat_path=str(angle_path),
        bs_energy_dat_path=str(energy_path),
        surface_radius_dat_path=str(radius_path),
        zmax_bs_dat_path=None if z_path is None else str(z_path),
        eta_bse=float(summary_meta["eta_bse"]),
        mean_bse_energy_kev=mean_energy_kev,
        peak_bse_energy_kev=peak_energy_kev,
        mean_bse_angle_deg=mean_angle_deg,
        peak_bse_angle_deg=peak_angle_deg,
        r50_nm=r50_nm,
        r90_nm=r90_nm,
        z50_nm=z50_nm,
        z90_nm=z90_nm,
        depth_source=depth_source,
        summary_metadata=summary_meta,
    )


def _infer_run_context(run_name: str) -> dict[str, Any]:
    match = _SIM_ID_PATTERN.match(run_name)
    if not match:
        return {}
    groups = match.groupdict()
    inferred: dict[str, Any] = {
        "material": groups["material"],
        "geometry": groups["geometry"],
        "energy_kev": float(groups["energy"]),
        "tilt_deg": float(groups["tilt"]),
    }
    if groups.get("suffix"):
        inferred["variant_suffix"] = groups["suffix"]
    return inferred


def _build_obsidian_value_table(values: dict[str, Any]) -> str:
    ordered_keys = (
        "eta_bse",
        "mean_bse_energy_kev",
        "peak_bse_energy_kev",
        "mean_bse_angle_deg",
        "peak_bse_angle_deg",
        "r50_nm",
        "r90_nm",
        "z50_nm",
        "z90_nm",
        "quality_flag",
    )
    rows = ["| Property | Value |", "|---|---:|"]
    for key in ordered_keys:
        value = values.get(key, "")
        rows.append(f"| `{key}` | {value} |")
    return "\n".join(rows)


def extract_casino_run_note_values(
    *,
    artifact_root: str | Path,
    run_id: str | None = None,
    material: str | None = None,
    geometry: str | None = None,
    energy_kev: float | None = None,
    tilt_deg: float | None = None,
    beam_diameter_nm: float | None = None,
    electrons: int | None = None,
    quality_flag: str = "usable",
    summary_dat: str | Path | None = None,
    bs_angle_dat: str | Path | None = None,
    bs_energy_dat: str | Path | None = None,
    surface_radius_dat: str | Path | None = None,
    zmax_bs_dat: str | Path | None = None,
) -> dict[str, Any]:
    artifact_path = Path(artifact_root).resolve()
    inferred_name = run_id or artifact_path.name
    inferred_context = _infer_run_context(inferred_name)
    parsed = parse_casino_export_run(
        run_id=inferred_name,
        artifact_root=artifact_path,
        summary_dat=summary_dat,
        bs_angle_dat=bs_angle_dat,
        bs_energy_dat=bs_energy_dat,
        surface_radius_dat=surface_radius_dat,
        zmax_bs_dat=zmax_bs_dat,
    )
    summary_meta = parsed.summary_metadata
    extracted: dict[str, Any] = {
        "sim_id": inferred_name,
        "artifact_root": str(artifact_path),
        "material": material or inferred_context.get("material") or summary_meta.get("material"),
        "geometry": geometry or inferred_context.get("geometry"),
        "energy_kev": (
            float(energy_kev)
            if energy_kev is not None
            else float(inferred_context.get("energy_kev", summary_meta.get("energy_kev", 0.0)))
        ),
        "tilt_deg": (
            float(tilt_deg)
            if tilt_deg is not None
            else float(inferred_context.get("tilt_deg", 0.0))
        ),
        "beam_diameter_nm": (
            float(beam_diameter_nm) if beam_diameter_nm is not None else None
        ),
        "electrons": (
            int(electrons)
            if electrons is not None
            else (
                int(summary_meta["electrons"])
                if summary_meta.get("electrons") is not None
                else None
            )
        ),
        "eta_bse": parsed.eta_bse,
        "mean_bse_energy_kev": parsed.mean_bse_energy_kev,
        "peak_bse_energy_kev": parsed.peak_bse_energy_kev,
        "mean_bse_angle_deg": parsed.mean_bse_angle_deg,
        "peak_bse_angle_deg": parsed.peak_bse_angle_deg,
        "r50_nm": parsed.r50_nm,
        "r90_nm": parsed.r90_nm,
        "z50_nm": parsed.z50_nm,
        "z90_nm": parsed.z90_nm,
        "quality_flag": quality_flag,
        "depth_source": parsed.depth_source,
        "source_files": {
            "summary_dat": parsed.summary_dat_path,
            "bs_angle_dat": parsed.bs_angle_dat_path,
            "bs_energy_dat": parsed.bs_energy_dat_path,
            "surface_radius_dat": parsed.surface_radius_dat_path,
            "zmax_bs_dat": parsed.zmax_bs_dat_path,
        },
    }
    missing_fields = [
        field_name
        for field_name in ("material", "geometry", "energy_kev", "tilt_deg", "beam_diameter_nm", "electrons")
        if extracted.get(field_name) in (None, "")
    ]
    extracted["missing_fields"] = missing_fields
    extracted["obsidian_fill_table"] = _build_obsidian_value_table(extracted)
    return extracted


def _load_assembly_run(raw: dict[str, Any], defaults: CasinoAssemblyDefaults) -> CasinoAssemblyRunSpec:
    run_id = str(raw["run_id"]).strip()
    if not run_id:
        raise ValueError("Assembly run spec must have a non-empty run_id.")
    return CasinoAssemblyRunSpec(
        run_id=run_id,
        artifact_root=str(raw["artifact_root"]).strip(),
        geometry=str(raw["geometry"]).strip(),
        tilt_deg=_require_non_negative("tilt_deg", raw["tilt_deg"]),
        material=None if raw.get("material") is None else str(raw["material"]).strip(),
        energy_kev=(
            None if raw.get("energy_kev") is None else _require_positive("energy_kev", raw["energy_kev"])
        ),
        beam_diameter_nm=(
            None
            if raw.get("beam_diameter_nm") is None
            else _require_positive("beam_diameter_nm", raw["beam_diameter_nm"])
        ),
        electrons=(
            None if raw.get("electrons") is None else int(_require_positive("electrons", raw["electrons"]))
        ),
        notes=str(raw.get("notes", defaults.notes)),
        quality_flag=str(raw.get("quality_flag", defaults.quality_flag)).strip() or defaults.quality_flag,
        defect_signal_proxy=(
            None
            if raw.get("defect_signal_proxy") is None
            else _require_positive("defect_signal_proxy", raw["defect_signal_proxy"])
        ),
        source_note=None if raw.get("source_note") is None else str(raw["source_note"]),
        summary_dat=None if raw.get("summary_dat") is None else str(raw["summary_dat"]),
        bs_angle_dat=None if raw.get("bs_angle_dat") is None else str(raw["bs_angle_dat"]),
        bs_energy_dat=None if raw.get("bs_energy_dat") is None else str(raw["bs_energy_dat"]),
        surface_radius_dat=(
            None if raw.get("surface_radius_dat") is None else str(raw["surface_radius_dat"])
        ),
        zmax_bs_dat=None if raw.get("zmax_bs_dat") is None else str(raw["zmax_bs_dat"]),
    )


def _load_assembly_record(raw: dict[str, Any], defaults: CasinoAssemblyDefaults) -> CasinoAssemblyRecordSpec:
    sim_id = str(raw["sim_id"]).strip()
    if not sim_id:
        raise ValueError("Assembly record spec must have a non-empty sim_id.")
    from_runs_raw = raw.get("from_runs", [])
    if not isinstance(from_runs_raw, list) or not from_runs_raw:
        raise ValueError(f"Assembly record '{sim_id}' must contain a non-empty from_runs list.")
    reducer = str(raw.get("reducer", "median")).strip().lower() or "median"
    if reducer not in {"median", "mean"}:
        raise ValueError(f"Unsupported reducer '{reducer}' for assembly record '{sim_id}'.")
    return CasinoAssemblyRecordSpec(
        sim_id=sim_id,
        from_runs=tuple(str(item).strip() for item in from_runs_raw),
        geometry=str(raw["geometry"]).strip(),
        tilt_deg=_require_non_negative("tilt_deg", raw["tilt_deg"]),
        material=None if raw.get("material") is None else str(raw["material"]).strip(),
        energy_kev=(
            None if raw.get("energy_kev") is None else _require_positive("energy_kev", raw["energy_kev"])
        ),
        beam_diameter_nm=(
            None
            if raw.get("beam_diameter_nm") is None
            else _require_positive("beam_diameter_nm", raw["beam_diameter_nm"])
        ),
        electrons=(
            None if raw.get("electrons") is None else int(_require_positive("electrons", raw["electrons"]))
        ),
        notes=str(raw.get("notes", defaults.notes)),
        quality_flag=str(raw.get("quality_flag", defaults.quality_flag)).strip() or defaults.quality_flag,
        defect_signal_proxy=(
            None
            if raw.get("defect_signal_proxy") is None
            else _require_positive("defect_signal_proxy", raw["defect_signal_proxy"])
        ),
        source_note=None if raw.get("source_note") is None else str(raw["source_note"]),
        reducer=reducer,
    )


def load_casino_assembly_config(path: str | Path) -> CasinoAssemblyConfig:
    p = Path(path).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    defaults_raw = raw.get("defaults", {})
    defaults = CasinoAssemblyDefaults(
        material=None if defaults_raw.get("material") is None else str(defaults_raw["material"]).strip(),
        energy_kev=(
            None
            if defaults_raw.get("energy_kev") is None
            else _require_positive("energy_kev", defaults_raw["energy_kev"])
        ),
        beam_diameter_nm=(
            None
            if defaults_raw.get("beam_diameter_nm") is None
            else _require_positive("beam_diameter_nm", defaults_raw["beam_diameter_nm"])
        ),
        electrons=(
            None
            if defaults_raw.get("electrons") is None
            else int(_require_positive("electrons", defaults_raw["electrons"]))
        ),
        quality_flag=str(defaults_raw.get("quality_flag", "usable")).strip() or "usable",
        notes=str(defaults_raw.get("notes", "")),
    )
    runs_raw = raw.get("runs", [])
    records_raw = raw.get("records", [])
    auto_single_run_records = bool(raw.get("auto_single_run_records", False))
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("Assembly config must contain a non-empty runs list.")
    if not isinstance(records_raw, list):
        raise ValueError("Assembly config 'records' must be a list.")
    if not records_raw and not auto_single_run_records:
        raise ValueError(
            "Assembly config must contain a non-empty records list unless auto_single_run_records is true."
        )
    runs = tuple(_load_assembly_run(item, defaults) for item in runs_raw)
    records = tuple(_load_assembly_record(item, defaults) for item in records_raw)
    return CasinoAssemblyConfig(
        schema_version=int(raw.get("schema_version", 1)),
        campaign_id=str(raw.get("campaign_id", p.stem)),
        runs=runs,
        records=records,
        auto_single_run_records=auto_single_run_records,
        defaults=defaults,
        source_archive=None if raw.get("source_archive") is None else str(raw["source_archive"]),
        source_campaign_note=(
            None if raw.get("source_campaign_note") is None else str(raw["source_campaign_note"])
        ),
        description=None if raw.get("description") is None else str(raw["description"]),
    )


def _load_record(raw: dict[str, Any]) -> CasinoBenchmarkRecord:
    sim_id = str(raw["sim_id"]).strip()
    if not sim_id:
        raise ValueError("sim_id must be non-empty")
    return CasinoBenchmarkRecord(
        sim_id=sim_id,
        material=str(raw["material"]).strip(),
        geometry=str(raw["geometry"]).strip(),
        energy_kev=_require_positive("energy_kev", raw["energy_kev"]),
        tilt_deg=_require_non_negative("tilt_deg", raw["tilt_deg"]),
        beam_diameter_nm=_require_positive("beam_diameter_nm", raw["beam_diameter_nm"]),
        electrons=int(_require_positive("electrons", raw["electrons"])),
        eta_bse=_require_positive("eta_bse", raw["eta_bse"]),
        mean_bse_energy_kev=_require_positive(
            "mean_bse_energy_kev", raw["mean_bse_energy_kev"]
        ),
        peak_bse_angle_deg=_require_non_negative(
            "peak_bse_angle_deg", raw["peak_bse_angle_deg"]
        ),
        r50_nm=_require_positive("r50_nm", raw["r50_nm"]),
        r90_nm=_require_positive("r90_nm", raw["r90_nm"]),
        z50_nm=_require_positive("z50_nm", raw["z50_nm"]),
        z90_nm=_require_positive("z90_nm", raw["z90_nm"]),
        notes=str(raw.get("notes", "")),
        quality_flag=str(raw.get("quality_flag", "usable")).strip() or "usable",
        defect_signal_proxy=(
            None
            if raw.get("defect_signal_proxy") is None
            else _require_positive("defect_signal_proxy", raw["defect_signal_proxy"])
        ),
        source_note=None if raw.get("source_note") is None else str(raw["source_note"]),
        artifact_root=None if raw.get("artifact_root") is None else str(raw["artifact_root"]),
    )


def load_casino_benchmark_campaign(path: str | Path) -> CasinoBenchmarkCampaign:
    p = Path(path).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    records_raw = raw.get("records", [])
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("Benchmark campaign must contain a non-empty 'records' list.")
    records = tuple(_load_record(item) for item in records_raw)
    return CasinoBenchmarkCampaign(
        schema_version=int(raw.get("schema_version", 1)),
        campaign_id=str(raw.get("campaign_id", p.stem)),
        records=records,
        source_archive=None if raw.get("source_archive") is None else str(raw["source_archive"]),
        source_campaign_note=(
            None
            if raw.get("source_campaign_note") is None
            else str(raw["source_campaign_note"])
        ),
        description=None if raw.get("description") is None else str(raw["description"]),
    )


def _is_usable(record: CasinoBenchmarkRecord) -> bool:
    return record.quality_flag.lower() not in {"invalid", "template", "reject", "rejected"}


def _pick_material_preset(campaign: CasinoBenchmarkCampaign, config_dir: str | Path):
    cfg = load_config_bundle(config_dir)
    usable = [r for r in campaign.records if _is_usable(r)]
    if not usable:
        raise ValueError("No usable benchmark records found.")
    material_name = usable[0].material
    if material_name in cfg.material.presets:
        return cfg, cfg.material.presets[material_name]
    aliases = {name.lower(): preset for name, preset in cfg.material.presets.items()}
    if material_name.lower() in aliases:
        return cfg, aliases[material_name.lower()]
    raise ValueError(
        f"Material '{material_name}' not found in config presets. Add it to material_params.yaml first."
    )


def _median(values: list[float], default: float) -> float:
    if not values:
        return float(default)
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _fit_obliquity_scale(
    baseline_eta: float,
    baseline_record: CasinoBenchmarkRecord,
    records: list[CasinoBenchmarkRecord],
    material_z: float,
) -> float:
    default_n = 9.0 / np.sqrt(max(material_z, 1e-6))
    fitted_scales: list[float] = []
    for rec in records:
        if rec.geometry != baseline_record.geometry or rec.energy_kev != baseline_record.energy_kev:
            continue
        if rec.tilt_deg <= 1e-6:
            continue
        ratio = rec.eta_bse / max(baseline_eta, 1e-9)
        cos_term = np.cos(np.radians(rec.tilt_deg))
        if not np.isfinite(ratio) or ratio <= 0.0 or cos_term <= 1e-6 or abs(cos_term - 1.0) <= 1e-9:
            continue
        observed_n = np.log(ratio) / (-np.log(cos_term))
        if np.isfinite(observed_n) and observed_n > 0.0:
            fitted_scales.append(float(observed_n / max(default_n, 1e-9)))
    return float(np.clip(_median(fitted_scales, 1.0), 0.25, 4.0))


def _default_blur_sigma_px(config_dir: str | Path, energy_kev: float, pixel_size_mm: float) -> float:
    cfg = load_config_bundle(config_dir)
    beam = default_beam_runtime(cfg.beam)
    spot_px = max(beam.focal_spot_mm, 0.0) / max(pixel_size_mm, 1e-6)
    energy_term = np.clip((max(energy_kev, 1e-6) / 60.0) ** 0.6, 0.5, 2.2)
    return float(np.clip(0.10 * spot_px + 0.35 * energy_term, 0.0, 4.0))


def _fit_defect_scale(
    records: list[CasinoBenchmarkRecord],
    geometry: str,
    default: float = 1.0,
) -> float:
    vals = [r.defect_signal_proxy for r in records if r.geometry == geometry and r.defect_signal_proxy]
    if not vals:
        return float(default)
    return float(np.clip(_median([float(v) for v in vals], default), 0.25, 4.0))


def _fit_tilt_response_curve(
    records: list[CasinoBenchmarkRecord],
    baseline_record: CasinoBenchmarkRecord,
    value_getter,
) -> tuple[list[float], list[float]]:
    baseline_value = float(value_getter(baseline_record))
    if baseline_value <= 0.0:
        return [0.0], [1.0]
    grouped: dict[float, list[float]] = {0.0: [1.0]}
    for rec in records:
        if rec.geometry != baseline_record.geometry or rec.energy_kev != baseline_record.energy_kev:
            continue
        raw_value = float(value_getter(rec))
        if not np.isfinite(raw_value) or raw_value <= 0.0:
            continue
        tilt_key = float(max(abs(rec.tilt_deg), 0.0))
        grouped.setdefault(tilt_key, []).append(raw_value / baseline_value)
    tilt_deg_knots = sorted(grouped)
    tilt_scale_knots = [
        float(np.clip(_median(grouped[tilt_deg], 1.0), 1e-4, 1e4))
        for tilt_deg in tilt_deg_knots
    ]
    return tilt_deg_knots, tilt_scale_knots


def _predict_eta_response(
    *,
    material_z: float,
    energy_kev: float,
    tilt_deg: float,
    yield_scale: float,
    reference_energy_kev: float,
    energy_exponent: float,
    obliquity_exponent_scale: float,
    tilt_deg_knots: list[float] | tuple[float, ...] = (),
    tilt_scale_knots: list[float] | tuple[float, ...] = (),
) -> float:
    cos_term = max(np.cos(np.radians(tilt_deg)), 1e-6)
    default_n = 9.0 / np.sqrt(max(material_z, 1e-6))
    fitted_n = default_n * obliquity_exponent_scale
    return float(
        eta_z(material_z)
        * yield_scale
        * (max(energy_kev, 1e-6) / max(reference_energy_kev, 1e-6)) ** energy_exponent
        * interpolate_tilt_scale(tilt_deg, tilt_deg_knots, tilt_scale_knots)
        / (cos_term**fitted_n)
    )


def _fit_yield_tilt_response_curve(
    records: list[CasinoBenchmarkRecord],
    baseline_record: CasinoBenchmarkRecord,
    *,
    material_z: float,
    yield_scale: float,
    obliquity_exponent_scale: float,
    reference_energy_kev: float,
    energy_exponent: float = 0.0,
) -> tuple[list[float], list[float]]:
    grouped: dict[float, list[float]] = {0.0: [1.0]}
    for rec in records:
        if rec.geometry != baseline_record.geometry or rec.energy_kev != baseline_record.energy_kev:
            continue
        predicted = _predict_eta_response(
            material_z=material_z,
            energy_kev=rec.energy_kev,
            tilt_deg=rec.tilt_deg,
            yield_scale=yield_scale,
            reference_energy_kev=reference_energy_kev,
            energy_exponent=energy_exponent,
            obliquity_exponent_scale=obliquity_exponent_scale,
        )
        if not np.isfinite(predicted) or predicted <= 0.0:
            continue
        tilt_key = float(max(abs(rec.tilt_deg), 0.0))
        grouped.setdefault(tilt_key, []).append(float(rec.eta_bse) / predicted)
    tilt_deg_knots = sorted(grouped)
    tilt_scale_knots = [
        float(np.clip(_median(grouped[tilt_deg], 1.0), 1e-4, 1e4))
        for tilt_deg in tilt_deg_knots
    ]
    return tilt_deg_knots, tilt_scale_knots


def _geometry_topo_gain(
    baseline: CasinoBenchmarkRecord,
    rec: CasinoBenchmarkRecord,
) -> float:
    inv_blur = float(max(baseline.r50_nm, 1e-9) / max(rec.r50_nm, 1e-9))
    yield_ratio = float(max(rec.eta_bse, 1e-9) / max(baseline.eta_bse, 1e-9))
    return float(np.clip(inv_blur * yield_ratio, 0.5, 3.0))


def _geometry_openness_scale(
    baseline: CasinoBenchmarkRecord,
    rec: CasinoBenchmarkRecord,
) -> float:
    blur_ratio = float(max(rec.r50_nm, 1e-9) / max(baseline.r50_nm, 1e-9))
    depth_ratio = float(max(rec.z50_nm, 1e-9) / max(baseline.z50_nm, 1e-9))
    return float(np.clip(0.5 * blur_ratio + 0.5 * depth_ratio, 0.5, 4.0))


def build_geometry_profile_library(
    campaign: CasinoBenchmarkCampaign,
    baseline: CasinoBenchmarkRecord,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {
        baseline.geometry: {
            "geometry": baseline.geometry,
            "yield_scale": 1.0,
            "blur_scale": 1.0,
            "depth_scale": 1.0,
            "topo_gain": 1.0,
            "openness_scale": 1.0,
            "reference_eta_bse": float(baseline.eta_bse),
            "reference_r50_nm": float(baseline.r50_nm),
            "reference_z50_nm": float(baseline.z50_nm),
            "source_sim_id": baseline.sim_id,
        }
    }
    for rec in campaign.records:
        if not _is_usable(rec):
            continue
        if rec.geometry in profiles:
            continue
        profiles[rec.geometry] = {
            "geometry": rec.geometry,
            "yield_scale": float(np.clip(rec.eta_bse / max(baseline.eta_bse, 1e-9), 0.25, 4.0)),
            "blur_scale": float(np.clip(rec.r50_nm / max(baseline.r50_nm, 1e-9), 0.1, 20.0)),
            "depth_scale": float(np.clip(rec.z50_nm / max(baseline.z50_nm, 1e-9), 0.1, 20.0)),
            "topo_gain": _geometry_topo_gain(baseline, rec),
            "openness_scale": _geometry_openness_scale(baseline, rec),
            "reference_eta_bse": float(rec.eta_bse),
            "reference_r50_nm": float(rec.r50_nm),
            "reference_z50_nm": float(rec.z50_nm),
            "source_sim_id": rec.sim_id,
        }
    return profiles


def fit_casino_benchmark_calibration(
    *,
    benchmark_summary_path: str | Path,
    output_dir: str | Path,
    config_dir: str | Path = "config",
    pixel_size_mm: float = 0.05,
) -> dict[str, Any]:
    campaign = load_casino_benchmark_campaign(benchmark_summary_path)
    cfg, material = _pick_material_preset(campaign, config_dir)
    usable = [r for r in campaign.records if _is_usable(r)]
    flatbulk = [r for r in usable if r.geometry == "flatbulk"]
    if not flatbulk:
        raise ValueError("Benchmark campaign must include at least one usable 'flatbulk' record.")

    baseline = min(flatbulk, key=lambda r: (abs(r.tilt_deg), r.energy_kev, r.sim_id))
    eta0_default = float(eta_z(material.z))
    yield_scale = float(np.clip(baseline.eta_bse / max(eta0_default, 1e-9), 0.1, 10.0))
    obliquity_scale = _fit_obliquity_scale(baseline.eta_bse, baseline, flatbulk, material.z)
    yield_tilt_deg_knots, yield_tilt_scale_knots = _fit_yield_tilt_response_curve(
        flatbulk,
        baseline,
        material_z=material.z,
        yield_scale=yield_scale,
        obliquity_exponent_scale=obliquity_scale,
        reference_energy_kev=float(baseline.energy_kev),
    )

    default_sigma_px = _default_blur_sigma_px(config_dir, baseline.energy_kev, pixel_size_mm)
    target_sigma_px = (baseline.r50_nm / 1.1774100225154747) / max(pixel_size_mm * 1_000_000.0, 1e-6)
    blur_scale = float(np.clip(target_sigma_px / max(default_sigma_px, 1e-9), 1e-4, 10.0))
    blur_tilt_deg_knots, blur_tilt_scale_knots = _fit_tilt_response_curve(
        flatbulk,
        baseline,
        lambda rec: rec.r50_nm,
    )

    range_um = kanaya_okayama_range(
        a=material.a,
        e0_kev=baseline.energy_kev,
        z=material.z,
        rho=material.rho,
    )
    range_scale = float(
        np.clip((baseline.z50_nm / max(range_um * 1_000.0, 1e-9)), 1e-4, 10.0)
    )
    depth_tilt_deg_knots, depth_tilt_scale_knots = _fit_tilt_response_curve(
        flatbulk,
        baseline,
        lambda rec: rec.z50_nm,
    )

    pore_visibility_scale = _fit_defect_scale(usable, "spherical_pore_single", default=1.0)
    groove_topography_scale = _fit_defect_scale(usable, "groove_surface", default=1.0)
    cluster_visibility_scale = _fit_defect_scale(usable, "pore_cluster_multi", default=1.0)
    geometry_profiles = build_geometry_profile_library(campaign, baseline)

    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = run_dir / "renderer_calibration.json"
    surrogate = {
        "source_campaign": campaign.campaign_id,
        "yield_model": {
            "scale": yield_scale,
            "reference_energy_kev": float(baseline.energy_kev),
            "energy_exponent": 0.0,
            "obliquity_exponent_scale": obliquity_scale,
            "tilt_deg_knots": yield_tilt_deg_knots,
            "tilt_scale_knots": yield_tilt_scale_knots,
        },
        "blur_model": {
            "scale": blur_scale,
            "reference_r50_nm": float(baseline.r50_nm),
            "reference_r90_nm": float(baseline.r90_nm),
            "tilt_deg_knots": blur_tilt_deg_knots,
            "tilt_scale_knots": blur_tilt_scale_knots,
        },
        "depth_model": {
            "range_scale": range_scale,
            "reference_z50_nm": float(baseline.z50_nm),
            "reference_z90_nm": float(baseline.z90_nm),
            "tilt_deg_knots": depth_tilt_deg_knots,
            "tilt_scale_knots": depth_tilt_scale_knots,
        },
        "defect_model": {
            "pore_visibility_scale": pore_visibility_scale,
            "groove_topography_scale": groove_topography_scale,
            "cluster_visibility_scale": cluster_visibility_scale,
        },
        "geometry_profiles": geometry_profiles,
    }
    metadata = {
        "source": "casino_benchmark_fit",
        "campaign_id": campaign.campaign_id,
        "source_archive": campaign.source_archive,
        "source_campaign_note": campaign.source_campaign_note,
        "material": material.name,
        "num_records_total": len(campaign.records),
        "num_records_usable": len(usable),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_summary": {
            "baseline_sim_id": baseline.sim_id,
            "baseline_eta_bse": baseline.eta_bse,
            "yield_scale": yield_scale,
            "obliquity_exponent_scale": obliquity_scale,
            "yield_tilt_knot_count": len(yield_tilt_deg_knots),
            "blur_scale": blur_scale,
            "range_scale": range_scale,
            "blur_tilt_knot_count": len(blur_tilt_deg_knots),
            "depth_tilt_knot_count": len(depth_tilt_deg_knots),
            "geometry_profile_count": len(geometry_profiles),
        },
    }
    calibration_file = save_renderer_calibration(
        calibration_path,
        compo_scale=1.0,
        compo_bias=0.0,
        topo_scale=1.0,
        topo_bias=0.0,
        surrogate=surrogate,
        metadata=metadata,
    )

    manifest = {
        "schema_version": 1,
        "campaign_id": campaign.campaign_id,
        "benchmark_summary_path": str(Path(benchmark_summary_path).resolve()),
        "calibration_file": calibration_file,
        "geometry_profile_file": str((run_dir / "geometry_profiles.json").resolve()),
        "records_used": [r.sim_id for r in usable],
        "source_archive": campaign.source_archive,
        "source_campaign_note": campaign.source_campaign_note,
        "generated_at_utc": metadata["generated_at_utc"],
    }
    geometry_profile_path = run_dir / "geometry_profiles.json"
    geometry_profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": campaign.campaign_id,
                "baseline_geometry": baseline.geometry,
                "baseline_sim_id": baseline.sim_id,
                "profiles": geometry_profiles,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "benchmark_fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = evaluate_casino_benchmark_fit(
        benchmark_summary_path=benchmark_summary_path,
        calibration_path=calibration_file,
        config_dir=cfg.config_dir,
        output_dir=run_dir,
    )

    return {
        "status": "ok",
        "campaign_id": campaign.campaign_id,
        "calibration_file": calibration_file,
        "geometry_profile_file": str(geometry_profile_path),
        "manifest_file": str(manifest_path),
        "fit_report": report,
        "surrogate": surrogate,
    }


def evaluate_casino_benchmark_fit(
    *,
    benchmark_summary_path: str | Path,
    calibration_path: str | Path,
    config_dir: str | Path = "config",
    output_dir: str | Path | None = None,
    pixel_size_mm: float = 0.05,
) -> dict[str, Any]:
    from ..renderer.calibration import load_renderer_calibration

    campaign = load_casino_benchmark_campaign(benchmark_summary_path)
    _, material = _pick_material_preset(campaign, config_dir)
    calibration = load_renderer_calibration(calibration_path)

    records_out: list[dict[str, Any]] = []
    eta_errors: list[float] = []
    blur_errors: list[float] = []
    depth_errors: list[float] = []

    for rec in campaign.records:
        if not _is_usable(rec):
            continue
        eta_pred = _predict_eta_response(
            material_z=material.z,
            energy_kev=rec.energy_kev,
            tilt_deg=rec.tilt_deg,
            yield_scale=calibration.surrogate.yield_model.scale,
            reference_energy_kev=calibration.surrogate.yield_model.reference_energy_kev,
            energy_exponent=calibration.surrogate.yield_model.energy_exponent,
            obliquity_exponent_scale=calibration.surrogate.yield_model.obliquity_exponent_scale,
            tilt_deg_knots=calibration.surrogate.yield_model.tilt_deg_knots,
            tilt_scale_knots=calibration.surrogate.yield_model.tilt_scale_knots,
        )

        default_sigma_px = _default_blur_sigma_px(config_dir, rec.energy_kev, pixel_size_mm)
        pred_sigma_px = (
            default_sigma_px
            * calibration.surrogate.blur_model.scale
            * interpolate_tilt_scale(
                rec.tilt_deg,
                calibration.surrogate.blur_model.tilt_deg_knots,
                calibration.surrogate.blur_model.tilt_scale_knots,
            )
        )
        pred_r50_nm = float(pred_sigma_px * 1.1774100225154747 * pixel_size_mm * 1_000_000.0)

        range_um = kanaya_okayama_range(
            a=material.a,
            e0_kev=rec.energy_kev,
            z=material.z,
            rho=material.rho,
        )
        pred_z50_nm = float(
            range_um
            * 1_000.0
            * calibration.surrogate.depth_model.range_scale
            * interpolate_tilt_scale(
                rec.tilt_deg,
                calibration.surrogate.depth_model.tilt_deg_knots,
                calibration.surrogate.depth_model.tilt_scale_knots,
            )
        )

        eta_rel = abs(eta_pred - rec.eta_bse) / max(rec.eta_bse, 1e-9)
        blur_rel = abs(pred_r50_nm - rec.r50_nm) / max(rec.r50_nm, 1e-9)
        depth_rel = abs(pred_z50_nm - rec.z50_nm) / max(rec.z50_nm, 1e-9)
        eta_errors.append(eta_rel)
        blur_errors.append(blur_rel)
        depth_errors.append(depth_rel)

        records_out.append(
            {
                "sim_id": rec.sim_id,
                "geometry": rec.geometry,
                "tilt_deg": rec.tilt_deg,
                "eta_bse": rec.eta_bse,
                "eta_pred": eta_pred,
                "eta_rel_error": eta_rel,
                "r50_nm": rec.r50_nm,
                "r50_pred_nm": pred_r50_nm,
                "r50_rel_error": blur_rel,
                "z50_nm": rec.z50_nm,
                "z50_pred_nm": pred_z50_nm,
                "z50_rel_error": depth_rel,
                "quality_flag": rec.quality_flag,
            }
        )

    summary = {
        "mean_eta_rel_error": float(np.mean(eta_errors)) if eta_errors else None,
        "mean_r50_rel_error": float(np.mean(blur_errors)) if blur_errors else None,
        "mean_z50_rel_error": float(np.mean(depth_errors)) if depth_errors else None,
        "max_eta_rel_error": float(np.max(eta_errors)) if eta_errors else None,
        "max_r50_rel_error": float(np.max(blur_errors)) if blur_errors else None,
        "max_z50_rel_error": float(np.max(depth_errors)) if depth_errors else None,
    }
    out = {
        "status": "ok",
        "campaign_id": campaign.campaign_id,
        "calibration_path": str(Path(calibration_path).resolve()),
        "summary": summary,
        "records": records_out,
    }
    if output_dir is not None:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "benchmark_fit_report.json"
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        out["report_path"] = str(p)
    return out


def _median_int(values: list[int], default: int | None = None) -> int:
    if not values:
        if default is None:
            raise ValueError("Expected at least one integer value to reduce.")
        return int(default)
    return int(round(float(np.median(np.asarray(values, dtype=np.float64)))))


def _mean(values: list[float], default: float) -> float:
    if not values:
        return float(default)
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _mean_int(values: list[int], default: int | None = None) -> int:
    if not values:
        if default is None:
            raise ValueError("Expected at least one integer value to reduce.")
        return int(default)
    return int(round(float(np.mean(np.asarray(values, dtype=np.float64)))))


def _reduce_float(values: list[float], reducer: str, default: float) -> float:
    if reducer == "median":
        return _median(values, default)
    if reducer == "mean":
        return _mean(values, default)
    raise ValueError(f"Unsupported float reducer '{reducer}'.")


def _reduce_int(values: list[int], reducer: str, default: int | None = None) -> int:
    if reducer == "median":
        return _median_int(values, default)
    if reducer == "mean":
        return _mean_int(values, default)
    raise ValueError(f"Unsupported int reducer '{reducer}'.")


def _resolve_metadata_value(
    explicit: Any,
    member_values: list[Any],
    default: Any = None,
) -> Any:
    if explicit is not None:
        return explicit
    uniq = []
    for value in member_values:
        if value is not None and value not in uniq:
            uniq.append(value)
    if len(uniq) == 1:
        return uniq[0]
    if uniq:
        return uniq[0]
    return default


def _make_auto_record(run: CasinoAssemblyRunSpec) -> CasinoAssemblyRecordSpec:
    return CasinoAssemblyRecordSpec(
        sim_id=run.run_id,
        from_runs=(run.run_id,),
        geometry=run.geometry,
        tilt_deg=run.tilt_deg,
        material=run.material,
        energy_kev=run.energy_kev,
        beam_diameter_nm=run.beam_diameter_nm,
        electrons=run.electrons,
        notes=run.notes,
        quality_flag=run.quality_flag,
        defect_signal_proxy=run.defect_signal_proxy,
        source_note=run.source_note,
        reducer="median",
    )


def assemble_casino_benchmark_campaign(
    *,
    export_root: str | Path,
    assembly_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    cfg = load_casino_assembly_config(assembly_config_path)
    parsed_runs: dict[str, ParsedCasinoRunMetrics] = {}
    run_specs_by_id = {run.run_id: run for run in cfg.runs}

    for run in cfg.runs:
        artifact_root = _normalize_artifact_root(export_root, run.artifact_root)
        parsed_runs[run.run_id] = parse_casino_export_run(
            run_id=run.run_id,
            artifact_root=artifact_root,
            summary_dat=run.summary_dat,
            bs_angle_dat=run.bs_angle_dat,
            bs_energy_dat=run.bs_energy_dat,
            surface_radius_dat=run.surface_radius_dat,
            zmax_bs_dat=run.zmax_bs_dat,
        )

    records: list[dict[str, Any]] = []
    extraction_report_runs: list[dict[str, Any]] = []
    for run_id, parsed in parsed_runs.items():
        extraction_report_runs.append(
            {
                "run_id": run_id,
                "artifact_root": parsed.artifact_root,
                "eta_bse": parsed.eta_bse,
                "mean_bse_energy_kev": parsed.mean_bse_energy_kev,
                "peak_bse_energy_kev": parsed.peak_bse_energy_kev,
                "mean_bse_angle_deg": parsed.mean_bse_angle_deg,
                "peak_bse_angle_deg": parsed.peak_bse_angle_deg,
                "r50_nm": parsed.r50_nm,
                "r90_nm": parsed.r90_nm,
                "z50_nm": parsed.z50_nm,
                "z90_nm": parsed.z90_nm,
                "depth_source": parsed.depth_source,
                "summary_dat_path": parsed.summary_dat_path,
                "bs_angle_dat_path": parsed.bs_angle_dat_path,
                "bs_energy_dat_path": parsed.bs_energy_dat_path,
                "surface_radius_dat_path": parsed.surface_radius_dat_path,
                "zmax_bs_dat_path": parsed.zmax_bs_dat_path,
            }
        )

    record_specs = list(cfg.records)
    if cfg.auto_single_run_records:
        assigned_run_ids = {run_id for rec in cfg.records for run_id in rec.from_runs}
        for run in cfg.runs:
            if run.run_id not in assigned_run_ids:
                record_specs.append(_make_auto_record(run))

    for rec in record_specs:
        members = [parsed_runs[run_id] for run_id in rec.from_runs]
        run_specs = [run_specs_by_id[run_id] for run_id in rec.from_runs]
        reducer = rec.reducer
        record = {
            "sim_id": rec.sim_id,
            "material": _resolve_metadata_value(
                rec.material,
                [run.material or parsed.summary_metadata.get("material") for run, parsed in zip(run_specs, members)],
                cfg.defaults.material,
            ),
            "geometry": rec.geometry,
            "energy_kev": _resolve_metadata_value(
                rec.energy_kev,
                [run.energy_kev or parsed.summary_metadata.get("energy_kev") for run, parsed in zip(run_specs, members)],
                cfg.defaults.energy_kev,
            ),
            "tilt_deg": rec.tilt_deg,
            "beam_diameter_nm": _resolve_metadata_value(
                rec.beam_diameter_nm,
                [run.beam_diameter_nm for run in run_specs],
                cfg.defaults.beam_diameter_nm,
            ),
            "electrons": _resolve_metadata_value(
                rec.electrons,
                [run.electrons or parsed.summary_metadata.get("electrons") for run, parsed in zip(run_specs, members)],
                cfg.defaults.electrons,
            ),
            "eta_bse": _reduce_float([m.eta_bse for m in members], reducer, members[0].eta_bse),
            "mean_bse_energy_kev": _reduce_float(
                [m.mean_bse_energy_kev for m in members], reducer, members[0].mean_bse_energy_kev
            ),
            "peak_bse_angle_deg": _reduce_float(
                [m.peak_bse_angle_deg for m in members], reducer, members[0].peak_bse_angle_deg
            ),
            "r50_nm": _reduce_float([m.r50_nm for m in members], reducer, members[0].r50_nm),
            "r90_nm": _reduce_float([m.r90_nm for m in members], reducer, members[0].r90_nm),
            "z50_nm": _reduce_float([m.z50_nm for m in members], reducer, members[0].z50_nm),
            "z90_nm": _reduce_float([m.z90_nm for m in members], reducer, members[0].z90_nm),
            "notes": rec.notes,
            "quality_flag": rec.quality_flag,
            "aggregation_mode": reducer,
            "num_source_runs": len(members),
        }
        if rec.defect_signal_proxy is not None:
            record["defect_signal_proxy"] = rec.defect_signal_proxy
        else:
            proxies = [run.defect_signal_proxy for run in run_specs if run.defect_signal_proxy is not None]
            if proxies:
                record["defect_signal_proxy"] = _reduce_float(
                    [float(v) for v in proxies], reducer, float(proxies[0])
                )
        if rec.source_note is not None:
            record["source_note"] = rec.source_note
        artifact_root_val = members[0].artifact_root if len(members) == 1 else None
        if artifact_root_val is not None:
            record["artifact_root"] = artifact_root_val
        records.append(record)

    campaign = {
        "schema_version": 1,
        "campaign_id": cfg.campaign_id,
        "source_archive": cfg.source_archive,
        "source_campaign_note": cfg.source_campaign_note,
        "description": cfg.description,
        "records": records,
    }

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(campaign, indent=2), encoding="utf-8")

    report = {
        "status": "ok",
        "campaign_id": cfg.campaign_id,
        "assembly_config_path": str(Path(assembly_config_path).resolve()),
        "export_root": str(Path(export_root).resolve()),
        "output_path": str(out_path),
        "parsed_runs": extraction_report_runs,
        "assembled_records": records,
    }
    report_path = out_path.with_name(f"{out_path.stem}_assembly_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def benchmark_campaign_to_dict(campaign: CasinoBenchmarkCampaign) -> dict[str, Any]:
    return {
        "schema_version": campaign.schema_version,
        "campaign_id": campaign.campaign_id,
        "source_archive": campaign.source_archive,
        "source_campaign_note": campaign.source_campaign_note,
        "description": campaign.description,
        "records": [asdict(record) for record in campaign.records],
    }
