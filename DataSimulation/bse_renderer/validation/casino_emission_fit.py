from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


_RUN_RE = re.compile(r"^(?P<tilt>-?\d+(?:\.\d+)?)deg-r(?P<repeat>\d+)$", re.IGNORECASE)
_TRAJECTORY_RE = re.compile(r'^"Trajectory"\s+(\d+)$')
_SCAN_POINT_DEFINITION_RE = re.compile(r'^"Scan Point"\s+(?P<scan_point>\d+)$')
_SCAN_POINT_RESULTS_RE = re.compile(r'^"Scan Point\s+(?P<scan_point>\d+)\s+Results\s*:\s*"$')
_TYPE_LINE_RE = re.compile(r'^"Type\s*:\s*"$')
_LABEL_RE = re.compile(r'^"(?P<label>[^"]+)"\s+(?P<value>[^\r\n]+)$')
_POLAR_BANDS_DEG = ((15.0, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, 75.0))
_WEAK_FRAME_SIN_ALPHA = 1e-4


@dataclass(frozen=True)
class SummaryRow:
    x_nm: float
    y_nm: float
    z_nm: float
    rx: float
    ry: float
    rz: float
    segment_length_nm: float
    energy_kev: float
    collision_type: str


@dataclass(frozen=True)
class ExitSample:
    run_id: str
    tilt_deg: float
    nominal_tilt_deg: float
    repeat_id: int
    trajectory_id: int
    x_nm: float
    y_nm: float
    z_nm: float
    rx: float
    ry: float
    rz: float
    exit_energy_kev: float
    phi_rad: float
    varphi_rad: float
    cos_phi: float
    scan_point_id: int = 0
    scan_point_block_index: int = 1
    scan_point_run_id: str | None = None


@dataclass(frozen=True)
class RunSummaryFitData:
    run_id: str
    tilt_deg: float
    nominal_tilt_deg: float
    repeat_id: int
    summary_path: str
    summary_sha256: str
    bs_angle_path: str | None
    bs_angle_sha256: str | None
    bs_energy_path: str | None
    bs_energy_sha256: str | None
    scan_point_count: int
    trajectory_count: int
    backscattered_count: int
    transmitted_count: int
    simple_count: int
    exit_sample_count: int
    reported_backscatter_count: int | None
    reported_transmitted_count: int | None
    reported_eta_bse: float | None
    reported_backscatter_count_delta: int | None
    reported_transmitted_count_delta: int | None
    reported_counts_match: bool | None
    reported_counts_within_export_tolerance: bool | None
    summary_vs_bsangle_cross_entropy: float | None
    summary_vs_bsangle_l1: float | None
    rr_polar_vs_bsangle_cross_entropy: float | None
    summary_vs_bsenergy_cross_entropy: float | None
    summary_vs_bsenergy_l1: float | None
    summary_energy_mean_kev: float | None
    export_energy_mean_kev: float | None
    entry_normal_x: float
    entry_normal_y: float
    entry_normal_z: float
    entry_tangent_1_x: float
    entry_tangent_1_y: float
    entry_tangent_1_z: float
    entry_tangent_2_x: float
    entry_tangent_2_y: float
    entry_tangent_2_z: float
    frame_normal_norm_error: float
    frame_tangent_1_norm_error: float
    frame_tangent_2_norm_error: float
    frame_normal_dot_tangent_1_abs: float
    frame_normal_dot_tangent_2_abs: float
    frame_tangent_dot_abs: float
    frame_handedness: float
    projected_beam_tangent_norm: float
    weak_azimuth_orientation: bool


@dataclass(frozen=True)
class ScanPointSummaryFitData:
    source_run_id: str
    scan_point_run_id: str
    scan_point_id: int
    scan_point_block_index: int
    nominal_tilt_deg: float
    measured_tilt_deg: float
    repeat_id: int
    scan_x: float | None
    scan_y: float | None
    initial_energy_kev: float | None
    summary_path: str
    summary_sha256: str
    result_line_number: int | None
    trajectory_count: int
    backscattered_count: int
    transmitted_count: int
    simple_count: int
    exit_sample_count: int
    reported_backscatter_count: int | None
    reported_transmitted_count: int | None
    reported_eta_bse: float | None
    backscatter_count_delta: int | None
    transmitted_count_delta: int | None
    counts_match: bool | None
    counts_within_export_tolerance: bool | None
    count_status: str


@dataclass(frozen=True)
class BFoldResult:
    holdout_repeat_id: int
    fitted_b: float
    train_objective: float
    test_negative_log_likelihood: float
    train_run_count: int
    test_run_count: int
    train_sample_count: int
    test_sample_count: int


@dataclass(frozen=True)
class BAlphaFoldResult:
    holdout_repeat_id: int
    fitted_b0: float
    fitted_b1: float
    train_objective: float
    test_negative_log_likelihood: float
    train_run_count: int
    test_run_count: int
    train_sample_count: int
    test_sample_count: int


def _parse_run_id(run_id: str) -> tuple[float, int]:
    match = _RUN_RE.match(run_id.strip())
    if not match:
        raise ValueError(f"Run id '{run_id}' does not match '<tilt>deg-r<repeat>'.")
    return float(match.group("tilt")), int(match.group("repeat"))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_entry_frame(entry_points_nm: list[tuple[float, float, float]], entry_dirs: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(entry_points_nm) < 3:
        raise ValueError("Need at least three entry points to infer the entrance plane.")
    points = np.asarray(entry_points_nm, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)

    beam_dir = np.mean(np.asarray(entry_dirs, dtype=np.float64), axis=0)
    beam_dir = beam_dir / max(float(np.linalg.norm(beam_dir)), 1e-12)
    if float(np.dot(normal, -beam_dir)) < 0.0:
        normal = -normal

    tangent_1 = beam_dir - float(np.dot(beam_dir, normal)) * normal
    tangent_1_norm = float(np.linalg.norm(tangent_1))
    if tangent_1_norm <= 1e-12:
        tangent_1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_1_norm = 1.0
    tangent_1 = tangent_1 / tangent_1_norm
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2_norm = float(np.linalg.norm(tangent_2))
    if tangent_2_norm <= 1e-12:
        tangent_2 = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        tangent_2_norm = 1.0
    tangent_2 = tangent_2 / tangent_2_norm
    measured_tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(normal, -beam_dir), -1.0, 1.0))))
    return normal, tangent_1, tangent_2, measured_tilt_deg


def _direction_to_angles(direction: np.ndarray, normal: np.ndarray, tangent_1: np.ndarray, tangent_2: np.ndarray) -> tuple[float, float, float]:
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("Exit direction has zero norm.")
    unit = vec / norm
    cos_phi = float(np.clip(np.dot(unit, normal), -1.0, 1.0))
    phi_rad = float(math.acos(cos_phi))
    varphi_rad = float(math.atan2(np.dot(unit, tangent_2), np.dot(unit, tangent_1)))
    return phi_rad, varphi_rad, cos_phi


def _parse_collision_row(line: str) -> SummaryRow | None:
    raw = line.strip()
    if not raw or raw.startswith('"'):
        return None
    parts = [part.strip() for part in raw.split("\t") if part.strip()]
    if len(parts) < 9:
        return None
    try:
        return SummaryRow(
            x_nm=float(parts[0]),
            y_nm=float(parts[1]),
            z_nm=float(parts[2]),
            rx=float(parts[3]),
            ry=float(parts[4]),
            rz=float(parts[5]),
            segment_length_nm=float(parts[6]),
            energy_kev=float(parts[7]),
            collision_type=parts[8],
        )
    except ValueError:
        return None


def _extract_label_value(line: str) -> tuple[str, str] | None:
    match = _LABEL_RE.match(line.strip())
    if not match:
        return None
    label = match.group("label").strip()
    value = match.group("value").strip().strip('"')
    return label, value


def _parse_distribution_file(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[tuple[float, float]] = []
    for line in lines[2:]:
        raw = line.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split("\t")]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        rows.append((x, y))
    if not rows:
        raise ValueError(f"Distribution file '{path}' did not contain numeric rows.")
    return np.asarray(rows, dtype=np.float64)


def _normalize_histogram(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("Histogram weights sum to zero.")
    return arr / total


def _cross_entropy(p_obs: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> float:
    obs = _normalize_histogram(p_obs)
    pred = _normalize_histogram(np.clip(p_pred, eps, None))
    return float(-np.sum(obs * np.log(pred)))


def _l1_distance(p_obs: np.ndarray, p_pred: np.ndarray) -> float:
    obs = _normalize_histogram(p_obs)
    pred = _normalize_histogram(p_pred)
    return float(np.abs(obs - pred).sum())


def _angle_bin_edges_from_centers(centers_deg: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers_deg, dtype=np.float64)
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError("Angle centers must be a non-empty 1D array.")
    if len(centers) == 1:
        width = 1.0
        return np.asarray([centers[0] - width / 2.0, centers[0] + width / 2.0], dtype=np.float64)
    deltas = np.diff(centers)
    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + deltas / 2.0
    edges[0] = centers[0] - deltas[0] / 2.0
    edges[-1] = centers[-1] + deltas[-1] / 2.0
    return edges


def _empirical_histogram(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bin_edges)
    return hist.astype(np.float64)


def _rr_polar_histogram(centers_deg: np.ndarray) -> np.ndarray:
    edges_deg = _angle_bin_edges_from_centers(centers_deg)
    edges_rad = np.radians(np.clip(edges_deg, 0.0, 90.0))
    masses = []
    for lo, hi in zip(edges_rad[:-1], edges_rad[1:]):
        mass = 0.5 * (math.sin(hi) ** 2 - math.sin(lo) ** 2) * 2.0
        masses.append(max(mass, 0.0))
    return _normalize_histogram(np.asarray(masses, dtype=np.float64))


def _histogram_to_mean(xy: np.ndarray) -> float:
    x = xy[:, 0]
    w = _normalize_histogram(xy[:, 1])
    return float(np.sum(x * w))


def _energy_histogram_from_samples(samples: list[ExitSample], energy_xy: np.ndarray) -> np.ndarray:
    centers = energy_xy[:, 0]
    edges = _angle_bin_edges_from_centers(centers)
    energies = np.asarray([sample.exit_energy_kev for sample in samples], dtype=np.float64)
    return _empirical_histogram(energies, edges)


def _angle_histogram_from_samples(samples: list[ExitSample], angle_xy: np.ndarray) -> np.ndarray:
    centers = angle_xy[:, 0]
    edges = _angle_bin_edges_from_centers(centers)
    angles_deg = np.degrees(np.asarray([sample.phi_rad for sample in samples], dtype=np.float64))
    return _empirical_histogram(angles_deg, edges)


def _iter_summary_lines(path: str | Path):
    """Read text or compressed CASINO exports without materializing the file."""
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def _parse_summary_file(path: str | Path, run_id: str, nominal_tilt_deg: float, repeat_id: int) -> tuple[list[ExitSample], dict[str, Any]]:
    # Only entrance and final boundary rows enter the angular analysis. Retain
    # at most two rows per history even for full collision-by-collision exports.
    lines = _iter_summary_lines(path)
    current_trajectory_id: int | None = None
    current_type: str | None = None
    current_rows: list[SummaryRow] = []
    awaiting_type = False

    trajectory_count = 0
    entry_points_nm: list[tuple[float, float, float]] = []
    entry_dirs: list[tuple[float, float, float]] = []
    trajectories: list[tuple[int, int, int, str | None, list[SummaryRow]]] = []
    scan_point_blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None
    scan_point_definitions: dict[int, dict[str, float | None]] = {}
    current_scan_point_definition_id: int | None = None

    def ensure_block() -> dict[str, Any]:
        nonlocal current_block
        if current_block is None:
            current_block = {
                "scan_point_id": 0,
                "scan_point_block_index": len(scan_point_blocks) + 1,
                "scan_point_run_id": f"{run_id}#sp0",
                "result_line_number": None,
                "scan_x": None,
                "scan_y": None,
                "initial_energy_kev": None,
                "trajectory_count": 0,
                "backscattered_count": 0,
                "transmitted_count": 0,
                "simple_count": 0,
                "exit_sample_count": 0,
                "reported_backscatter_count": None,
                "reported_transmitted_count": None,
                "reported_eta_bse": None,
            }
            scan_point_blocks.append(current_block)
        return current_block

    def finalize_current() -> None:
        nonlocal current_trajectory_id, current_type, current_rows
        if current_trajectory_id is None:
            return
        block = ensure_block()
        trajectories.append(
            (
                int(block["scan_point_block_index"]),
                int(block["scan_point_id"]),
                current_trajectory_id,
                current_type,
                list(current_rows),
            )
        )
        if current_rows:
            first = current_rows[0]
            entry_points_nm.append((first.x_nm, first.y_nm, first.z_nm))
            entry_dirs.append((first.rx, first.ry, first.rz))
        current_trajectory_id = None
        current_type = None
        current_rows = []

    for line_number, line in enumerate(lines, start=1):
        scan_point_definition_match = _SCAN_POINT_DEFINITION_RE.match(line.strip())
        if scan_point_definition_match:
            current_scan_point_definition_id = int(scan_point_definition_match.group("scan_point"))
            scan_point_definitions.setdefault(
                current_scan_point_definition_id,
                {"scan_x": None, "scan_y": None, "initial_energy_kev": None},
            )
            continue
        scan_point_match = _SCAN_POINT_RESULTS_RE.match(line.strip())
        if scan_point_match:
            finalize_current()
            scan_point_id = int(scan_point_match.group("scan_point"))
            scan_definition = scan_point_definitions.get(
                scan_point_id,
                {"scan_x": None, "scan_y": None, "initial_energy_kev": None},
            )
            current_block = {
                "scan_point_id": scan_point_id,
                "scan_point_block_index": len(scan_point_blocks) + 1,
                "scan_point_run_id": f"{run_id}#sp{scan_point_id}",
                "result_line_number": line_number,
                **scan_definition,
                "trajectory_count": 0,
                "backscattered_count": 0,
                "transmitted_count": 0,
                "simple_count": 0,
                "exit_sample_count": 0,
                "reported_backscatter_count": None,
                "reported_transmitted_count": None,
                "reported_eta_bse": None,
            }
            scan_point_blocks.append(current_block)
            awaiting_type = False
            continue
        trajectory_match = _TRAJECTORY_RE.match(line.strip())
        if trajectory_match:
            finalize_current()
            block = ensure_block()
            current_trajectory_id = int(trajectory_match.group(1))
            current_type = None
            current_rows = []
            awaiting_type = False
            trajectory_count += 1
            block["trajectory_count"] += 1
            continue
        if _TYPE_LINE_RE.match(line.strip()):
            awaiting_type = True
            continue
        if awaiting_type:
            candidate = line.strip().strip('"')
            if candidate in {"Backscattered", "Transmitted", "Simple"}:
                current_type = candidate
                awaiting_type = False
                continue
            if candidate:
                awaiting_type = False
        label_value = _extract_label_value(line)
        if label_value is not None:
            label, value = label_value
            if current_scan_point_definition_id is not None and label in {"X", "Y", "Initial Energy"}:
                try:
                    numeric_value = float(value)
                except ValueError:
                    numeric_value = None
                definition = scan_point_definitions[current_scan_point_definition_id]
                if label == "X":
                    definition["scan_x"] = numeric_value
                elif label == "Y":
                    definition["scan_y"] = numeric_value
                else:
                    definition["initial_energy_kev"] = numeric_value
            if label == "BackScattering Coefficient":
                block = ensure_block()
                try:
                    block["reported_eta_bse"] = float(value)
                except ValueError:
                    block["reported_eta_bse"] = None
            elif label == "nb backscattered":
                block = ensure_block()
                try:
                    block["reported_backscatter_count"] = int(float(value))
                except ValueError:
                    block["reported_backscatter_count"] = None
            elif label == "nb transmitted":
                block = ensure_block()
                try:
                    block["reported_transmitted_count"] = int(float(value))
                except ValueError:
                    block["reported_transmitted_count"] = None
        row = _parse_collision_row(line)
        if row is not None and current_trajectory_id is not None:
            if len(current_rows) < 2:
                current_rows.append(row)
            else:
                current_rows[-1] = row
    finalize_current()

    normal, tangent_1, tangent_2, measured_tilt_deg = _fit_entry_frame(entry_points_nm, entry_dirs)
    beam_dir = np.mean(np.asarray(entry_dirs, dtype=np.float64), axis=0)
    beam_dir = beam_dir / max(float(np.linalg.norm(beam_dir)), 1e-12)
    projected_beam = beam_dir - float(np.dot(beam_dir, normal)) * normal
    projected_beam_tangent_norm = float(np.linalg.norm(projected_beam))
    frame_handedness = float(np.dot(np.cross(normal, tangent_1), tangent_2))
    exit_samples: list[ExitSample] = []
    blocks_by_index = {int(block["scan_point_block_index"]): block for block in scan_point_blocks}
    for block_index, scan_point_id, trajectory_id, trajectory_type, rows in trajectories:
        block = blocks_by_index[block_index]
        if trajectory_type == "Backscattered":
            block["backscattered_count"] += 1
            if len(rows) >= 2:
                exit_row = rows[-1]
                phi_rad, varphi_rad, cos_phi = _direction_to_angles(
                    np.asarray([exit_row.rx, exit_row.ry, exit_row.rz], dtype=np.float64),
                    normal,
                    tangent_1,
                    tangent_2,
                )
                exit_samples.append(
                    ExitSample(
                        run_id=run_id,
                        tilt_deg=measured_tilt_deg,
                        nominal_tilt_deg=nominal_tilt_deg,
                        repeat_id=repeat_id,
                        trajectory_id=trajectory_id,
                        x_nm=exit_row.x_nm,
                        y_nm=exit_row.y_nm,
                        z_nm=exit_row.z_nm,
                        rx=exit_row.rx,
                        ry=exit_row.ry,
                        rz=exit_row.rz,
                        exit_energy_kev=exit_row.energy_kev,
                        phi_rad=phi_rad,
                        varphi_rad=varphi_rad,
                        cos_phi=cos_phi,
                        scan_point_id=scan_point_id,
                        scan_point_block_index=block_index,
                        scan_point_run_id=str(block["scan_point_run_id"]),
                    )
                )
                block["exit_sample_count"] += 1
        elif trajectory_type == "Transmitted":
            block["transmitted_count"] += 1
        elif trajectory_type == "Simple":
            block["simple_count"] += 1

    for block in scan_point_blocks:
        reported_backscatter = block["reported_backscatter_count"]
        reported_transmitted = block["reported_transmitted_count"]
        block["backscatter_count_delta"] = (
            None
            if reported_backscatter is None
            else int(block["backscattered_count"]) - int(reported_backscatter)
        )
        block["transmitted_count_delta"] = (
            None
            if reported_transmitted is None
            else int(block["transmitted_count"]) - int(reported_transmitted)
        )
        available_deltas = [
            block[key]
            for key in ("backscatter_count_delta", "transmitted_count_delta")
            if block[key] is not None
        ]
        block["counts_match"] = None if not available_deltas else all(int(delta) == 0 for delta in available_deltas)
        block["counts_within_export_tolerance"] = (
            None if not available_deltas else all(abs(int(delta)) <= 1 for delta in available_deltas)
        )
        if not available_deltas:
            block["count_status"] = "unavailable"
        elif block["counts_match"]:
            block["count_status"] = "exact_match"
        elif block["counts_within_export_tolerance"]:
            block["count_status"] = "within_one_trajectory_export_tolerance"
        else:
            block["count_status"] = "mismatch"

    def sum_reported(key: str) -> int | None:
        values = [block[key] for block in scan_point_blocks if block[key] is not None]
        return None if not values else int(sum(int(value) for value in values))

    reported_backscatter_count = sum_reported("reported_backscatter_count")
    reported_transmitted_count = sum_reported("reported_transmitted_count")
    eta_values = [
        (float(block["reported_eta_bse"]), block["reported_backscatter_count"])
        for block in scan_point_blocks
        if block["reported_eta_bse"] is not None
    ]
    eta_weights = [
        float(reported_count) / eta
        for eta, reported_count in eta_values
        if reported_count is not None and eta > 0.0
    ]
    if eta_values and len(eta_weights) == len(eta_values) and sum(eta_weights) > 0.0:
        reported_eta_bse = float(
            sum(eta * weight for (eta, _), weight in zip(eta_values, eta_weights)) / sum(eta_weights)
        )
        eta_aggregation = "inferred_primary_count_weighted"
    elif eta_values:
        reported_eta_bse = float(np.mean([eta for eta, _ in eta_values]))
        eta_aggregation = "arithmetic_mean"
    else:
        reported_eta_bse = None
        eta_aggregation = None

    backscattered_count = int(sum(int(block["backscattered_count"]) for block in scan_point_blocks))
    transmitted_count = int(sum(int(block["transmitted_count"]) for block in scan_point_blocks))
    simple_count = int(sum(int(block["simple_count"]) for block in scan_point_blocks))
    reported_backscatter_delta = (
        None if reported_backscatter_count is None else backscattered_count - reported_backscatter_count
    )
    reported_transmitted_delta = (
        None if reported_transmitted_count is None else transmitted_count - reported_transmitted_count
    )
    aggregate_deltas = [
        delta for delta in (reported_backscatter_delta, reported_transmitted_delta) if delta is not None
    ]
    reported_counts_match = None if not aggregate_deltas else all(delta == 0 for delta in aggregate_deltas)
    reported_counts_within_export_tolerance = (
        None
        if not aggregate_deltas
        else all(abs(delta) <= len(scan_point_blocks) for delta in aggregate_deltas)
    )

    summary_meta = {
        "scan_point_count": len(scan_point_blocks),
        "scan_point_blocks": scan_point_blocks,
        "trajectory_count": trajectory_count,
        "backscattered_count": backscattered_count,
        "transmitted_count": transmitted_count,
        "simple_count": simple_count,
        "exit_sample_count": len(exit_samples),
        "reported_backscatter_count": reported_backscatter_count,
        "reported_transmitted_count": reported_transmitted_count,
        "reported_eta_bse": reported_eta_bse,
        "reported_eta_bse_aggregation": eta_aggregation,
        "reported_backscatter_count_delta": reported_backscatter_delta,
        "reported_transmitted_count_delta": reported_transmitted_delta,
        "reported_counts_match": reported_counts_match,
        "reported_counts_within_export_tolerance": reported_counts_within_export_tolerance,
        "measured_tilt_deg": measured_tilt_deg,
        "entry_beam_direction": beam_dir.tolist(),
        "entry_normal": normal.tolist(),
        "entry_tangent_1": tangent_1.tolist(),
        "entry_tangent_2": tangent_2.tolist(),
        "frame_normal_norm_error": abs(float(np.linalg.norm(normal)) - 1.0),
        "frame_tangent_1_norm_error": abs(float(np.linalg.norm(tangent_1)) - 1.0),
        "frame_tangent_2_norm_error": abs(float(np.linalg.norm(tangent_2)) - 1.0),
        "frame_normal_dot_tangent_1_abs": abs(float(np.dot(normal, tangent_1))),
        "frame_normal_dot_tangent_2_abs": abs(float(np.dot(normal, tangent_2))),
        "frame_tangent_dot_abs": abs(float(np.dot(tangent_1, tangent_2))),
        "frame_handedness": frame_handedness,
        "projected_beam_tangent_norm": projected_beam_tangent_norm,
        "weak_azimuth_orientation": projected_beam_tangent_norm < _WEAK_FRAME_SIN_ALPHA,
    }
    return exit_samples, summary_meta


def build_trajectory_angle_audit(
    summary_path: str | Path,
    run_id: str,
    nominal_tilt_deg: float,
    repeat_id: int,
    trajectory_id: int,
    scan_point_id: int = 0,
    b_value: float | None = None,
) -> dict[str, Any]:
    """Recompute one trajectory's RR angles from its raw CASINO direction.

    The incident beam direction points into the specimen.  The fitted surface
    normal points out of the specimen, so incidence uses ``-beam_direction``.
    Azimuth zero follows the incident beam projected into the entrance plane.
    """
    samples, meta = _parse_summary_file(
        summary_path,
        run_id,
        nominal_tilt_deg=nominal_tilt_deg,
        repeat_id=repeat_id,
    )
    matches = [
        sample
        for sample in samples
        if sample.trajectory_id == trajectory_id and sample.scan_point_id == scan_point_id
    ]
    if not matches:
        raise ValueError(
            f"No backscattered exit sample for run={run_id!r}, "
            f"scan_point_id={scan_point_id}, trajectory_id={trajectory_id}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous trajectory selection for run={run_id!r}, "
            f"scan_point_id={scan_point_id}, trajectory_id={trajectory_id}."
        )

    sample = matches[0]
    beam = np.asarray(meta["entry_beam_direction"], dtype=np.float64)
    normal = np.asarray(meta["entry_normal"], dtype=np.float64)
    tangent_1 = np.asarray(meta["entry_tangent_1"], dtype=np.float64)
    tangent_2 = np.asarray(meta["entry_tangent_2"], dtype=np.float64)
    direction = np.asarray([sample.rx, sample.ry, sample.rz], dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    unit = direction / direction_norm

    alpha_cosine = float(np.clip(np.dot(normal, -beam), -1.0, 1.0))
    phi_cosine = float(np.clip(np.dot(unit, normal), -1.0, 1.0))
    tangent_1_component = float(np.dot(unit, tangent_1))
    tangent_2_component = float(np.dot(unit, tangent_2))
    alpha_rad = float(math.acos(alpha_cosine))
    phi_rad = float(math.acos(phi_cosine))
    varphi_rad = float(math.atan2(tangent_2_component, tangent_1_component))

    reconstructed = (
        math.cos(phi_rad) * normal
        + math.sin(phi_rad) * math.cos(varphi_rad) * tangent_1
        + math.sin(phi_rad) * math.sin(varphi_rad) * tangent_2
    )
    asymmetry_factor = float(math.sin(alpha_rad) * math.sin(phi_rad) * math.cos(varphi_rad))
    baseline_density_per_sr = float(math.cos(phi_rad) / math.pi)
    rr_evaluation = None
    if b_value is not None:
        bracket = float(1.0 + b_value * asymmetry_factor)
        rr_evaluation = {
            "b": float(b_value),
            "bracket_1_plus_b_times_asymmetry_factor": bracket,
            "density_per_sr": float(baseline_density_per_sr * bracket),
        }

    return {
        "schema_version": 1,
        "source": {
            "run_id": run_id,
            "scan_point_id": int(scan_point_id),
            "scan_point_run_id": sample.scan_point_run_id,
            "trajectory_id": int(trajectory_id),
            "summary_path": str(Path(summary_path).resolve()),
            "summary_sha256": _sha256_file(summary_path),
        },
        "angle_convention": {
            "alpha": "angle between outward surface normal n and direction toward the source -b",
            "phi": "polar exit angle between unit exit direction u and outward surface normal n",
            "varphi": "signed exit azimuth atan2(u dot t2, u dot t1), in [-pi, pi]",
            "tangent_1": "normalized projection of the into-specimen beam direction b onto the entrance plane",
            "tangent_2": "n cross t1",
        },
        "frame": {
            "beam_direction_into_specimen_b": beam.tolist(),
            "outward_surface_normal_n": normal.tolist(),
            "in_plane_tangent_t1": tangent_1.tolist(),
            "in_plane_tangent_t2": tangent_2.tolist(),
            "orthogonality_max_abs_error": float(
                max(
                    abs(np.dot(normal, tangent_1)),
                    abs(np.dot(normal, tangent_2)),
                    abs(np.dot(tangent_1, tangent_2)),
                )
            ),
            "handedness_n_cross_t1_dot_t2": float(np.dot(np.cross(normal, tangent_1), tangent_2)),
        },
        "trajectory": {
            "raw_exit_direction": direction.tolist(),
            "raw_exit_direction_norm": direction_norm,
            "unit_exit_direction_u": unit.tolist(),
            "exit_position_nm": [sample.x_nm, sample.y_nm, sample.z_nm],
            "exit_energy_kev": sample.exit_energy_kev,
        },
        "angles": {
            "alpha": {
                "formula": "acos(clip(n dot (-b), -1, 1))",
                "cosine_argument": alpha_cosine,
                "radians": alpha_rad,
                "degrees": float(math.degrees(alpha_rad)),
                "nominal_degrees": float(nominal_tilt_deg),
                "measured_minus_nominal_degrees": float(math.degrees(alpha_rad) - nominal_tilt_deg),
            },
            "phi": {
                "formula": "acos(clip(u dot n, -1, 1))",
                "cosine_argument": phi_cosine,
                "radians": phi_rad,
                "degrees": float(math.degrees(phi_rad)),
            },
            "varphi": {
                "formula": "atan2(u dot t2, u dot t1)",
                "u_dot_t1": tangent_1_component,
                "u_dot_t2": tangent_2_component,
                "radians": varphi_rad,
                "degrees": float(math.degrees(varphi_rad)),
            },
        },
        "checks": {
            "parser_phi_abs_error_rad": abs(phi_rad - sample.phi_rad),
            "parser_varphi_abs_error_rad": abs(varphi_rad - sample.varphi_rad),
            "parser_alpha_abs_error_deg": abs(math.degrees(alpha_rad) - sample.tilt_deg),
            "unit_direction_reconstruction_l2_error": float(np.linalg.norm(reconstructed - unit)),
        },
        "rr_terms": {
            "formula": "p(phi,varphi|alpha,B) = cos(phi)/pi * (1 + B sin(alpha) sin(phi) cos(varphi))",
            "baseline_density_cos_phi_over_pi_per_sr": baseline_density_per_sr,
            "sin_alpha": float(math.sin(alpha_rad)),
            "sin_phi": float(math.sin(phi_rad)),
            "cos_varphi": float(math.cos(varphi_rad)),
            "asymmetry_factor_sin_alpha_sin_phi_cos_varphi": asymmetry_factor,
            "evaluation": rr_evaluation,
        },
    }


def write_trajectory_angle_audit(
    output_dir: str | Path,
    **audit_kwargs: Any,
) -> dict[str, Any]:
    """Write JSON and Markdown artifacts for ``build_trajectory_angle_audit``."""
    audit = build_trajectory_angle_audit(**audit_kwargs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "trajectory_angle_audit.json"
    markdown_path = output_path / "trajectory_angle_audit.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    source = audit["source"]
    frame = audit["frame"]
    trajectory = audit["trajectory"]
    angles = audit["angles"]
    rr_terms = audit["rr_terms"]
    checks = audit["checks"]
    evaluation = rr_terms["evaluation"]
    lines = [
        "# Stage 2: one CASINO trajectory to RR angles",
        "",
        f"- Run: `{source['run_id']}`",
        f"- Scan point: `{source['scan_point_run_id']}`",
        f"- Trajectory: `{source['trajectory_id']}`",
        f"- Source SHA-256: `{source['summary_sha256']}`",
        "",
        "## Coordinate convention",
        "",
        "`b` points into the specimen, `n` points out of the specimen, `t1` is the in-plane projection of `b`, and `t2 = n x t1`.",
        "Thus `varphi = 0` points along the projected into-specimen beam direction.",
        "",
        "## Raw vectors and fitted frame",
        "",
        f"- `b = {frame['beam_direction_into_specimen_b']}`",
        f"- `n = {frame['outward_surface_normal_n']}`",
        f"- `t1 = {frame['in_plane_tangent_t1']}`",
        f"- `t2 = {frame['in_plane_tangent_t2']}`",
        f"- raw exit direction `r = {trajectory['raw_exit_direction']}`",
        f"- `||r|| = {trajectory['raw_exit_direction_norm']:.12f}`",
        f"- unit exit direction `u = {trajectory['unit_exit_direction_u']}`",
        "",
        "## Angle calculations",
        "",
        "| Angle | Calculation | Result |",
        "|---|---|---:|",
        f"| alpha | `acos(n dot (-b))`, cosine argument `{angles['alpha']['cosine_argument']:.12f}` | `{angles['alpha']['degrees']:.9f} deg` |",
        f"| phi | `acos(u dot n)`, cosine argument `{angles['phi']['cosine_argument']:.12f}` | `{angles['phi']['degrees']:.9f} deg` |",
        f"| varphi | `atan2(u dot t2, u dot t1)`, components `({angles['varphi']['u_dot_t2']:.12f}, {angles['varphi']['u_dot_t1']:.12f})` | `{angles['varphi']['degrees']:.9f} deg` |",
        "",
        f"The nominal alpha is `{angles['alpha']['nominal_degrees']:.6f} deg`; the plane-fitted value differs by `{angles['alpha']['measured_minus_nominal_degrees']:.9f} deg`.",
        "",
        "## Entry into the RR likelihood",
        "",
        "`p(phi,varphi | alpha,B) = cos(phi)/pi * [1 + B sin(alpha) sin(phi) cos(varphi)]`",
        "",
        f"For this trajectory, `sin(alpha) sin(phi) cos(varphi) = {rr_terms['asymmetry_factor_sin_alpha_sin_phi_cos_varphi']:.12f}` and the Lambertian baseline `cos(phi)/pi = {rr_terms['baseline_density_cos_phi_over_pi_per_sr']:.12f} sr^-1`.",
    ]
    if evaluation is not None:
        lines.extend(
            [
                f"Using `B = {evaluation['b']:.12f}`, the bracket is `{evaluation['bracket_1_plus_b_times_asymmetry_factor']:.12f}` and the trajectory density is `{evaluation['density_per_sr']:.12f} sr^-1`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Numerical checks",
            "",
            f"- Maximum frame orthogonality error: `{frame['orthogonality_max_abs_error']:.3e}`",
            f"- Direction reconstruction error: `{checks['unit_direction_reconstruction_l2_error']:.3e}`",
            f"- Recomputed versus parser phi error: `{checks['parser_phi_abs_error_rad']:.3e} rad`",
            f"- Recomputed versus parser varphi error: `{checks['parser_varphi_abs_error_rad']:.3e} rad`",
            f"- Recomputed versus parser alpha error: `{checks['parser_alpha_abs_error_deg']:.3e} deg`",
            "",
            "The reconstruction is `u = cos(phi)n + sin(phi)cos(varphi)t1 + sin(phi)sin(varphi)t2`.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def _collect_run_data(
    cas_root: str | Path,
    export_root: str | Path,
) -> tuple[list[RunSummaryFitData], list[ScanPointSummaryFitData], list[ExitSample]]:
    cas_dir = Path(cas_root).resolve()
    export_dir = Path(export_root).resolve()
    run_summaries: list[RunSummaryFitData] = []
    scan_point_summaries: list[ScanPointSummaryFitData] = []
    all_samples: list[ExitSample] = []

    for summary_path in sorted(cas_dir.rglob("*_summary.dat")):
        run_id = summary_path.name[: -len("_summary.dat")]
        nominal_tilt_deg, repeat_id = _parse_run_id(run_id)
        bs_angle_path = export_dir / run_id / f"{run_id}_BSAngle.dat"
        bs_energy_path = export_dir / run_id / f"{run_id}_BSEnergy.dat"
        summary_sha256 = _sha256_file(summary_path)
        bs_angle_sha256 = _sha256_file(bs_angle_path) if bs_angle_path.exists() else None
        bs_energy_sha256 = _sha256_file(bs_energy_path) if bs_energy_path.exists() else None

        exit_samples, meta = _parse_summary_file(summary_path, run_id, nominal_tilt_deg, repeat_id)
        all_samples.extend(exit_samples)
        measured_tilt_deg = float(meta["measured_tilt_deg"])
        entry_normal = meta["entry_normal"]
        entry_tangent_1 = meta["entry_tangent_1"]
        entry_tangent_2 = meta["entry_tangent_2"]

        for block in meta["scan_point_blocks"]:
            scan_point_summaries.append(
                ScanPointSummaryFitData(
                    source_run_id=run_id,
                    scan_point_run_id=str(block["scan_point_run_id"]),
                    scan_point_id=int(block["scan_point_id"]),
                    scan_point_block_index=int(block["scan_point_block_index"]),
                    nominal_tilt_deg=nominal_tilt_deg,
                    measured_tilt_deg=measured_tilt_deg,
                    repeat_id=repeat_id,
                    scan_x=block["scan_x"],
                    scan_y=block["scan_y"],
                    initial_energy_kev=block["initial_energy_kev"],
                    summary_path=str(summary_path),
                    summary_sha256=summary_sha256,
                    result_line_number=block["result_line_number"],
                    trajectory_count=int(block["trajectory_count"]),
                    backscattered_count=int(block["backscattered_count"]),
                    transmitted_count=int(block["transmitted_count"]),
                    simple_count=int(block["simple_count"]),
                    exit_sample_count=int(block["exit_sample_count"]),
                    reported_backscatter_count=block["reported_backscatter_count"],
                    reported_transmitted_count=block["reported_transmitted_count"],
                    reported_eta_bse=block["reported_eta_bse"],
                    backscatter_count_delta=block["backscatter_count_delta"],
                    transmitted_count_delta=block["transmitted_count_delta"],
                    counts_match=block["counts_match"],
                    counts_within_export_tolerance=block["counts_within_export_tolerance"],
                    count_status=str(block["count_status"]),
                )
            )

        angle_ce = angle_l1 = rr_angle_ce = None
        energy_ce = energy_l1 = summary_mean = export_mean = None
        if bs_angle_path.exists() and exit_samples:
            angle_xy = _parse_distribution_file(bs_angle_path)
            angle_obs = angle_xy[:, 1]
            angle_empirical = _angle_histogram_from_samples(exit_samples, angle_xy)
            angle_ce = _cross_entropy(angle_obs, angle_empirical)
            angle_l1 = _l1_distance(angle_obs, angle_empirical)
            rr_angle_ce = _cross_entropy(angle_obs, _rr_polar_histogram(angle_xy[:, 0]))
        if bs_energy_path.exists() and exit_samples:
            energy_xy = _parse_distribution_file(bs_energy_path)
            energy_obs = energy_xy[:, 1]
            energy_empirical = _energy_histogram_from_samples(exit_samples, energy_xy)
            energy_ce = _cross_entropy(energy_obs, energy_empirical)
            energy_l1 = _l1_distance(energy_obs, energy_empirical)
            summary_mean = float(np.mean([sample.exit_energy_kev for sample in exit_samples]))
            export_mean = _histogram_to_mean(energy_xy)

        run_summaries.append(
            RunSummaryFitData(
                run_id=run_id,
                tilt_deg=measured_tilt_deg,
                nominal_tilt_deg=nominal_tilt_deg,
                repeat_id=repeat_id,
                summary_path=str(summary_path),
                summary_sha256=summary_sha256,
                bs_angle_path=str(bs_angle_path) if bs_angle_path.exists() else None,
                bs_angle_sha256=bs_angle_sha256,
                bs_energy_path=str(bs_energy_path) if bs_energy_path.exists() else None,
                bs_energy_sha256=bs_energy_sha256,
                scan_point_count=int(meta["scan_point_count"]),
                trajectory_count=int(meta["trajectory_count"]),
                backscattered_count=int(meta["backscattered_count"]),
                transmitted_count=int(meta["transmitted_count"]),
                simple_count=int(meta["simple_count"]),
                exit_sample_count=int(meta["exit_sample_count"]),
                reported_backscatter_count=meta["reported_backscatter_count"],
                reported_transmitted_count=meta["reported_transmitted_count"],
                reported_eta_bse=meta["reported_eta_bse"],
                reported_backscatter_count_delta=meta["reported_backscatter_count_delta"],
                reported_transmitted_count_delta=meta["reported_transmitted_count_delta"],
                reported_counts_match=meta["reported_counts_match"],
                reported_counts_within_export_tolerance=meta["reported_counts_within_export_tolerance"],
                summary_vs_bsangle_cross_entropy=angle_ce,
                summary_vs_bsangle_l1=angle_l1,
                rr_polar_vs_bsangle_cross_entropy=rr_angle_ce,
                summary_vs_bsenergy_cross_entropy=energy_ce,
                summary_vs_bsenergy_l1=energy_l1,
                summary_energy_mean_kev=summary_mean,
                export_energy_mean_kev=export_mean,
                entry_normal_x=float(entry_normal[0]),
                entry_normal_y=float(entry_normal[1]),
                entry_normal_z=float(entry_normal[2]),
                entry_tangent_1_x=float(entry_tangent_1[0]),
                entry_tangent_1_y=float(entry_tangent_1[1]),
                entry_tangent_1_z=float(entry_tangent_1[2]),
                entry_tangent_2_x=float(entry_tangent_2[0]),
                entry_tangent_2_y=float(entry_tangent_2[1]),
                entry_tangent_2_z=float(entry_tangent_2[2]),
                frame_normal_norm_error=float(meta["frame_normal_norm_error"]),
                frame_tangent_1_norm_error=float(meta["frame_tangent_1_norm_error"]),
                frame_tangent_2_norm_error=float(meta["frame_tangent_2_norm_error"]),
                frame_normal_dot_tangent_1_abs=float(meta["frame_normal_dot_tangent_1_abs"]),
                frame_normal_dot_tangent_2_abs=float(meta["frame_normal_dot_tangent_2_abs"]),
                frame_tangent_dot_abs=float(meta["frame_tangent_dot_abs"]),
                frame_handedness=float(meta["frame_handedness"]),
                projected_beam_tangent_norm=float(meta["projected_beam_tangent_norm"]),
                weak_azimuth_orientation=bool(meta["weak_azimuth_orientation"]),
            )
        )

    if not run_summaries:
        raise ValueError(f"No '*_summary.dat' files found under '{cas_dir}'.")
    return run_summaries, scan_point_summaries, all_samples


def _build_b_grid(tilt_deg_values: np.ndarray, grid_points: int) -> np.ndarray:
    max_sin_alpha = float(np.max(np.abs(np.sin(np.radians(tilt_deg_values)))))
    if max_sin_alpha <= 1e-12:
        limit = 1.0
    else:
        limit = min(2.0, 0.999 / max_sin_alpha)
    return np.linspace(-limit, limit, max(101, int(grid_points)), dtype=np.float64)


def _group_samples_by_run(samples: list[ExitSample]) -> dict[str, list[ExitSample]]:
    grouped: dict[str, list[ExitSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.run_id, []).append(sample)
    return grouped


def _vector_from_run(row: RunSummaryFitData, prefix: str) -> np.ndarray:
    return np.asarray(
        [getattr(row, f"{prefix}_x"), getattr(row, f"{prefix}_y"), getattr(row, f"{prefix}_z")],
        dtype=np.float64,
    )


def _build_coordinate_frame_diagnostics(
    run_summaries: list[RunSummaryFitData],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_by_repeat: dict[int, RunSummaryFitData] = {}
    ordered = sorted(run_summaries, key=lambda row: (row.repeat_id, row.nominal_tilt_deg, row.run_id))

    for run in ordered:
        normal = _vector_from_run(run, "entry_normal")
        tangent_1 = _vector_from_run(run, "entry_tangent_1")
        tangent_2 = _vector_from_run(run, "entry_tangent_2")
        previous = previous_by_repeat.get(run.repeat_id)
        previous_run_id: str | None = None
        previous_nominal_tilt_deg: float | None = None
        normal_dot_previous: float | None = None
        tangent_1_dot_previous: float | None = None
        tangent_2_dot_previous: float | None = None
        tangent_1_change_deg: float | None = None
        tangent_2_change_deg: float | None = None
        continuity_evaluable = False
        tangent_flip_detected = False

        if previous is not None:
            previous_run_id = previous.run_id
            previous_nominal_tilt_deg = float(previous.nominal_tilt_deg)
            previous_normal = _vector_from_run(previous, "entry_normal")
            previous_tangent_1 = _vector_from_run(previous, "entry_tangent_1")
            previous_tangent_2 = _vector_from_run(previous, "entry_tangent_2")
            normal_dot_previous = float(np.clip(np.dot(previous_normal, normal), -1.0, 1.0))
            tangent_1_dot_previous = float(np.clip(np.dot(previous_tangent_1, tangent_1), -1.0, 1.0))
            tangent_2_dot_previous = float(np.clip(np.dot(previous_tangent_2, tangent_2), -1.0, 1.0))
            tangent_1_change_deg = float(math.degrees(math.acos(tangent_1_dot_previous)))
            tangent_2_change_deg = float(math.degrees(math.acos(tangent_2_dot_previous)))
            continuity_evaluable = not (run.weak_azimuth_orientation or previous.weak_azimuth_orientation)
            tangent_flip_detected = bool(
                continuity_evaluable and (tangent_1_dot_previous < 0.0 or tangent_2_dot_previous < 0.0)
            )

        rows.append(
            {
                "run_id": run.run_id,
                "repeat_id": int(run.repeat_id),
                "nominal_tilt_deg": float(run.nominal_tilt_deg),
                "measured_tilt_deg": float(run.tilt_deg),
                "entry_normal_x": float(normal[0]),
                "entry_normal_y": float(normal[1]),
                "entry_normal_z": float(normal[2]),
                "entry_tangent_1_x": float(tangent_1[0]),
                "entry_tangent_1_y": float(tangent_1[1]),
                "entry_tangent_1_z": float(tangent_1[2]),
                "entry_tangent_2_x": float(tangent_2[0]),
                "entry_tangent_2_y": float(tangent_2[1]),
                "entry_tangent_2_z": float(tangent_2[2]),
                "normal_norm_error": float(run.frame_normal_norm_error),
                "tangent_1_norm_error": float(run.frame_tangent_1_norm_error),
                "tangent_2_norm_error": float(run.frame_tangent_2_norm_error),
                "normal_dot_tangent_1_abs": float(run.frame_normal_dot_tangent_1_abs),
                "normal_dot_tangent_2_abs": float(run.frame_normal_dot_tangent_2_abs),
                "tangent_1_dot_tangent_2_abs": float(run.frame_tangent_dot_abs),
                "handedness": float(run.frame_handedness),
                "projected_beam_tangent_norm": float(run.projected_beam_tangent_norm),
                "weak_azimuth_orientation": bool(run.weak_azimuth_orientation),
                "previous_run_id": previous_run_id,
                "previous_nominal_tilt_deg": previous_nominal_tilt_deg,
                "normal_dot_previous": normal_dot_previous,
                "tangent_1_dot_previous": tangent_1_dot_previous,
                "tangent_2_dot_previous": tangent_2_dot_previous,
                "tangent_1_change_deg": tangent_1_change_deg,
                "tangent_2_change_deg": tangent_2_change_deg,
                "continuity_evaluable": continuity_evaluable,
                "tangent_flip_detected": tangent_flip_detected,
            }
        )
        previous_by_repeat[run.repeat_id] = run

    evaluable = [row for row in rows if row["continuity_evaluable"]]
    summary = {
        "run_count": len(rows),
        "weak_orientation_run_count": sum(bool(row["weak_azimuth_orientation"]) for row in rows),
        "weak_orientation_run_ids": [row["run_id"] for row in rows if row["weak_azimuth_orientation"]],
        "continuity_comparison_count": len(evaluable),
        "tangent_flip_count": sum(bool(row["tangent_flip_detected"]) for row in evaluable),
        "tangent_flip_run_ids": [row["run_id"] for row in evaluable if row["tangent_flip_detected"]],
        "minimum_tangent_1_dot_previous": None
        if not evaluable
        else float(min(float(row["tangent_1_dot_previous"]) for row in evaluable)),
        "minimum_tangent_2_dot_previous": None
        if not evaluable
        else float(min(float(row["tangent_2_dot_previous"]) for row in evaluable)),
        "maximum_norm_error": None
        if not rows
        else float(
            max(
                max(row["normal_norm_error"], row["tangent_1_norm_error"], row["tangent_2_norm_error"])
                for row in rows
            )
        ),
        "maximum_orthogonality_error": None
        if not rows
        else float(
            max(
                max(
                    row["normal_dot_tangent_1_abs"],
                    row["normal_dot_tangent_2_abs"],
                    row["tangent_1_dot_tangent_2_abs"],
                )
                for row in rows
            )
        ),
        "minimum_handedness": None if not rows else float(min(row["handedness"] for row in rows)),
    }
    return rows, summary


def _harmonic_statistics(
    samples: list[ExitSample],
    *,
    varphi_override: np.ndarray | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Need at least one exit sample for harmonic diagnostics.")
    phi = np.asarray([sample.phi_rad for sample in samples], dtype=np.float64)
    varphi = (
        np.asarray([sample.varphi_rad for sample in samples], dtype=np.float64)
        if varphi_override is None
        else np.asarray(varphi_override, dtype=np.float64)
    )
    if varphi.shape != phi.shape:
        raise ValueError("varphi_override must match the number of samples.")
    measured_tilt = np.asarray([sample.tilt_deg for sample in samples], dtype=np.float64)
    nominal_tilt = np.asarray([sample.nominal_tilt_deg for sample in samples], dtype=np.float64)
    cos_1_values = np.cos(varphi)
    sin_1_values = np.sin(varphi)
    cos_2_values = np.cos(2.0 * varphi)
    sin_2_values = np.sin(2.0 * varphi)
    cos_1 = float(2.0 * np.mean(cos_1_values))
    sin_1 = float(2.0 * np.mean(sin_1_values))
    cos_2 = float(2.0 * np.mean(cos_2_values))
    sin_2 = float(2.0 * np.mean(sin_2_values))
    first_amplitude = float(math.hypot(cos_1, sin_1))
    second_amplitude = float(math.hypot(cos_2, sin_2))
    first_phase_deg = float(math.degrees(math.atan2(sin_1, cos_1)))
    sample_count = int(len(samples))
    ddof = 1 if sample_count > 1 else 0
    cos_1_standard_error = float(2.0 * np.std(cos_1_values, ddof=ddof) / math.sqrt(sample_count))
    sin_1_standard_error = float(2.0 * np.std(sin_1_values, ddof=ddof) / math.sqrt(sample_count))
    measured_denominator = float(np.mean(np.sin(np.radians(measured_tilt)) * np.sin(phi)))
    nominal_denominator = float(np.mean(np.sin(np.radians(nominal_tilt)) * np.sin(phi)))

    def diagnostic_b(denominator: float) -> tuple[float | None, float | None, float | None, float | None]:
        if abs(denominator) < _WEAK_FRAME_SIN_ALPHA:
            return None, None, None, None
        value = float(cos_1 / denominator)
        standard_error = float(cos_1_standard_error / abs(denominator))
        return (
            value,
            standard_error,
            float(value - 1.96 * standard_error),
            float(value + 1.96 * standard_error),
        )

    b_measured, b_standard_error, b_ci_lower, b_ci_upper = diagnostic_b(measured_denominator)
    b_nominal, _, _, _ = diagnostic_b(nominal_denominator)
    edges = np.linspace(-math.pi, math.pi, 37, dtype=np.float64)
    counts, _ = np.histogram(varphi, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    observed = counts.astype(np.float64) / max(float(np.mean(counts)), 1e-12)
    predicted = 1.0 + cos_1 * np.cos(centres) + sin_1 * np.sin(centres)
    harmonic_rms_residual = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    return {
        "sample_count": sample_count,
        "measured_tilt_deg_mean": float(np.mean(measured_tilt)),
        "nominal_tilt_deg_mean": float(np.mean(nominal_tilt)),
        "phi_deg_mean": float(np.degrees(np.mean(phi))),
        "mean_sin_phi": float(np.mean(np.sin(phi))),
        "mean_measured_sin_alpha_sin_phi": measured_denominator,
        "mean_nominal_sin_alpha_sin_phi": nominal_denominator,
        "cos_first": cos_1,
        "sin_first": sin_1,
        "first_amplitude": first_amplitude,
        "first_phase_deg": first_phase_deg,
        "cos_first_standard_error": cos_1_standard_error,
        "sin_first_standard_error": sin_1_standard_error,
        "cos_first_ci_lower": float(cos_1 - 1.96 * cos_1_standard_error),
        "cos_first_ci_upper": float(cos_1 + 1.96 * cos_1_standard_error),
        "cos_second": cos_2,
        "sin_second": sin_2,
        "second_amplitude": second_amplitude,
        "first_to_second_amplitude_ratio": None
        if second_amplitude <= 1e-12
        else float(first_amplitude / second_amplitude),
        "harmonic_rms_residual": harmonic_rms_residual,
        "diagnostic_b_measured_tilt": b_measured,
        "diagnostic_b_nominal_tilt": b_nominal,
        "diagnostic_b_standard_error": b_standard_error,
        "diagnostic_b_ci_lower": b_ci_lower,
        "diagnostic_b_ci_upper": b_ci_upper,
    }


_REPEAT_HARMONIC_KEYS = (
    "repeat_cos_first_mean",
    "repeat_cos_first_std",
    "repeat_cos_first_standard_error",
    "repeat_cos_first_ci_lower",
    "repeat_cos_first_ci_upper",
    "repeat_diagnostic_b_mean",
    "repeat_diagnostic_b_std",
    "repeat_diagnostic_b_standard_error",
    "repeat_diagnostic_b_ci_lower",
    "repeat_diagnostic_b_ci_upper",
)


def _repeat_metric_summary(rows: list[dict[str, Any]], key: str, prefix: str) -> dict[str, float | None]:
    values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64)
    if values.size == 0:
        return {
            f"repeat_{prefix}_mean": None,
            f"repeat_{prefix}_std": None,
            f"repeat_{prefix}_standard_error": None,
            f"repeat_{prefix}_ci_lower": None,
            f"repeat_{prefix}_ci_upper": None,
        }
    mean = float(np.mean(values))
    std = 0.0 if values.size == 1 else float(np.std(values, ddof=1))
    standard_error = float(std / math.sqrt(values.size))
    return {
        f"repeat_{prefix}_mean": mean,
        f"repeat_{prefix}_std": std,
        f"repeat_{prefix}_standard_error": standard_error,
        f"repeat_{prefix}_ci_lower": float(mean - 1.96 * standard_error),
        f"repeat_{prefix}_ci_upper": float(mean + 1.96 * standard_error),
    }


def _make_harmonic_row(
    samples: list[ExitSample],
    *,
    aggregation_level: str,
    run_id: str | None,
    repeat_id: int | None,
    nominal_tilt_deg: float,
    polar_min_deg: float | None,
    polar_max_deg: float | None,
    run_count: int,
    repeat_count: int,
    repeat_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "aggregation_level": aggregation_level,
        "run_id": run_id,
        "repeat_id": repeat_id,
        "nominal_tilt_deg": float(nominal_tilt_deg),
        "polar_min_deg": polar_min_deg,
        "polar_max_deg": polar_max_deg,
        "run_count": int(run_count),
        "repeat_count": int(repeat_count),
        **_harmonic_statistics(samples),
    }
    row.update({key: None for key in _REPEAT_HARMONIC_KEYS})
    if repeat_rows:
        row.update(_repeat_metric_summary(repeat_rows, "cos_first", "cos_first"))
        row.update(
            _repeat_metric_summary(repeat_rows, "diagnostic_b_measured_tilt", "diagnostic_b")
        )
    return row


def _build_azimuth_harmonic_diagnostics(samples: list[ExitSample]) -> dict[str, Any]:
    grouped_by_run = _group_samples_by_run(samples)
    per_run_rows: list[dict[str, Any]] = []
    per_tilt_rows: list[dict[str, Any]] = []
    per_polar_band_rows: list[dict[str, Any]] = []

    for run_id, run_samples in sorted(grouped_by_run.items()):
        first = run_samples[0]
        per_run_rows.append(
            _make_harmonic_row(
                run_samples,
                aggregation_level="run",
                run_id=run_id,
                repeat_id=int(first.repeat_id),
                nominal_tilt_deg=float(first.nominal_tilt_deg),
                polar_min_deg=None,
                polar_max_deg=None,
                run_count=1,
                repeat_count=1,
            )
        )
        for polar_min, polar_max in _POLAR_BANDS_DEG:
            band_samples = [
                sample
                for sample in run_samples
                if polar_min <= math.degrees(sample.phi_rad) < polar_max
            ]
            if band_samples:
                per_polar_band_rows.append(
                    _make_harmonic_row(
                        band_samples,
                        aggregation_level="run",
                        run_id=run_id,
                        repeat_id=int(first.repeat_id),
                        nominal_tilt_deg=float(first.nominal_tilt_deg),
                        polar_min_deg=polar_min,
                        polar_max_deg=polar_max,
                        run_count=1,
                        repeat_count=1,
                    )
                )

    nominal_tilts = sorted({float(sample.nominal_tilt_deg) for sample in samples})
    for nominal_tilt in nominal_tilts:
        tilt_samples = [sample for sample in samples if float(sample.nominal_tilt_deg) == nominal_tilt]
        tilt_run_rows = [row for row in per_run_rows if float(row["nominal_tilt_deg"]) == nominal_tilt]
        run_ids = {sample.run_id for sample in tilt_samples}
        repeat_ids = {sample.repeat_id for sample in tilt_samples}
        per_tilt_rows.append(
            _make_harmonic_row(
                tilt_samples,
                aggregation_level="tilt",
                run_id=None,
                repeat_id=None,
                nominal_tilt_deg=nominal_tilt,
                polar_min_deg=None,
                polar_max_deg=None,
                run_count=len(run_ids),
                repeat_count=len(repeat_ids),
                repeat_rows=tilt_run_rows,
            )
        )
        for polar_min, polar_max in _POLAR_BANDS_DEG:
            band_samples = [
                sample
                for sample in tilt_samples
                if polar_min <= math.degrees(sample.phi_rad) < polar_max
            ]
            if not band_samples:
                continue
            band_run_rows = [
                row
                for row in per_polar_band_rows
                if row["aggregation_level"] == "run"
                and float(row["nominal_tilt_deg"]) == nominal_tilt
                and float(row["polar_min_deg"]) == polar_min
                and float(row["polar_max_deg"]) == polar_max
            ]
            per_polar_band_rows.append(
                _make_harmonic_row(
                    band_samples,
                    aggregation_level="tilt",
                    run_id=None,
                    repeat_id=None,
                    nominal_tilt_deg=nominal_tilt,
                    polar_min_deg=polar_min,
                    polar_max_deg=polar_max,
                    run_count=len({sample.run_id for sample in band_samples}),
                    repeat_count=len({sample.repeat_id for sample in band_samples}),
                    repeat_rows=band_run_rows,
                )
            )

    nonzero_samples = [
        sample
        for sample in samples
        if abs(math.sin(math.radians(float(sample.tilt_deg)))) >= _WEAK_FRAME_SIN_ALPHA
    ]
    controls: dict[str, Any] = {}
    if nonzero_samples:
        original_varphi = np.asarray([sample.varphi_rad for sample in nonzero_samples], dtype=np.float64)
        rng = np.random.default_rng(20260831)
        uniform_varphi = rng.uniform(-math.pi, math.pi, size=len(nonzero_samples))
        reversed_varphi = (original_varphi + math.pi + math.pi) % (2.0 * math.pi) - math.pi
        original_stats = _harmonic_statistics(nonzero_samples)
        uniform_stats = _harmonic_statistics(nonzero_samples, varphi_override=uniform_varphi)
        reversed_stats = _harmonic_statistics(nonzero_samples, varphi_override=reversed_varphi)
        controls = {
            "nonzero_tilt_original": original_stats,
            "uniform_azimuth": uniform_stats,
            "tangent_reversal": reversed_stats,
            "tangent_reversal_cos_sign_check": float(original_stats["cos_first"] + reversed_stats["cos_first"]),
            "tangent_reversal_sin_sign_check": float(original_stats["sin_first"] + reversed_stats["sin_first"]),
            "tangent_reversal_amplitude_delta": float(
                original_stats["first_amplitude"] - reversed_stats["first_amplitude"]
            ),
        }

    zero_tilt_row = None if not per_tilt_rows else min(per_tilt_rows, key=lambda row: abs(float(row["nominal_tilt_deg"])))
    informative_rows = [row for row in per_tilt_rows if abs(float(row["cos_first"])) >= 0.02]
    sine_to_cosine = [
        abs(float(row["sin_first"])) / max(abs(float(row["cos_first"])), 1e-12)
        for row in informative_rows
    ]
    sign_changes: list[dict[str, float]] = []
    ordered_informative = sorted(informative_rows, key=lambda row: float(row["nominal_tilt_deg"]))
    for previous, current in zip(ordered_informative, ordered_informative[1:]):
        if float(previous["cos_first"]) * float(current["cos_first"]) < 0.0:
            sign_changes.append(
                {
                    "from_nominal_tilt_deg": float(previous["nominal_tilt_deg"]),
                    "to_nominal_tilt_deg": float(current["nominal_tilt_deg"]),
                }
            )

    polar_consistency_rows: list[dict[str, Any]] = []
    tilt_band_rows = [row for row in per_polar_band_rows if row["aggregation_level"] == "tilt"]
    for nominal_tilt in nominal_tilts:
        values = np.asarray(
            [
                float(row["diagnostic_b_measured_tilt"])
                for row in tilt_band_rows
                if float(row["nominal_tilt_deg"]) == nominal_tilt
                and row["diagnostic_b_measured_tilt"] is not None
            ],
            dtype=np.float64,
        )
        polar_consistency_rows.append(
            {
                "nominal_tilt_deg": nominal_tilt,
                "polar_band_count": int(values.size),
                "diagnostic_b_mean": None if values.size == 0 else float(np.mean(values)),
                "diagnostic_b_std": None if values.size <= 1 else float(np.std(values, ddof=1)),
                "diagnostic_b_range": None if values.size == 0 else float(np.max(values) - np.min(values)),
            }
        )

    summary = {
        "per_run_row_count": len(per_run_rows),
        "per_tilt_row_count": len(per_tilt_rows),
        "per_polar_band_row_count": len(per_polar_band_rows),
        "polar_bands_deg": [[lower, upper] for lower, upper in _POLAR_BANDS_DEG],
        "zero_tilt_nominal_deg": None if zero_tilt_row is None else float(zero_tilt_row["nominal_tilt_deg"]),
        "zero_tilt_first_amplitude": None if zero_tilt_row is None else float(zero_tilt_row["first_amplitude"]),
        "informative_tilt_count": len(informative_rows),
        "median_abs_sine_to_cosine_ratio": None if not sine_to_cosine else float(np.median(sine_to_cosine)),
        "first_harmonic_dominant_tilt_count": sum(
            float(row["first_amplitude"]) > float(row["second_amplitude"]) for row in informative_rows
        ),
        "cosine_sign_changes": sign_changes,
        "interpretation": (
            "cos_first and sin_first are Fourier diagnostics of the azimuthal exit distribution. "
            "diagnostic_b_measured_tilt divides the cosine coefficient by mean(sin(alpha) sin(phi)); "
            "trajectory NLL remains the primary estimator."
        ),
    }
    return {
        "summary": summary,
        "controls": controls,
        "per_run": per_run_rows,
        "per_tilt": per_tilt_rows,
        "per_polar_band": per_polar_band_rows,
        "polar_band_consistency": polar_consistency_rows,
    }


def _sample_emission_components(samples: list[ExitSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_phi = np.asarray([sample.cos_phi for sample in samples], dtype=np.float64)
    if not len(cos_phi) or not np.isfinite(cos_phi).all() or np.any((cos_phi <= 0) | (cos_phi > 1)):
        raise ValueError("Trajectory likelihood requires finite outward exits with 0 < cos(phi) <= 1. Use benchmark-fit-b-surface to resolve landing faces and exit selection.")
    phi = np.asarray([sample.phi_rad for sample in samples], dtype=np.float64)
    varphi = np.asarray([sample.varphi_rad for sample in samples], dtype=np.float64)
    tilt_deg = np.asarray([sample.tilt_deg for sample in samples], dtype=np.float64)
    asym_factor = np.sin(np.radians(tilt_deg)) * np.sin(phi) * np.cos(varphi)
    return cos_phi, asym_factor, tilt_deg


def _pooled_negative_log_likelihood(
    cos_phi: np.ndarray,
    asym_factor: np.ndarray,
    b_values: np.ndarray | float,
) -> np.ndarray | float:
    if not len(cos_phi) or not np.isfinite(cos_phi).all() or np.any((cos_phi <= 0) | (cos_phi > 1)):
        raise ValueError("Invalid outward-hemisphere likelihood population.")
    b = np.atleast_1d(np.asarray(b_values, dtype=np.float64))
    sample_count = len(cos_phi)
    nll_sum = np.zeros_like(b)
    base_nll_sum = float(
        -np.sum(np.log(np.clip(cos_phi, 1e-12, None)) - math.log(math.pi))
    )
    for b_start in range(0, len(b), 64):
        b_stop = min(b_start + 64, len(b))
        b_chunk = b[b_start:b_stop]
        chunk_sum = np.full_like(b_chunk, base_nll_sum)
        for sample_start in range(0, sample_count, 65536):
            sample_stop = min(sample_start + 65536, sample_count)
            asym_chunk = asym_factor[sample_start:sample_stop, np.newaxis]
            asymmetry_term = np.clip(1.0 + asym_chunk * b_chunk[np.newaxis, :], 1e-12, None)
            chunk_sum -= np.sum(np.log(asymmetry_term), axis=0)
        nll_sum[b_start:b_stop] = chunk_sum
    nll = nll_sum / max(sample_count, 1)
    if np.isscalar(b_values):
        return float(nll[0])
    return nll


def _pooled_nll_derivative(asym_factor: np.ndarray, b_values: np.ndarray | float) -> np.ndarray | float:
    b = np.atleast_1d(np.asarray(b_values, dtype=np.float64))
    sample_count = len(asym_factor)
    derivative_sum = np.zeros_like(b)
    for b_start in range(0, len(b), 64):
        b_stop = min(b_start + 64, len(b))
        b_chunk = b[b_start:b_stop]
        chunk_sum = np.zeros_like(b_chunk)
        for sample_start in range(0, sample_count, 65536):
            sample_stop = min(sample_start + 65536, sample_count)
            asym_chunk = asym_factor[sample_start:sample_stop, np.newaxis]
            denom = np.clip(1.0 + asym_chunk * b_chunk[np.newaxis, :], 1e-12, None)
            chunk_sum -= np.sum(asym_chunk / denom, axis=0)
        derivative_sum[b_start:b_stop] = chunk_sum
    derivative = derivative_sum / max(sample_count, 1)
    if np.isscalar(b_values):
        return float(derivative[0])
    return derivative


def _pooled_nll_second_derivative(asym_factor: np.ndarray, b_values: np.ndarray | float) -> np.ndarray | float:
    b = np.atleast_1d(np.asarray(b_values, dtype=np.float64))
    sample_count = len(asym_factor)
    curvature_sum = np.zeros_like(b)
    for b_start in range(0, len(b), 64):
        b_stop = min(b_start + 64, len(b))
        b_chunk = b[b_start:b_stop]
        chunk_sum = np.zeros_like(b_chunk)
        for sample_start in range(0, sample_count, 65536):
            sample_stop = min(sample_start + 65536, sample_count)
            asym_chunk = asym_factor[sample_start:sample_stop, np.newaxis]
            denom = np.clip(1.0 + asym_chunk * b_chunk[np.newaxis, :], 1e-12, None)
            chunk_sum += np.sum((asym_chunk**2) / (denom**2), axis=0)
        curvature_sum[b_start:b_stop] = chunk_sum
    curvature = curvature_sum / max(sample_count, 1)
    if np.isscalar(b_values):
        return float(curvature[0])
    return curvature


def _finite_or_none(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _observed_feasible_interval(asym_factor: np.ndarray) -> tuple[float, float]:
    lower = float("-inf")
    upper = float("inf")
    positive = asym_factor > 1e-12
    negative = asym_factor < -1e-12
    if np.any(positive):
        lower = float(np.max(-1.0 / asym_factor[positive]))
    if np.any(negative):
        upper = float(np.min(-1.0 / asym_factor[negative]))
    return lower, upper


def _domain_feasible_interval(tilt_deg_values: np.ndarray) -> tuple[float, float]:
    max_sin_alpha = float(np.max(np.abs(np.sin(np.radians(tilt_deg_values)))))
    if max_sin_alpha <= 1e-12:
        return float("-inf"), float("inf")
    limit = 1.0 / max_sin_alpha
    return -limit, limit


def _fit_linear_trend(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, Any] | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size < 2 or np.allclose(x, x[0]):
        return None
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": r_squared,
        "point_count": int(x.size),
    }


def _fit_pooled_b_with_diagnostics(
    samples: list[ExitSample],
    *,
    grid_points: int = 2001,
    b_grid: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Need at least one exit sample to fit B.")

    cos_phi, asym_factor, tilt_deg_values = _sample_emission_components(samples)
    if b_grid is None:
        b_grid = _build_b_grid(tilt_deg_values, grid_points)
    else:
        b_grid = np.asarray(b_grid, dtype=np.float64)
        if b_grid.ndim != 1 or len(b_grid) < 2:
            raise ValueError("B grid must contain at least two points.")

    objective_grid = np.asarray(_pooled_negative_log_likelihood(cos_phi, asym_factor, b_grid), dtype=np.float64)
    derivative_grid = np.asarray(_pooled_nll_derivative(asym_factor, b_grid), dtype=np.float64)
    curvature_grid = np.asarray(_pooled_nll_second_derivative(asym_factor, b_grid), dtype=np.float64)

    grid_idx = int(np.argmin(objective_grid))
    grid_fitted_b = float(b_grid[grid_idx])
    grid_objective = float(objective_grid[grid_idx])

    solver_lower = float(b_grid[0])
    solver_upper = float(b_grid[-1])
    max_abs_asym = float(np.max(np.abs(asym_factor))) if len(asym_factor) else 0.0

    if max_abs_asym <= 1e-12:
        fitted_b = 0.0
        solution_method = "flat_objective"
    else:
        lower_derivative = float(_pooled_nll_derivative(asym_factor, solver_lower))
        upper_derivative = float(_pooled_nll_derivative(asym_factor, solver_upper))
        if lower_derivative >= 0.0:
            fitted_b = solver_lower
            solution_method = "lower_boundary"
        elif upper_derivative <= 0.0:
            fitted_b = solver_upper
            solution_method = "upper_boundary"
        else:
            lo = solver_lower
            hi = solver_upper
            fitted_b = 0.5 * (lo + hi)
            solution_method = "interior_root"
            for _ in range(120):
                mid = 0.5 * (lo + hi)
                derivative_mid = float(_pooled_nll_derivative(asym_factor, mid))
                fitted_b = mid
                if abs(derivative_mid) <= 1e-12 or abs(hi - lo) <= 1e-10:
                    break
                if derivative_mid > 0.0:
                    hi = mid
                else:
                    lo = mid

    negative_log_likelihood = float(_pooled_negative_log_likelihood(cos_phi, asym_factor, fitted_b))
    derivative_at_fit = float(_pooled_nll_derivative(asym_factor, fitted_b))
    curvature_at_fit = float(_pooled_nll_second_derivative(asym_factor, fitted_b))

    sample_count = int(len(samples))
    approx_standard_error = None
    approx_ci = None
    if curvature_at_fit > 1e-12 and sample_count > 0:
        approx_standard_error = float(math.sqrt(1.0 / (sample_count * curvature_at_fit)))
        approx_ci = {
            "lower": float(max(solver_lower, fitted_b - 1.96 * approx_standard_error)),
            "upper": float(min(solver_upper, fitted_b + 1.96 * approx_standard_error)),
        }

    observed_interval = _observed_feasible_interval(asym_factor)
    domain_interval = _domain_feasible_interval(tilt_deg_values)
    observed_density_margin = float(np.min(1.0 + fitted_b * asym_factor))
    max_sin_alpha = float(np.max(np.abs(np.sin(np.radians(tilt_deg_values))))) if len(tilt_deg_values) else 0.0
    domain_density_margin = None if max_sin_alpha <= 1e-12 else float(1.0 - abs(fitted_b) * max_sin_alpha)

    fit = {
        "fitted_b": float(fitted_b),
        "negative_log_likelihood": negative_log_likelihood,
        "grid_fitted_b": grid_fitted_b,
        "grid_negative_log_likelihood": grid_objective,
        "root_grid_delta_b": float(fitted_b - grid_fitted_b),
        "root_grid_delta_negative_log_likelihood": float(negative_log_likelihood - grid_objective),
        "solution_method": solution_method,
        "sample_count": sample_count,
        "solver_interval": {
            "lower": solver_lower,
            "upper": solver_upper,
        },
        "observed_feasible_interval": {
            "lower": _finite_or_none(observed_interval[0]),
            "upper": _finite_or_none(observed_interval[1]),
        },
        "domain_feasible_interval": {
            "lower": _finite_or_none(domain_interval[0]),
            "upper": _finite_or_none(domain_interval[1]),
        },
        "derivative_at_fitted_b": derivative_at_fit,
        "second_derivative_at_fitted_b": curvature_at_fit,
        "approx_standard_error": approx_standard_error,
        "approx_95pct_confidence_interval": approx_ci,
        "observed_min_density_margin": observed_density_margin,
        "domain_min_density_margin": domain_density_margin,
    }
    return fit, b_grid, objective_grid, derivative_grid, curvature_grid


def _evaluate_objective_for_samples(samples: list[ExitSample], b_grid: np.ndarray, equal_run_weight: bool) -> np.ndarray:
    if not samples:
        raise ValueError("Need at least one exit sample to evaluate the objective.")
    if not equal_run_weight:
        cos_phi, asym_factor, _ = _sample_emission_components(samples)
        return np.asarray(_pooled_negative_log_likelihood(cos_phi, asym_factor, b_grid), dtype=np.float64)
    grouped = _group_samples_by_run(samples)

    objectives: list[np.ndarray] = []
    for run_samples in grouped.values():
        alpha = math.radians(run_samples[0].tilt_deg)
        sin_alpha = math.sin(alpha)
        phi = np.asarray([sample.phi_rad for sample in run_samples], dtype=np.float64)
        varphi = np.asarray([sample.varphi_rad for sample in run_samples], dtype=np.float64)
        base = np.cos(phi)
        asym_factor = sin_alpha * np.sin(phi) * np.cos(varphi)
        run_nll = np.asarray(_pooled_negative_log_likelihood(base, asym_factor, b_grid), dtype=np.float64)
        objectives.append(run_nll)

    stacked = np.vstack(objectives)
    if equal_run_weight:
        return stacked.mean(axis=0)
    weights = np.asarray([len(grouped[run_id]) for run_id in grouped], dtype=np.float64)
    return np.average(stacked, axis=0, weights=weights)


def _fit_b_from_samples(samples: list[ExitSample], grid_points: int = 2001, equal_run_weight: bool = True) -> tuple[float, float, np.ndarray, np.ndarray]:
    if not equal_run_weight:
        fit, b_grid, objective, _, _ = _fit_pooled_b_with_diagnostics(samples, grid_points=grid_points)
        return float(fit["fitted_b"]), float(fit["negative_log_likelihood"]), b_grid, objective
    tilt_values = np.asarray([sample.tilt_deg for sample in samples], dtype=np.float64)
    b_grid = _build_b_grid(tilt_values, grid_points)
    objective = _evaluate_objective_for_samples(samples, b_grid, equal_run_weight=equal_run_weight)
    best_idx = int(np.argmin(objective))
    return float(b_grid[best_idx]), float(objective[best_idx]), b_grid, objective


def _evaluate_b_alpha_sin_objective_for_samples(
    samples: list[ExitSample],
    b0: float,
    b1: float,
    *,
    equal_run_weight: bool,
) -> float:
    if not samples:
        raise ValueError("Need at least one exit sample to evaluate the B(alpha) objective.")
    grouped = _group_samples_by_run(samples)

    log_pi = math.log(math.pi)
    run_objectives: list[float] = []
    run_weights: list[float] = []
    for run_samples in grouped.values():
        alpha = math.radians(run_samples[0].tilt_deg)
        sin_alpha = math.sin(alpha)
        phi = np.asarray([sample.phi_rad for sample in run_samples], dtype=np.float64)
        varphi = np.asarray([sample.varphi_rad for sample in run_samples], dtype=np.float64)
        base = np.cos(phi)
        effective_b = b0 + b1 * sin_alpha
        asym_factor = sin_alpha * np.sin(phi) * np.cos(varphi)
        prob = base * (1.0 + asym_factor * effective_b) / math.pi
        prob = np.clip(prob, 1e-12, None)
        run_objectives.append(float(-np.mean(np.log(prob))))
        run_weights.append(float(len(run_samples)))

    if equal_run_weight:
        return float(np.mean(run_objectives))
    return float(np.average(np.asarray(run_objectives, dtype=np.float64), weights=np.asarray(run_weights, dtype=np.float64)))


def _build_run_objective_rows(samples: list[ExitSample], b_grid: np.ndarray) -> list[dict[str, Any]]:
    grouped = _group_samples_by_run(samples)
    out: list[dict[str, Any]] = []
    for run_id in sorted(grouped):
        run_samples = grouped[run_id]
        fit, _, objective_curve, _, _ = _fit_pooled_b_with_diagnostics(run_samples, b_grid=b_grid)
        tilt_deg = float(run_samples[0].tilt_deg)
        out.append(
            {
                "run_id": run_id,
                "tilt_deg": tilt_deg,
                "nominal_tilt_deg": float(run_samples[0].nominal_tilt_deg),
                "repeat_id": int(run_samples[0].repeat_id),
                "sample_count": len(run_samples),
                "sin_alpha": float(math.sin(math.radians(tilt_deg))),
                "fitted_b": float(fit["fitted_b"]),
                "grid_fitted_b": float(fit["grid_fitted_b"]),
                "root_grid_delta_b": float(fit["root_grid_delta_b"]),
                "negative_log_likelihood": float(fit["negative_log_likelihood"]),
                "objective_curve": objective_curve,
            }
        )
    return out


def _fit_b_alpha_sin_from_run_objectives(
    run_objective_rows: list[dict[str, Any]],
    b_grid: np.ndarray,
    *,
    equal_run_weight: bool,
    search_points: int = 401,
    refine_rounds: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not run_objective_rows:
        raise ValueError("Need at least one run objective row to fit B(alpha).")
    if len(b_grid) < 2:
        raise ValueError("B grid must contain at least two points.")

    sin_alpha = np.asarray([row["sin_alpha"] for row in run_objective_rows], dtype=np.float64)
    sample_weights = np.asarray([row["sample_count"] for row in run_objective_rows], dtype=np.float64)
    weights = np.ones_like(sample_weights) / len(sample_weights) if equal_run_weight else sample_weights / sample_weights.sum()
    objective_curves = [np.asarray(row["objective_curve"], dtype=np.float64) for row in run_objective_rows]
    b_min = float(b_grid[0])
    b_max = float(b_grid[-1])
    step = float(b_grid[1] - b_grid[0])

    max_sin_alpha = max(float(np.max(np.abs(sin_alpha))), 1e-12)
    fitted_bs = np.asarray([row["fitted_b"] for row in run_objective_rows], dtype=np.float64)
    span = max(float(fitted_bs.max() - fitted_bs.min()), 0.25)
    b0_lo = b_min
    b0_hi = b_max
    b1_abs = min(8.0, max(2.0, 3.0 * max(abs(b_min), abs(b_max)) / max_sin_alpha, 2.5 * span / max_sin_alpha))
    b1_lo = -b1_abs
    b1_hi = b1_abs

    best_b0 = 0.0
    best_b1 = 0.0
    best_objective = float("inf")

    for _ in range(max(1, int(refine_rounds))):
        b0_grid = np.linspace(b0_lo, b0_hi, max(51, int(search_points)), dtype=np.float64)
        b1_grid = np.linspace(b1_lo, b1_hi, max(51, int(search_points)), dtype=np.float64)
        surface = np.zeros((len(b0_grid), len(b1_grid)), dtype=np.float64)
        valid = np.ones_like(surface, dtype=bool)

        for idx, row in enumerate(run_objective_rows):
            curve = objective_curves[idx]
            effective_b = b0_grid[:, np.newaxis] + b1_grid[np.newaxis, :] * row["sin_alpha"]
            run_valid = (effective_b >= b_min) & (effective_b <= b_max)
            valid &= run_valid

            idx_float = (np.clip(effective_b, b_min, b_max) - b_min) / step
            idx0 = np.floor(idx_float).astype(np.int64)
            idx0 = np.clip(idx0, 0, len(b_grid) - 2)
            frac = np.clip(idx_float - idx0, 0.0, 1.0)
            interp = curve[idx0] * (1.0 - frac) + curve[idx0 + 1] * frac
            surface += weights[idx] * interp

        surface = np.where(valid, surface, np.inf)
        best_index = np.unravel_index(int(np.argmin(surface)), surface.shape)
        best_b0 = float(b0_grid[best_index[0]])
        best_b1 = float(b1_grid[best_index[1]])
        best_objective = float(surface[best_index])

        if len(b0_grid) > 1:
            b0_half_width = max(float(b0_grid[1] - b0_grid[0]) * 4.0, 1e-3)
            b1_half_width = max(float(b1_grid[1] - b1_grid[0]) * 4.0, 1e-3)
            b0_lo = max(float(b0_grid[0]), best_b0 - b0_half_width)
            b0_hi = min(float(b0_grid[-1]), best_b0 + b0_half_width)
            b1_lo = max(float(b1_grid[0]), best_b1 - b1_half_width)
            b1_hi = min(float(b1_grid[-1]), best_b1 + b1_half_width)

    prediction_rows: list[dict[str, Any]] = []
    for row in run_objective_rows:
        predicted_b = best_b0 + best_b1 * row["sin_alpha"]
        predicted_objective = float(np.interp(predicted_b, b_grid, row["objective_curve"]))
        prediction_rows.append(
            {
                "run_id": row["run_id"],
                "nominal_tilt_deg": row["nominal_tilt_deg"],
                "measured_tilt_deg": row["tilt_deg"],
                "repeat_id": row["repeat_id"],
                "sample_count": row["sample_count"],
                "observed_fitted_b": row["fitted_b"],
                "predicted_b": float(predicted_b),
                "residual_b": float(row["fitted_b"] - predicted_b),
                "observed_negative_log_likelihood": row["negative_log_likelihood"],
                "predicted_negative_log_likelihood": predicted_objective,
            }
        )

    fit = {
        "b0": best_b0,
        "b1": best_b1,
        "negative_log_likelihood": best_objective,
        "search_points": max(51, int(search_points)),
        "refine_rounds": max(1, int(refine_rounds)),
    }
    return fit, prediction_rows


def _fit_b_alpha_sin_from_samples(
    samples: list[ExitSample],
    *,
    scalar_grid_points: int = 2001,
    law_search_points: int = 401,
    equal_run_weight: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    tilt_values = np.asarray([sample.tilt_deg for sample in samples], dtype=np.float64)
    b_grid = _build_b_grid(tilt_values, scalar_grid_points)
    run_objective_rows = _build_run_objective_rows(samples, b_grid)
    fit, prediction_rows = _fit_b_alpha_sin_from_run_objectives(
        run_objective_rows,
        b_grid,
        equal_run_weight=equal_run_weight,
        search_points=law_search_points,
    )
    fit["negative_log_likelihood"] = _evaluate_b_alpha_sin_objective_for_samples(
        samples,
        fit["b0"],
        fit["b1"],
        equal_run_weight=equal_run_weight,
    )
    return fit, prediction_rows, b_grid


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    sqrt_w = np.sqrt(np.clip(w_arr, 1e-12, None))
    xw = x_arr * sqrt_w[:, np.newaxis]
    yw = y_arr * sqrt_w
    coeffs, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    return np.asarray(coeffs, dtype=np.float64)


def _evaluate_parametric_curve_objective(
    run_objective_rows: list[dict[str, Any]],
    b_grid: np.ndarray,
    coeffs: np.ndarray,
    *,
    equal_run_weight: bool,
) -> float:
    coeff_arr = np.asarray(coeffs, dtype=np.float64)
    b_min = float(b_grid[0])
    b_max = float(b_grid[-1])
    objectives: list[float] = []
    weights: list[float] = []
    for row in run_objective_rows:
        sin_alpha = float(row["sin_alpha"])
        predicted_b = float(coeff_arr[0] + coeff_arr[1] * sin_alpha + coeff_arr[2] * sin_alpha * sin_alpha)
        if predicted_b < b_min or predicted_b > b_max:
            return float("inf")
        objectives.append(float(np.interp(predicted_b, b_grid, row["objective_curve"])))
        weights.append(1.0 if equal_run_weight else float(row["sample_count"]))
    if equal_run_weight:
        return float(np.mean(objectives))
    return float(np.average(np.asarray(objectives, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))


def _refine_parametric_coeffs_pattern_search(
    run_objective_rows: list[dict[str, Any]],
    b_grid: np.ndarray,
    initial_coeffs: np.ndarray,
    *,
    equal_run_weight: bool,
    initial_steps: np.ndarray,
    max_iterations: int = 80,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, float]:
    coeffs = np.asarray(initial_coeffs, dtype=np.float64).copy()
    steps = np.asarray(initial_steps, dtype=np.float64).copy()
    best_objective = _evaluate_parametric_curve_objective(
        run_objective_rows,
        b_grid,
        coeffs,
        equal_run_weight=equal_run_weight,
    )
    if not np.isfinite(best_objective):
        return coeffs, best_objective

    for _ in range(max_iterations):
        improved = False
        for dim in range(len(coeffs)):
            for sign in (-1.0, 1.0):
                candidate = coeffs.copy()
                candidate[dim] += sign * steps[dim]
                candidate_objective = _evaluate_parametric_curve_objective(
                    run_objective_rows,
                    b_grid,
                    candidate,
                    equal_run_weight=equal_run_weight,
                )
                if candidate_objective + 1e-12 < best_objective:
                    coeffs = candidate
                    best_objective = candidate_objective
                    improved = True
        if not improved:
            steps *= 0.5
            if float(np.max(steps)) < tolerance:
                break
    return coeffs, best_objective


def _evaluate_b_alpha_quadratic_objective_for_samples(
    samples: list[ExitSample],
    b0: float,
    b1: float,
    b2: float,
    *,
    equal_run_weight: bool,
) -> float:
    if not samples:
        raise ValueError("Need at least one exit sample to evaluate the quadratic B(alpha) objective.")
    grouped = _group_samples_by_run(samples)

    run_objectives: list[float] = []
    run_weights: list[float] = []
    for run_samples in grouped.values():
        alpha = math.radians(run_samples[0].tilt_deg)
        sin_alpha = math.sin(alpha)
        phi = np.asarray([sample.phi_rad for sample in run_samples], dtype=np.float64)
        varphi = np.asarray([sample.varphi_rad for sample in run_samples], dtype=np.float64)
        base = np.cos(phi)
        effective_b = b0 + b1 * sin_alpha + b2 * sin_alpha * sin_alpha
        asym_factor = sin_alpha * np.sin(phi) * np.cos(varphi)
        prob = base * (1.0 + asym_factor * effective_b) / math.pi
        prob = np.clip(prob, 1e-12, None)
        run_objectives.append(float(-np.mean(np.log(prob))))
        run_weights.append(float(len(run_samples)))

    if equal_run_weight:
        return float(np.mean(run_objectives))
    return float(np.average(np.asarray(run_objectives, dtype=np.float64), weights=np.asarray(run_weights, dtype=np.float64)))


def _fit_b_alpha_quadratic_from_samples(
    samples: list[ExitSample],
    *,
    scalar_grid_points: int = 2001,
    equal_run_weight: bool,
    initial_sin_fit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    tilt_values = np.asarray([sample.tilt_deg for sample in samples], dtype=np.float64)
    b_grid = _build_b_grid(tilt_values, scalar_grid_points)
    run_objective_rows = _build_run_objective_rows(samples, b_grid)
    sin_alpha = np.asarray([row["sin_alpha"] for row in run_objective_rows], dtype=np.float64)
    observed_b = np.asarray([row["fitted_b"] for row in run_objective_rows], dtype=np.float64)
    sample_weights = np.asarray([row["sample_count"] for row in run_objective_rows], dtype=np.float64)
    fit_weights = np.ones_like(sample_weights) if equal_run_weight else sample_weights
    design = np.column_stack([np.ones_like(sin_alpha), sin_alpha, sin_alpha * sin_alpha])
    lsq_coeffs = _weighted_lstsq(design, observed_b, fit_weights)

    if initial_sin_fit is None:
        initial_sin_fit, _, _ = _fit_b_alpha_sin_from_samples(
            samples,
            scalar_grid_points=scalar_grid_points,
            law_search_points=201,
            equal_run_weight=equal_run_weight,
        )

    b_span = max(float(b_grid[-1] - b_grid[0]), 0.5)
    starts = [
        np.asarray([float(np.mean(observed_b)), 0.0, 0.0], dtype=np.float64),
        np.asarray([initial_sin_fit["b0"], initial_sin_fit["b1"], 0.0], dtype=np.float64),
        lsq_coeffs,
    ]
    initial_steps = np.asarray([0.25 * b_span, 0.5 * b_span, 0.5 * b_span], dtype=np.float64)

    best_coeffs = starts[0]
    best_curve_objective = float("inf")
    for start in starts:
        coeffs, curve_objective = _refine_parametric_coeffs_pattern_search(
            run_objective_rows,
            b_grid,
            start,
            equal_run_weight=equal_run_weight,
            initial_steps=initial_steps,
        )
        if curve_objective < best_curve_objective:
            best_coeffs = coeffs
            best_curve_objective = curve_objective

    b0, b1, b2 = [float(value) for value in best_coeffs]
    prediction_rows: list[dict[str, Any]] = []
    for row in run_objective_rows:
        s = float(row["sin_alpha"])
        predicted_b = float(b0 + b1 * s + b2 * s * s)
        predicted_objective = float(np.interp(predicted_b, b_grid, row["objective_curve"]))
        prediction_rows.append(
            {
                "run_id": row["run_id"],
                "nominal_tilt_deg": row["nominal_tilt_deg"],
                "measured_tilt_deg": row["tilt_deg"],
                "repeat_id": row["repeat_id"],
                "sample_count": row["sample_count"],
                "observed_fitted_b": row["fitted_b"],
                "predicted_b": predicted_b,
                "residual_b": float(row["fitted_b"] - predicted_b),
                "observed_negative_log_likelihood": row["negative_log_likelihood"],
                "predicted_negative_log_likelihood": predicted_objective,
            }
        )

    fit = {
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "negative_log_likelihood": _evaluate_b_alpha_quadratic_objective_for_samples(
            samples,
            b0,
            b1,
            b2,
            equal_run_weight=equal_run_weight,
        ),
        "curve_search_negative_log_likelihood": best_curve_objective,
        "formula": "B(alpha_deg) = b0 + b1 * sin(alpha_deg) + b2 * sin(alpha_deg)^2",
    }
    return fit, prediction_rows, b_grid


def _fit_per_run(samples: list[ExitSample], grid_points: int) -> list[dict[str, Any]]:
    grouped = _group_samples_by_run(samples)
    out: list[dict[str, Any]] = []
    for run_id in sorted(grouped):
        run_samples = grouped[run_id]
        fit, _, _, _, _ = _fit_pooled_b_with_diagnostics(run_samples, grid_points=grid_points)
        out.append(
            {
                "run_id": run_id,
                "tilt_deg": run_samples[0].tilt_deg,
                "nominal_tilt_deg": run_samples[0].nominal_tilt_deg,
                "repeat_id": run_samples[0].repeat_id,
                "sample_count": len(run_samples),
                "fitted_b": float(fit["fitted_b"]),
                "grid_fitted_b": float(fit["grid_fitted_b"]),
                "root_grid_delta_b": float(fit["root_grid_delta_b"]),
                "negative_log_likelihood": float(fit["negative_log_likelihood"]),
                "approx_standard_error": fit["approx_standard_error"],
            }
        )
    return out


def _fit_per_tilt(samples: list[ExitSample], grid_points: int) -> list[dict[str, Any]]:
    grouped: dict[float, list[ExitSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.nominal_tilt_deg, []).append(sample)
    out: list[dict[str, Any]] = []
    for nominal_tilt_deg in sorted(grouped):
        tilt_samples = grouped[nominal_tilt_deg]
        fit, _, _, _, _ = _fit_pooled_b_with_diagnostics(tilt_samples, grid_points=grid_points)
        out.append(
            {
                "nominal_tilt_deg": nominal_tilt_deg,
                "measured_tilt_deg_mean": float(np.mean([sample.tilt_deg for sample in tilt_samples])),
                "run_count": len({sample.run_id for sample in tilt_samples}),
                "sample_count": len(tilt_samples),
                "fitted_b": float(fit["fitted_b"]),
                "grid_fitted_b": float(fit["grid_fitted_b"]),
                "root_grid_delta_b": float(fit["root_grid_delta_b"]),
                "negative_log_likelihood": float(fit["negative_log_likelihood"]),
                "approx_standard_error": fit["approx_standard_error"],
            }
        )
    return out


def _leave_one_repeat_out(samples: list[ExitSample], grid_points: int) -> list[BFoldResult]:
    repeat_ids = sorted({sample.repeat_id for sample in samples})
    if len(repeat_ids) < 2:
        return []
    folds: list[BFoldResult] = []
    for repeat_id in repeat_ids:
        train = [sample for sample in samples if sample.repeat_id != repeat_id]
        test = [sample for sample in samples if sample.repeat_id == repeat_id]
        if not train or not test:
            continue
        fit, _, _, _, _ = _fit_pooled_b_with_diagnostics(train, grid_points=grid_points)
        test_grid = np.asarray([float(fit["fitted_b"])], dtype=np.float64)
        test_objective = float(_evaluate_objective_for_samples(test, test_grid, equal_run_weight=False)[0])
        folds.append(
            BFoldResult(
                holdout_repeat_id=repeat_id,
                fitted_b=float(fit["fitted_b"]),
                train_objective=float(fit["negative_log_likelihood"]),
                test_negative_log_likelihood=test_objective,
                train_run_count=len({sample.run_id for sample in train}),
                test_run_count=len({sample.run_id for sample in test}),
                train_sample_count=len(train),
                test_sample_count=len(test),
            )
        )
    return folds


def _leave_one_repeat_out_b_alpha_sin(
    samples: list[ExitSample],
    *,
    scalar_grid_points: int,
    law_search_points: int,
) -> list[BAlphaFoldResult]:
    repeat_ids = sorted({sample.repeat_id for sample in samples})
    if len(repeat_ids) < 2:
        return []
    folds: list[BAlphaFoldResult] = []
    for repeat_id in repeat_ids:
        train = [sample for sample in samples if sample.repeat_id != repeat_id]
        test = [sample for sample in samples if sample.repeat_id == repeat_id]
        if not train or not test:
            continue
        fit, _, _ = _fit_b_alpha_sin_from_samples(
            train,
            scalar_grid_points=scalar_grid_points,
            law_search_points=law_search_points,
            equal_run_weight=False,
        )
        test_objective = _evaluate_b_alpha_sin_objective_for_samples(
            test,
            fit["b0"],
            fit["b1"],
            equal_run_weight=False,
        )
        folds.append(
            BAlphaFoldResult(
                holdout_repeat_id=repeat_id,
                fitted_b0=fit["b0"],
                fitted_b1=fit["b1"],
                train_objective=fit["negative_log_likelihood"],
                test_negative_log_likelihood=test_objective,
                train_run_count=len({sample.run_id for sample in train}),
                test_run_count=len({sample.run_id for sample in test}),
                train_sample_count=len(train),
                test_sample_count=len(test),
            )
        )
    return folds


def _fit_b_alpha_sin_per_tilt(
    per_tilt_rows: list[dict[str, Any]],
    fit: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in per_tilt_rows:
        sin_alpha = math.sin(math.radians(float(row["measured_tilt_deg_mean"])))
        predicted_b = float(fit["b0"] + fit["b1"] * sin_alpha)
        out.append(
            {
                "nominal_tilt_deg": row["nominal_tilt_deg"],
                "measured_tilt_deg_mean": row["measured_tilt_deg_mean"],
                "observed_fitted_b": row["fitted_b"],
                "predicted_b": predicted_b,
                "residual_b": float(row["fitted_b"] - predicted_b),
                "sample_count": row["sample_count"],
                "run_count": row["run_count"],
            }
        )
    return out


def _leave_one_repeat_out_b_alpha_quadratic(
    samples: list[ExitSample],
    *,
    scalar_grid_points: int,
) -> list[dict[str, Any]]:
    repeat_ids = sorted({sample.repeat_id for sample in samples})
    if len(repeat_ids) < 2:
        return []
    folds: list[dict[str, Any]] = []
    for repeat_id in repeat_ids:
        train = [sample for sample in samples if sample.repeat_id != repeat_id]
        test = [sample for sample in samples if sample.repeat_id == repeat_id]
        if not train or not test:
            continue
        sin_fit, _, _ = _fit_b_alpha_sin_from_samples(
            train,
            scalar_grid_points=scalar_grid_points,
            law_search_points=201,
            equal_run_weight=False,
        )
        fit, _, _ = _fit_b_alpha_quadratic_from_samples(
            train,
            scalar_grid_points=scalar_grid_points,
            equal_run_weight=False,
            initial_sin_fit=sin_fit,
        )
        test_objective = _evaluate_b_alpha_quadratic_objective_for_samples(
            test,
            fit["b0"],
            fit["b1"],
            fit["b2"],
            equal_run_weight=False,
        )
        folds.append(
            {
                "holdout_repeat_id": repeat_id,
                "fitted_b0": fit["b0"],
                "fitted_b1": fit["b1"],
                "fitted_b2": fit["b2"],
                "train_objective": fit["negative_log_likelihood"],
                "test_negative_log_likelihood": test_objective,
                "train_run_count": len({sample.run_id for sample in train}),
                "test_run_count": len({sample.run_id for sample in test}),
                "train_sample_count": len(train),
                "test_sample_count": len(test),
            }
        )
    return folds


def _fit_b_alpha_quadratic_per_tilt(
    per_tilt_rows: list[dict[str, Any]],
    fit: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in per_tilt_rows:
        s = math.sin(math.radians(float(row["measured_tilt_deg_mean"])))
        predicted_b = float(fit["b0"] + fit["b1"] * s + fit["b2"] * s * s)
        out.append(
            {
                "nominal_tilt_deg": row["nominal_tilt_deg"],
                "measured_tilt_deg_mean": row["measured_tilt_deg_mean"],
                "observed_fitted_b": row["fitted_b"],
                "predicted_b": predicted_b,
                "residual_b": float(row["fitted_b"] - predicted_b),
                "sample_count": row["sample_count"],
                "run_count": row["run_count"],
            }
        )
    return out


def _residual_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    residuals = np.asarray([abs(float(row["residual_b"])) for row in rows], dtype=np.float64)
    return {
        "mean_abs": float(np.mean(residuals)),
        "max_abs": float(np.max(residuals)),
    }


def _summarize_parametric_model(
    name: str,
    fit: dict[str, Any],
    fold_section: dict[str, Any],
    per_tilt_rows: list[dict[str, Any]],
    per_run_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tilt_metrics = _residual_metrics(per_tilt_rows)
    run_metrics = _residual_metrics(per_run_rows)
    out = {
        "name": name,
        "fit": fit,
        "cross_validation": fold_section,
        "per_tilt_metrics": tilt_metrics,
        "per_run_metrics": run_metrics,
    }
    if "b2" in fit:
        out["formula"] = f"B(alpha) = {fit['b0']:.6f} + {fit['b1']:.6f} sin(alpha) + {fit['b2']:.6f} sin^2(alpha)"
    else:
        out["formula"] = f"B(alpha) = {fit['b0']:.6f} + {fit['b1']:.6f} sin(alpha)"
    return out


def _get_primary_constant_fit(report: dict[str, Any]) -> dict[str, Any]:
    fit = report["fit"]
    return fit.get("global_pooled") or fit.get("global_sample_weighted") or fit["global_equal_run"]


def _get_primary_parametric_fit(report: dict[str, Any], key: str) -> dict[str, Any]:
    fit = report["fit"]
    return fit.get(f"{key}_sample_weighted") or fit[f"{key}_equal_run"]


def _rr_density_mass(alpha_rad: float, b_value: float, steps_phi: int = 181, steps_varphi: int = 361) -> float:
    phi = np.linspace(0.0, math.pi / 2.0, steps_phi, dtype=np.float64)
    varphi = np.linspace(-math.pi, math.pi, steps_varphi, dtype=np.float64)
    p_grid, v_grid = np.meshgrid(phi, varphi, indexing="ij")
    density = np.cos(p_grid) * (1.0 + b_value * math.sin(alpha_rad) * np.sin(p_grid) * np.cos(v_grid)) / math.pi
    weighted = density * np.sin(p_grid)
    integral_varphi = np.trapezoid(weighted, varphi, axis=1)
    return float(np.trapezoid(integral_varphi, phi))


def _build_rr_normalization_check(tilt_deg_values: np.ndarray, b_value: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for tilt_deg in sorted({round(float(value), 6) for value in tilt_deg_values}):
        mass = _rr_density_mass(math.radians(tilt_deg), b_value)
        rows.append(
            {
                "measured_tilt_deg": float(tilt_deg),
                "b": float(b_value),
                "mass": mass,
                "abs_error": abs(mass - 1.0),
            }
        )
    return {
        "model": "constant_b",
        "rows": rows,
        "max_abs_error": None if not rows else float(max(row["abs_error"] for row in rows)),
    }


def _choose_recommended_model(report: dict[str, Any]) -> tuple[str, str, str | None]:
    return ("lambertian", "Legacy likelihood rankings do not establish a qualified B model. Use B=0 pending surface selection and model adequacy checks.",
            "Per-tilt rows below are diagnostic constant-fit rows, not the selected renderer model.")


def _rank_legacy_models(report: dict[str, Any]) -> tuple[str, str, str | None]:
    constant_cv = report["fit"]["leave_one_repeat_out"]["mean_test_negative_log_likelihood"]
    sin_section = _summarize_parametric_model(
        "b_alpha_sin",
        _get_primary_parametric_fit(report, "b_alpha_sin"),
        report["fit"]["b_alpha_sin_leave_one_repeat_out"],
        report["fit"]["b_alpha_sin_per_tilt"],
        report["fit"]["b_alpha_sin_per_run"],
    )
    quad_section = _summarize_parametric_model(
        "b_alpha_quadratic",
        _get_primary_parametric_fit(report, "b_alpha_quadratic"),
        report["fit"]["b_alpha_quadratic_leave_one_repeat_out"],
        report["fit"]["b_alpha_quadratic_per_tilt"],
        report["fit"]["b_alpha_quadratic_per_run"],
    )

    sin_cv = sin_section["cross_validation"]["mean_test_negative_log_likelihood"]
    quad_cv = quad_section["cross_validation"]["mean_test_negative_log_likelihood"]
    sin_mae = sin_section["per_tilt_metrics"]["mean_abs"]
    quad_mae = quad_section["per_tilt_metrics"]["mean_abs"]

    if quad_cv is not None and sin_cv is not None:
        quad_cv_gain = float(sin_cv - quad_cv)
        quad_mae_gain = float(sin_mae - quad_mae)
        if quad_cv_gain > 1e-3 and quad_mae_gain > 0.05:
            caution = None
            if quad_mae > 0.2:
                caution = (
                    "The quadratic law is the best of the tested simple laws, but it still leaves noticeable per-tilt residuals "
                    f"(mean abs per-tilt residual {quad_mae:.6f}, max {quad_section['per_tilt_metrics']['max_abs']:.6f})."
                )
            return (
                "b_alpha_quadratic",
                f"The quadratic law reduced leave-one-repeat-out NLL by {quad_cv_gain:.6f} versus the simple sin(alpha) law and reduced mean abs per-tilt residual by {quad_mae_gain:.6f}.",
                caution,
            )

    if sin_cv is not None and constant_cv is not None and sin_cv + 1e-3 < constant_cv:
        caution = None
        if sin_mae > 0.2:
            caution = (
                "The simple sin(alpha) law improves likelihood, but it does not track the per-tilt fitted B values closely "
                f"(mean abs per-tilt residual {sin_mae:.6f}, max {sin_section['per_tilt_metrics']['max_abs']:.6f})."
            )
        return (
            "b_alpha_sin",
            f"B(alpha)=B0+B1 sin(alpha) reduced pooled holdout NLL by {constant_cv - sin_cv:.6f} relative to the constant model.",
            caution,
        )

    return ("constant_b", "A single constant B remains adequate relative to the tested tilt laws.", None)


def _write_per_tilt_comparison_plot(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    per_tilt_rows = report["fit"]["per_tilt"]
    x = np.asarray([row["measured_tilt_deg_mean"] for row in per_tilt_rows], dtype=np.float64)
    observed = np.asarray([row["fitted_b"] for row in per_tilt_rows], dtype=np.float64)
    constant_line = np.full_like(x, float(_get_primary_constant_fit(report)["fitted_b"]))
    sin_pred = np.asarray([row["predicted_b"] for row in report["fit"]["b_alpha_sin_per_tilt"]], dtype=np.float64)
    quad_pred = np.asarray([row["predicted_b"] for row in report["fit"]["b_alpha_quadratic_per_tilt"]], dtype=np.float64)

    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x[order], observed[order], "o-", label="Observed per-tilt fitted B", linewidth=2.0)
    ax.plot(x[order], constant_line[order], "--", label="Constant B", linewidth=1.5)
    ax.plot(x[order], sin_pred[order], "s-", label="B0 + B1 sin(alpha)", linewidth=1.8)
    ax.plot(x[order], quad_pred[order], "^-", label="B0 + B1 sin(alpha) + B2 sin^2(alpha)", linewidth=1.8)
    ax.set_xlabel("Measured tilt alpha (deg)")
    ax.set_ylabel("B")
    ax.set_title("CASINO-derived B by tilt")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_objective_diagnostics_plot(path: Path, objective_rows: list[dict[str, Any]], fitted_b: float) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    if not objective_rows:
        return {"status": "skipped", "reason": "no objective rows", "path": None}

    b = np.asarray([float(row["b"]) for row in objective_rows], dtype=np.float64)
    nll = np.asarray([float(row["negative_log_likelihood"]) for row in objective_rows], dtype=np.float64)
    derivative = np.asarray([float(row["derivative"]) for row in objective_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    axes[0].plot(b, nll, linewidth=2.0)
    axes[0].axvline(fitted_b, linestyle="--", color="tab:red", linewidth=1.5)
    axes[0].set_ylabel("Negative log-likelihood")
    axes[0].set_title("Pooled RR fit objective")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(b, derivative, linewidth=2.0)
    axes[1].axhline(0.0, linestyle=":", color="black", linewidth=1.0)
    axes[1].axvline(fitted_b, linestyle="--", color="tab:red", linewidth=1.5)
    axes[1].set_xlabel("B")
    axes[1].set_ylabel("dNLL / dB")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_run_trend_plot(path: Path, per_run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    if not per_run_rows:
        return {"status": "skipped", "reason": "no per-run rows", "path": None}

    measured_tilt = np.asarray([float(row["tilt_deg"]) for row in per_run_rows], dtype=np.float64)
    sin_alpha = np.sin(np.radians(measured_tilt))
    fitted_b = np.asarray([float(row["fitted_b"]) for row in per_run_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].scatter(measured_tilt, fitted_b, s=36)
    axes[0].set_xlabel("Measured tilt alpha (deg)")
    axes[0].set_ylabel("Per-run fitted B")
    axes[0].set_title("B_run vs measured tilt")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(sin_alpha, fitted_b, s=36)
    axes[1].set_xlabel("sin(alpha)")
    axes[1].set_ylabel("Per-run fitted B")
    axes[1].set_title("B_run vs sin(alpha)")
    axes[1].grid(True, alpha=0.3)

    if len(per_run_rows) >= 2 and not np.allclose(measured_tilt, measured_tilt[0]):
        tilt_coeffs = np.polyfit(measured_tilt, fitted_b, 1)
        x_fit = np.linspace(float(measured_tilt.min()), float(measured_tilt.max()), 128, dtype=np.float64)
        axes[0].plot(x_fit, tilt_coeffs[0] * x_fit + tilt_coeffs[1], "--", color="tab:red", linewidth=1.5)
    if len(per_run_rows) >= 2 and not np.allclose(sin_alpha, sin_alpha[0]):
        sin_coeffs = np.polyfit(sin_alpha, fitted_b, 1)
        s_fit = np.linspace(float(sin_alpha.min()), float(sin_alpha.max()), 128, dtype=np.float64)
        axes[1].plot(s_fit, sin_coeffs[0] * s_fit + sin_coeffs[1], "--", color="tab:red", linewidth=1.5)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_azimuth_harmonics_plot(path: Path, diagnostics: dict[str, Any]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    per_tilt = diagnostics["per_tilt"]
    if not per_tilt:
        return {"status": "skipped", "reason": "no per-tilt harmonic rows", "path": None}
    rows = sorted(per_tilt, key=lambda row: float(row["nominal_tilt_deg"]))
    tilt = np.asarray([float(row["nominal_tilt_deg"]) for row in rows], dtype=np.float64)
    cosine = np.asarray([float(row["cos_first"]) for row in rows], dtype=np.float64)
    sine = np.asarray([float(row["sin_first"]) for row in rows], dtype=np.float64)
    first = np.asarray([float(row["first_amplitude"]) for row in rows], dtype=np.float64)
    second = np.asarray([float(row["second_amplitude"]) for row in rows], dtype=np.float64)
    cosine_error = np.asarray(
        [0.0 if row["repeat_cos_first_standard_error"] is None else float(row["repeat_cos_first_standard_error"]) for row in rows],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 11.0), sharex=True)
    axes[0].errorbar(tilt, cosine, yerr=cosine_error, marker="o", capsize=3, label=r"$C_1=2\langle\cos\varphi\rangle$")
    axes[0].plot(tilt, sine, "s-", label=r"$S_1=2\langle\sin\varphi\rangle$")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("First harmonic coefficient")
    axes[0].set_title("CASINO exit-azimuth harmonics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(tilt, first, "o-", label=r"$A_1=\sqrt{C_1^2+S_1^2}$")
    axes[1].plot(tilt, second, "s-", label=r"$A_2=\sqrt{C_2^2+S_2^2}$")
    axes[1].set_ylabel("Harmonic amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    band_rows = [
        row
        for row in diagnostics["per_polar_band"]
        if row["aggregation_level"] == "tilt" and row["diagnostic_b_measured_tilt"] is not None
    ]
    for polar_min, polar_max in _POLAR_BANDS_DEG:
        selected = sorted(
            [
                row
                for row in band_rows
                if float(row["polar_min_deg"]) == polar_min and float(row["polar_max_deg"]) == polar_max
            ],
            key=lambda row: float(row["nominal_tilt_deg"]),
        )
        if selected:
            axes[2].plot(
                [float(row["nominal_tilt_deg"]) for row in selected],
                [float(row["diagnostic_b_measured_tilt"]) for row in selected],
                "o-",
                label=f"{polar_min:.0f} to {polar_max:.0f} deg polar",
            )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Nominal surface tilt alpha (deg)")
    axes[2].set_ylabel("Diagnostic B from $C_1$")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_azimuth_histograms_plot(path: Path, samples: list[ExitSample]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    available = sorted({float(sample.nominal_tilt_deg) for sample in samples})
    if not available:
        return {"status": "skipped", "reason": "no exit samples", "path": None}
    targets = [0.0, 15.0, 30.0, 45.0, 60.0, 70.0]
    selected_tilts: list[float] = []
    for target in targets:
        nearest = min(available, key=lambda value: abs(value - target))
        if nearest not in selected_tilts:
            selected_tilts.append(nearest)
    selected_tilts = selected_tilts[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=True, sharey=True)
    edges = np.linspace(-math.pi, math.pi, 37, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for ax, nominal_tilt in zip(axes.flat, selected_tilts):
        tilt_samples = [sample for sample in samples if float(sample.nominal_tilt_deg) == nominal_tilt]
        varphi = np.asarray([sample.varphi_rad for sample in tilt_samples], dtype=np.float64)
        stats = _harmonic_statistics(tilt_samples)
        ax.hist(varphi, bins=edges, density=True, alpha=0.55, label="CASINO")
        fitted_density = (
            1.0
            + float(stats["cos_first"]) * np.cos(centres)
            + float(stats["sin_first"]) * np.sin(centres)
        ) / (2.0 * math.pi)
        ax.plot(centres, fitted_density, color="tab:red", linewidth=2.0, label="First harmonic")
        ax.set_title(
            f"alpha={nominal_tilt:.0f} deg\nC1={stats['cos_first']:.3f}, S1={stats['sin_first']:.3f}"
        )
        ax.grid(True, alpha=0.25)
    for ax in axes.flat[len(selected_tilts):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Exit azimuth varphi (rad)")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Probability density")
    first_axis = next(ax for ax in axes.flat if ax.get_visible())
    first_axis.legend(fontsize=8)
    fig.suptitle("Exit-azimuth oscillation and first-harmonic fit", y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _write_coordinate_frame_plot(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    evaluable = [row for row in rows if row["continuity_evaluable"]]
    if not rows:
        return {"status": "skipped", "reason": "no coordinate-frame rows", "path": None}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for repeat_id in sorted({int(row["repeat_id"]) for row in evaluable}):
        selected = sorted(
            [row for row in evaluable if int(row["repeat_id"]) == repeat_id],
            key=lambda row: float(row["nominal_tilt_deg"]),
        )
        axes[0].plot(
            [float(row["nominal_tilt_deg"]) for row in selected],
            [float(row["tangent_1_dot_previous"]) for row in selected],
            "o-",
            label=f"t1, repeat {repeat_id}",
        )
        axes[0].plot(
            [float(row["nominal_tilt_deg"]) for row in selected],
            [float(row["tangent_2_dot_previous"]) for row in selected],
            "s--",
            label=f"t2, repeat {repeat_id}",
        )
    axes[0].axhline(0.0, color="tab:red", linestyle=":", linewidth=1.2, label="flip threshold")
    axes[0].set_xlabel("Nominal tilt alpha (deg)")
    axes[0].set_ylabel("Dot product with previous-tilt tangent")
    axes[0].set_title("Azimuth-frame continuity")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    index = np.arange(len(rows))
    norm_error = np.asarray(
        [max(row["normal_norm_error"], row["tangent_1_norm_error"], row["tangent_2_norm_error"]) for row in rows],
        dtype=np.float64,
    )
    orth_error = np.asarray(
        [max(row["normal_dot_tangent_1_abs"], row["normal_dot_tangent_2_abs"], row["tangent_1_dot_tangent_2_abs"]) for row in rows],
        dtype=np.float64,
    )
    axes[1].semilogy(index, np.maximum(norm_error, 1e-18), "o", label="unit-vector error")
    axes[1].semilogy(index, np.maximum(orth_error, 1e-18), "s", label="orthogonality error")
    axes[1].set_xlabel("Run index (repeat then tilt)")
    axes[1].set_ylabel("Absolute frame error")
    axes[1].set_title("Coordinate-frame numerical checks")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "reason": None, "path": str(path)}


def _build_gui_summary(report: dict[str, Any]) -> dict[str, Any]:
    constant_fit = _get_primary_constant_fit(report)
    constant_cv = report["fit"]["leave_one_repeat_out"]
    constant_b = float(constant_fit["fitted_b"])
    constant_per_tilt = [
        {
            "nominal_tilt_deg": row["nominal_tilt_deg"],
            "measured_tilt_deg_mean": row["measured_tilt_deg_mean"],
            "observed_fitted_b": row["fitted_b"],
            "predicted_b": constant_b,
            "residual_b": float(row["fitted_b"] - constant_b),
            "sample_count": row["sample_count"],
            "run_count": row["run_count"],
        }
        for row in report["fit"]["per_tilt"]
    ]
    constant_per_run = [
        {
            "run_id": row["run_id"],
            "nominal_tilt_deg": row["nominal_tilt_deg"],
            "measured_tilt_deg": row["tilt_deg"],
            "observed_fitted_b": row["fitted_b"],
            "predicted_b": constant_b,
            "residual_b": float(row["fitted_b"] - constant_b),
            "sample_count": row["sample_count"],
        }
        for row in report["fit"]["per_run"]
    ]
    constant_metrics = _residual_metrics(constant_per_tilt)
    constant_run_metrics = _residual_metrics(constant_per_run)
    sin_model = _summarize_parametric_model(
        "b_alpha_sin",
        _get_primary_parametric_fit(report, "b_alpha_sin"),
        report["fit"]["b_alpha_sin_leave_one_repeat_out"],
        report["fit"]["b_alpha_sin_per_tilt"],
        report["fit"]["b_alpha_sin_per_run"],
    )
    quad_model = _summarize_parametric_model(
        "b_alpha_quadratic",
        _get_primary_parametric_fit(report, "b_alpha_quadratic"),
        report["fit"]["b_alpha_quadratic_leave_one_repeat_out"],
        report["fit"]["b_alpha_quadratic_per_tilt"],
        report["fit"]["b_alpha_quadratic_per_run"],
    )
    recommended_model, recommendation_reason, caution = _choose_recommended_model(report)
    if recommended_model == "b_alpha_quadratic":
        selected_per_tilt = report["fit"]["b_alpha_quadratic_per_tilt"]
        selected_metrics = quad_model["per_tilt_metrics"]
        selected_run_metrics = quad_model["per_run_metrics"]
    elif recommended_model == "b_alpha_sin":
        selected_per_tilt = report["fit"]["b_alpha_sin_per_tilt"]
        selected_metrics = sin_model["per_tilt_metrics"]
        selected_run_metrics = sin_model["per_run_metrics"]
    else:
        selected_per_tilt = constant_per_tilt
        selected_metrics = constant_metrics
        selected_run_metrics = constant_run_metrics
    return {
        "schema_version": report["schema_version"],
        "recommended_model": recommended_model,
        "recommendation_reason": recommendation_reason,
        "caution": caution,
        "headline_metrics": {
            "run_count": report["campaign"]["run_count"],
            "exit_sample_count": report["campaign"]["exit_sample_count"],
            "constant_b": constant_fit["fitted_b"],
            "constant_negative_log_likelihood": constant_fit["negative_log_likelihood"],
            "constant_cv_negative_log_likelihood": constant_cv["mean_test_negative_log_likelihood"],
            "selected_model_mean_abs_per_tilt_residual_b": selected_metrics["mean_abs"],
            "selected_model_max_abs_per_tilt_residual_b": selected_metrics["max_abs"],
            "selected_model_mean_abs_per_run_residual_b": selected_run_metrics["mean_abs"],
        },
        "model_comparison": {
            "constant_b": {
                "fitted_b": constant_fit["fitted_b"],
                "negative_log_likelihood": constant_fit["negative_log_likelihood"],
                "cv_negative_log_likelihood": constant_cv["mean_test_negative_log_likelihood"],
            },
            "b_alpha_sin": sin_model,
            "b_alpha_quadratic": quad_model,
        },
        "per_tilt": selected_per_tilt,
        "validation": report["validation"],
        "artifacts": report.get("artifacts", {}),
    }


def _build_constant_vs_quadratic_per_tilt_rows(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_tilt_rows = report["fit"]["per_tilt"]
    quadratic_rows = {
        float(row["nominal_tilt_deg"]): row
        for row in report["fit"]["b_alpha_quadratic_per_tilt"]
    }
    constant_b = float(_get_primary_constant_fit(report)["fitted_b"])
    quadratic_formula = _get_primary_parametric_fit(report, "b_alpha_quadratic")["formula"]

    rows: list[dict[str, Any]] = []
    constant_win_count = 0
    quadratic_win_count = 0
    tie_count = 0

    for row in per_tilt_rows:
        nominal_tilt = float(row["nominal_tilt_deg"])
        observed_b = float(row["fitted_b"])
        quadratic_row = quadratic_rows[nominal_tilt]
        quadratic_b = float(quadratic_row["predicted_b"])

        constant_residual_b = observed_b - constant_b
        quadratic_residual_b = observed_b - quadratic_b
        constant_abs_residual_b = abs(constant_residual_b)
        quadratic_abs_residual_b = abs(quadratic_residual_b)

        if quadratic_abs_residual_b + 1e-12 < constant_abs_residual_b:
            winner = "quadratic"
            quadratic_win_count += 1
        elif constant_abs_residual_b + 1e-12 < quadratic_abs_residual_b:
            winner = "constant"
            constant_win_count += 1
        else:
            winner = "tie"
            tie_count += 1

        rows.append(
            {
                "nominal_tilt_deg": nominal_tilt,
                "measured_tilt_deg_mean": float(row["measured_tilt_deg_mean"]),
                "observed_fitted_b": observed_b,
                "constant_b": constant_b,
                "constant_residual_b": constant_residual_b,
                "constant_abs_residual_b": constant_abs_residual_b,
                "quadratic_b": quadratic_b,
                "quadratic_residual_b": quadratic_residual_b,
                "quadratic_abs_residual_b": quadratic_abs_residual_b,
                "winner": winner,
                "quadratic_minus_constant_abs_residual_b": quadratic_abs_residual_b - constant_abs_residual_b,
                "sample_count": int(row["sample_count"]),
                "run_count": int(row["run_count"]),
            }
        )

    summary = {
        "metric": "absolute residual in per-tilt fitted B against the observed per-tilt fit",
        "constant_b": constant_b,
        "quadratic_formula": quadratic_formula,
        "quadratic_win_count": quadratic_win_count,
        "constant_win_count": constant_win_count,
        "tie_count": tie_count,
        "tilt_count": len(rows),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_summary(
    path: Path,
    report: dict[str, Any],
    run_rows: list[dict[str, Any]],
    per_tilt_rows: list[dict[str, Any]],
) -> None:
    gui_summary = _build_gui_summary(report)
    constant_vs_quadratic_rows, constant_vs_quadratic_summary = _build_constant_vs_quadratic_per_tilt_rows(report)
    constant_fit = _get_primary_constant_fit(report)
    sin_fit = _get_primary_parametric_fit(report, "b_alpha_sin")
    quad_fit = _get_primary_parametric_fit(report, "b_alpha_quadratic")
    harmonic_validation = report["validation"]["azimuth_harmonics"]
    harmonic_summary = harmonic_validation["summary"]
    harmonic_controls = harmonic_validation["controls"]
    original_control = harmonic_controls.get("nonzero_tilt_original", {})
    uniform_control = harmonic_controls.get("uniform_azimuth", {})
    reversed_control = harmonic_controls.get("tangent_reversal", {})
    lines = [
        "# CASINO Emission B Fit",
        "",
        f"- Source summary files / runs: {report['campaign']['run_count']}",
        f"- CASINO scan-point result blocks: {report['campaign'].get('scan_point_block_count', report['campaign']['run_count'])}",
        f"- Exit samples: {report['campaign']['exit_sample_count']}",
        f"- Fitted pooled B: {constant_fit['fitted_b']:.6f}",
        f"- Pooled objective: {constant_fit['negative_log_likelihood']:.6f}",
        f"- Root/grid B agreement: {constant_fit['root_grid_delta_b']:.6e}",
        f"- Observed feasible interval: [{constant_fit['observed_feasible_interval']['lower']}, {constant_fit['observed_feasible_interval']['upper']}]",
        f"- Domain feasible interval: [{constant_fit['domain_feasible_interval']['lower']}, {constant_fit['domain_feasible_interval']['upper']}]",
        f"- Approx. standard error: {constant_fit['approx_standard_error'] if constant_fit['approx_standard_error'] is not None else 'None'}",
        f"- Sin-law: B(alpha) = {sin_fit['b0']:.6f} + {sin_fit['b1']:.6f} sin(alpha)",
        f"- Sin-law objective: {sin_fit['negative_log_likelihood']:.6f}",
        f"- Quadratic law: B(alpha) = {quad_fit['b0']:.6f} + {quad_fit['b1']:.6f} sin(alpha) + {quad_fit['b2']:.6f} sin^2(alpha)",
        f"- Quadratic objective: {quad_fit['negative_log_likelihood']:.6f}",
        f"- Recommended model: `{gui_summary['recommended_model']}`",
        "",
        "## Notes",
        "",
        "- One CASINO summary file may contain multiple scan-point result blocks; schema 5 preserves their IDs and count checks in `scan_point_summary.csv`.",
        "- `summary.dat` backscattered second-row exits are the primary fit data.",
        "- `BSAngle.dat` does not identify `B` directly because the azimuthal asymmetry term integrates out in the 1D polar-angle marginal.",
        "- `BSAngle.dat` and `BSEnergy.dat` are used here as consistency checks against the parsed summary trajectories.",
        "",
        "## Azimuthal Oscillation Evidence",
        "",
        f"- Nonzero-tilt cosine coefficient C1: {original_control.get('cos_first', 'None')}",
        f"- Nonzero-tilt sine coefficient S1: {original_control.get('sin_first', 'None')}",
        f"- Uniform-azimuth control amplitude A1: {uniform_control.get('first_amplitude', 'None')}",
        f"- Tangent-reversal cosine coefficient C1: {reversed_control.get('cos_first', 'None')}",
        f"- Tangent-reversal amplitude delta: {harmonic_controls.get('tangent_reversal_amplitude_delta', 'None')}",
        f"- Zero-tilt first-harmonic amplitude: {harmonic_summary.get('zero_tilt_first_amplitude', 'None')}",
        f"- Median |S1/C1| across informative tilts: {harmonic_summary.get('median_abs_sine_to_cosine_ratio', 'None')}",
        f"- First-harmonic-dominant tilts: {harmonic_summary.get('first_harmonic_dominant_tilt_count', 0)} of {harmonic_summary.get('informative_tilt_count', 0)}",
        f"- Coordinate-frame tangent flips: {report['diagnostics']['coordinate_frame']['tangent_flip_count']}",
        "- Interpretation: C1 measures the cosine oscillation predicted by the Reimer-Riepenhausen term; S1 is a phase-error check. The likelihood fit remains the primary B estimate.",
        "",
        "## Per-Tilt B",
        "",
        "| Nominal Tilt (deg) | Measured Tilt Mean (deg) | Fitted B | Sin-Law B | Sin Residual | Quad-Law B | Quad Residual | Samples | Runs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_tilt_predictions_sin = {
        float(row["nominal_tilt_deg"]): row for row in report["fit"]["b_alpha_sin_per_tilt"]
    }
    per_tilt_predictions_quad = {
        float(row["nominal_tilt_deg"]): row for row in report["fit"]["b_alpha_quadratic_per_tilt"]
    }
    for row in per_tilt_rows:
        sin_prediction = per_tilt_predictions_sin[float(row["nominal_tilt_deg"])]
        quad_prediction = per_tilt_predictions_quad[float(row["nominal_tilt_deg"])]
        lines.append(
            f"| {row['nominal_tilt_deg']:.0f} | {row['measured_tilt_deg_mean']:.3f} | {row['fitted_b']:.6f} | {sin_prediction['predicted_b']:.6f} | {sin_prediction['residual_b']:.6f} | {quad_prediction['predicted_b']:.6f} | {quad_prediction['residual_b']:.6f} | {row['sample_count']} | {row['run_count']} |"
        )
    lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            f"- Constant-B pooled NLL: {constant_fit['negative_log_likelihood']:.6f}",
            f"- Sin-law pooled NLL: {sin_fit['negative_log_likelihood']:.6f}",
            f"- Quadratic-law pooled NLL: {quad_fit['negative_log_likelihood']:.6f}",
            f"- Leave-one-repeat-out constant mean test NLL: {report['fit']['leave_one_repeat_out']['mean_test_negative_log_likelihood']:.6f}",
            f"- Leave-one-repeat-out sin-law mean test NLL: {report['fit']['b_alpha_sin_leave_one_repeat_out']['mean_test_negative_log_likelihood']:.6f}",
            f"- Leave-one-repeat-out quadratic mean test NLL: {report['fit']['b_alpha_quadratic_leave_one_repeat_out']['mean_test_negative_log_likelihood']:.6f}",
            f"- RR normalization max |mass-1|: {report['validation']['rr_density_normalization']['max_abs_error']:.6e}",
            f"- Recommendation reason: {gui_summary['recommendation_reason']}",
            f"- Caution: {gui_summary['caution'] or 'None'}",
            f"- Constant vs quadratic tilt wins: quadratic {constant_vs_quadratic_summary['quadratic_win_count']}, constant {constant_vs_quadratic_summary['constant_win_count']}, ties {constant_vs_quadratic_summary['tie_count']}",
            f"- Objective plot: {report.get('artifacts', {}).get('objective_plot_path') or 'not generated'}",
            f"- Per-tilt plot: {report.get('artifacts', {}).get('per_tilt_plot_path') or 'not generated'}",
            f"- Run-trend plot: {report.get('artifacts', {}).get('run_trend_plot_path') or 'not generated'}",
            f"- Azimuth harmonics plot: {report.get('artifacts', {}).get('azimuth_harmonics_plot_path') or 'not generated'}",
            f"- Azimuth histograms plot: {report.get('artifacts', {}).get('azimuth_histograms_plot_path') or 'not generated'}",
            f"- Coordinate-frame plot: {report.get('artifacts', {}).get('coordinate_frame_plot_path') or 'not generated'}",
            "",
            "## Constant vs Quadratic Per Tilt",
            "",
            "| Nominal Tilt (deg) | Observed B | Constant B | Constant |residual| | Quadratic B | Quadratic |residual| | Winner |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in constant_vs_quadratic_rows:
        lines.append(
            f"| {row['nominal_tilt_deg']:.0f} | {row['observed_fitted_b']:.6f} | {row['constant_b']:.6f} | {row['constant_abs_residual_b']:.6f} | {row['quadratic_b']:.6f} | {row['quadratic_abs_residual_b']:.6f} | {row['winner']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Run B",
            "",
            "| Run | Nominal Tilt (deg) | Measured Tilt (deg) | Fitted B | Sin-Law B | Sin Residual | Quad-Law B | Quad Residual | Samples |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    per_run_predictions_sin = {row["run_id"]: row for row in report["fit"]["b_alpha_sin_per_run"]}
    per_run_predictions_quad = {row["run_id"]: row for row in report["fit"]["b_alpha_quadratic_per_run"]}
    for row in run_rows:
        sin_prediction = per_run_predictions_sin[row["run_id"]]
        quad_prediction = per_run_predictions_quad[row["run_id"]]
        lines.append(
            f"| {row['run_id']} | {row['nominal_tilt_deg']:.0f} | {row['tilt_deg']:.3f} | {row['fitted_b']:.6f} | {sin_prediction['predicted_b']:.6f} | {sin_prediction['residual_b']:.6f} | {quad_prediction['predicted_b']:.6f} | {quad_prediction['residual_b']:.6f} | {row['sample_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_casino_emission_b(
    *,
    cas_root: str | Path,
    export_root: str | Path,
    output_dir: str | Path,
    grid_points: int = 2001,
) -> dict[str, Any]:
    run_summaries, scan_point_summaries, samples = _collect_run_data(
        cas_root=cas_root,
        export_root=export_root,
    )
    if not samples:
        raise ValueError("No backscattered exit samples were found in the summary files.")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    pooled_fit, b_grid, objective_grid, derivative_grid, curvature_grid = _fit_pooled_b_with_diagnostics(
        samples,
        grid_points=grid_points,
    )
    global_b, global_objective, _, _ = _fit_b_from_samples(
        samples,
        grid_points=grid_points,
        equal_run_weight=True,
    )
    per_run_rows = _fit_per_run(samples, grid_points=grid_points)
    per_tilt_rows = _fit_per_tilt(samples, grid_points=grid_points)
    fold_rows = _leave_one_repeat_out(samples, grid_points=grid_points)
    law_fit_equal, law_per_run_rows_equal, _ = _fit_b_alpha_sin_from_samples(
        samples,
        scalar_grid_points=grid_points,
        law_search_points=max(101, min(401, int(math.sqrt(max(101, grid_points))) * 10 + 1)),
        equal_run_weight=True,
    )
    law_fit_weighted, law_per_run_rows, _ = _fit_b_alpha_sin_from_samples(
        samples,
        scalar_grid_points=grid_points,
        law_search_points=max(101, min(401, int(math.sqrt(max(101, grid_points))) * 10 + 1)),
        equal_run_weight=False,
    )
    law_per_tilt_rows = _fit_b_alpha_sin_per_tilt(per_tilt_rows, law_fit_weighted)
    law_fold_rows = _leave_one_repeat_out_b_alpha_sin(
        samples,
        scalar_grid_points=grid_points,
        law_search_points=max(101, min(401, int(math.sqrt(max(101, grid_points))) * 10 + 1)),
    )
    quadratic_fit_equal, quadratic_per_run_rows_equal, _ = _fit_b_alpha_quadratic_from_samples(
        samples,
        scalar_grid_points=grid_points,
        equal_run_weight=True,
        initial_sin_fit=law_fit_equal,
    )
    quadratic_fit_weighted, quadratic_per_run_rows, _ = _fit_b_alpha_quadratic_from_samples(
        samples,
        scalar_grid_points=grid_points,
        equal_run_weight=False,
        initial_sin_fit=law_fit_weighted,
    )
    quadratic_per_tilt_rows = _fit_b_alpha_quadratic_per_tilt(per_tilt_rows, quadratic_fit_weighted)
    quadratic_fold_rows = _leave_one_repeat_out_b_alpha_quadratic(
        samples,
        scalar_grid_points=grid_points,
    )

    sample_rows = [asdict(sample) for sample in samples]
    run_rows = [asdict(run) for run in run_summaries]
    scan_point_rows = [asdict(scan_point) for scan_point in scan_point_summaries]
    coordinate_frame_rows, coordinate_frame_summary = _build_coordinate_frame_diagnostics(run_summaries)
    azimuth_harmonics = _build_azimuth_harmonic_diagnostics(samples)
    fold_dicts = [asdict(fold) for fold in fold_rows]
    law_fold_dicts = [asdict(fold) for fold in law_fold_rows]
    quadratic_fold_dicts = list(quadratic_fold_rows)
    objective_rows = [
        {
            "b": float(b),
            "negative_log_likelihood": float(obj),
            "derivative": float(deriv),
            "second_derivative": float(curvature),
        }
        for b, obj, deriv, curvature in zip(b_grid, objective_grid, derivative_grid, curvature_grid)
    ]

    run_angle_ce = [row["summary_vs_bsangle_cross_entropy"] for row in run_rows if row["summary_vs_bsangle_cross_entropy"] is not None]
    run_angle_l1 = [row["summary_vs_bsangle_l1"] for row in run_rows if row["summary_vs_bsangle_l1"] is not None]
    rr_angle_ce = [row["rr_polar_vs_bsangle_cross_entropy"] for row in run_rows if row["rr_polar_vs_bsangle_cross_entropy"] is not None]
    run_energy_ce = [row["summary_vs_bsenergy_cross_entropy"] for row in run_rows if row["summary_vs_bsenergy_cross_entropy"] is not None]
    run_energy_l1 = [row["summary_vs_bsenergy_l1"] for row in run_rows if row["summary_vs_bsenergy_l1"] is not None]
    scan_point_count_mismatches = [row for row in scan_point_rows if row["counts_match"] is False]
    scan_point_count_failures = [
        row for row in scan_point_rows if row["counts_within_export_tolerance"] is False
    ]
    multi_scan_point_runs = [row for row in run_rows if int(row["scan_point_count"]) > 1]
    ingestion_diagnostics = {
        "source_summary_file_count": len(run_rows),
        "scan_point_block_count": len(scan_point_rows),
        "multi_scan_point_source_file_count": len(multi_scan_point_runs),
        "single_scan_point_source_file_count": sum(int(row["scan_point_count"]) == 1 for row in run_rows),
        "maximum_scan_points_per_source_file": max(int(row["scan_point_count"]) for row in run_rows),
        "scan_point_count_mismatch_count": len(scan_point_count_mismatches),
        "scan_point_count_mismatch_ids": [row["scan_point_run_id"] for row in scan_point_count_mismatches],
        "scan_point_count_beyond_export_tolerance_count": len(scan_point_count_failures),
        "scan_point_count_beyond_export_tolerance_ids": [
            row["scan_point_run_id"] for row in scan_point_count_failures
        ],
        "all_reported_counts_match_exactly": not scan_point_count_mismatches,
        "all_reported_counts_within_export_tolerance": not scan_point_count_failures,
        "export_count_tolerance": (
            "At most one trajectory per scan-point block. CASINO summary files contain trajectory "
            "IDs 0..19998 for a 20000-electron scan, so a reported category can exceed parsed "
            "trajectory membership by one when the omitted trajectory belongs to that category."
        ),
        "interpretation": (
            "A summary file is one CASINO run and may contain multiple scan-point result blocks. "
            "Each block has a stable '<run_id>#sp<scan_point_id>' identifier; statistical grouping "
            "continues to use the source run_id until the Stage 2 resampling design is selected."
        ),
    }
    tilt_mismatch_runs = [
        {
            "run_id": row["run_id"],
            "nominal_tilt_deg": row["nominal_tilt_deg"],
            "measured_tilt_deg": row["tilt_deg"],
            "tilt_delta_deg": row["tilt_deg"] - row["nominal_tilt_deg"],
        }
        for row in run_rows
        if abs(float(row["tilt_deg"]) - float(row["nominal_tilt_deg"])) > 5.0
    ]
    mismatch_run_ids = {row["run_id"] for row in tilt_mismatch_runs}
    consistent_samples = [sample for sample in samples if sample.run_id not in mismatch_run_ids]
    consistent_fit: dict[str, Any] | None = None
    if consistent_samples and len(consistent_samples) < len(samples):
        consistent_fit_result, _, _, _, _ = _fit_pooled_b_with_diagnostics(
            consistent_samples,
            grid_points=grid_points,
        )
        consistent_fold_rows = _leave_one_repeat_out(consistent_samples, grid_points=grid_points)
        consistent_fit = {
            **consistent_fit_result,
            "run_count": len({sample.run_id for sample in consistent_samples}),
            "exit_sample_count": len(consistent_samples),
            "leave_one_repeat_out": {
                "folds": [asdict(fold) for fold in consistent_fold_rows],
                "mean_fitted_b": None
                if not consistent_fold_rows
                else float(np.mean([fold.fitted_b for fold in consistent_fold_rows])),
                "std_fitted_b": None
                if not consistent_fold_rows
                else float(np.std([fold.fitted_b for fold in consistent_fold_rows])),
                "mean_test_negative_log_likelihood": None
                if not consistent_fold_rows
                else float(np.mean([fold.test_negative_log_likelihood for fold in consistent_fold_rows])),
            },
        }

    tilt_values = np.asarray([sample.tilt_deg for sample in samples], dtype=np.float64)
    normalization_check = _build_rr_normalization_check(tilt_values, float(pooled_fit["fitted_b"]))
    run_trend_diagnostics = {
        "linear_vs_measured_tilt_deg": _fit_linear_trend(
            np.asarray([float(row["tilt_deg"]) for row in per_run_rows], dtype=np.float64),
            np.asarray([float(row["fitted_b"]) for row in per_run_rows], dtype=np.float64),
        ),
        "linear_vs_sin_alpha": _fit_linear_trend(
            np.sin(np.radians(np.asarray([float(row["tilt_deg"]) for row in per_run_rows], dtype=np.float64))),
            np.asarray([float(row["fitted_b"]) for row in per_run_rows], dtype=np.float64),
        ),
    }

    report = {
        "schema_version": 5,
        "campaign": {
            "cas_root": str(Path(cas_root).resolve()),
            "export_root": str(Path(export_root).resolve()),
            "run_count": len(run_rows),
            "source_summary_file_count": len(run_rows),
            "scan_point_block_count": len(scan_point_rows),
            "exit_sample_count": len(sample_rows),
            "nominal_tilt_deg_values": sorted({row["nominal_tilt_deg"] for row in run_rows}),
            "measured_tilt_deg_values": sorted({row["tilt_deg"] for row in run_rows}),
            "tilt_mismatch_runs": tilt_mismatch_runs,
        },
        "fit": {
            "global_pooled": dict(pooled_fit),
            "global_equal_run": {
                "fitted_b": global_b,
                "negative_log_likelihood": global_objective,
            },
            "global_sample_weighted": dict(pooled_fit),
            "b_alpha_sin_equal_run": {
                "b0": law_fit_equal["b0"],
                "b1": law_fit_equal["b1"],
                "negative_log_likelihood": law_fit_equal["negative_log_likelihood"],
                "formula": "B(alpha_deg) = b0 + b1 * sin(alpha_deg)",
            },
            "b_alpha_sin_sample_weighted": {
                "b0": law_fit_weighted["b0"],
                "b1": law_fit_weighted["b1"],
                "negative_log_likelihood": law_fit_weighted["negative_log_likelihood"],
                "formula": "B(alpha_deg) = b0 + b1 * sin(alpha_deg)",
            },
            "b_alpha_quadratic_equal_run": {
                "b0": quadratic_fit_equal["b0"],
                "b1": quadratic_fit_equal["b1"],
                "b2": quadratic_fit_equal["b2"],
                "negative_log_likelihood": quadratic_fit_equal["negative_log_likelihood"],
                "formula": quadratic_fit_equal["formula"],
            },
            "b_alpha_quadratic_sample_weighted": {
                "b0": quadratic_fit_weighted["b0"],
                "b1": quadratic_fit_weighted["b1"],
                "b2": quadratic_fit_weighted["b2"],
                "negative_log_likelihood": quadratic_fit_weighted["negative_log_likelihood"],
                "formula": quadratic_fit_weighted["formula"],
            },
            "b_alpha_sin_equal_run_per_run": law_per_run_rows_equal,
            "b_alpha_quadratic_equal_run_per_run": quadratic_per_run_rows_equal,
            "global_equal_run_consistent_tilts_only": consistent_fit,
            "global_pooled_consistent_tilts_only": consistent_fit,
            "per_run": per_run_rows,
            "per_tilt": per_tilt_rows,
            "b_alpha_sin_per_run": law_per_run_rows,
            "b_alpha_sin_per_tilt": law_per_tilt_rows,
            "b_alpha_quadratic_per_run": quadratic_per_run_rows,
            "b_alpha_quadratic_per_tilt": quadratic_per_tilt_rows,
            "leave_one_repeat_out": {
                "folds": fold_dicts,
                "mean_fitted_b": None if not fold_rows else float(np.mean([row["fitted_b"] for row in fold_dicts])),
                "std_fitted_b": None if not fold_rows else float(np.std([row["fitted_b"] for row in fold_dicts])),
                "mean_test_negative_log_likelihood": None
                if not fold_rows
                else float(np.mean([row["test_negative_log_likelihood"] for row in fold_dicts])),
            },
            "b_alpha_sin_leave_one_repeat_out": {
                "folds": law_fold_dicts,
                "mean_fitted_b0": None if not law_fold_rows else float(np.mean([row["fitted_b0"] for row in law_fold_dicts])),
                "std_fitted_b0": None if not law_fold_rows else float(np.std([row["fitted_b0"] for row in law_fold_dicts])),
                "mean_fitted_b1": None if not law_fold_rows else float(np.mean([row["fitted_b1"] for row in law_fold_dicts])),
                "std_fitted_b1": None if not law_fold_rows else float(np.std([row["fitted_b1"] for row in law_fold_dicts])),
                "mean_test_negative_log_likelihood": None
                if not law_fold_rows
                else float(np.mean([row["test_negative_log_likelihood"] for row in law_fold_dicts])),
            },
            "b_alpha_quadratic_leave_one_repeat_out": {
                "folds": quadratic_fold_dicts,
                "mean_fitted_b0": None if not quadratic_fold_rows else float(np.mean([row["fitted_b0"] for row in quadratic_fold_dicts])),
                "std_fitted_b0": None if not quadratic_fold_rows else float(np.std([row["fitted_b0"] for row in quadratic_fold_dicts])),
                "mean_fitted_b1": None if not quadratic_fold_rows else float(np.mean([row["fitted_b1"] for row in quadratic_fold_dicts])),
                "std_fitted_b1": None if not quadratic_fold_rows else float(np.std([row["fitted_b1"] for row in quadratic_fold_dicts])),
                "mean_fitted_b2": None if not quadratic_fold_rows else float(np.mean([row["fitted_b2"] for row in quadratic_fold_dicts])),
                "std_fitted_b2": None if not quadratic_fold_rows else float(np.std([row["fitted_b2"] for row in quadratic_fold_dicts])),
                "mean_test_negative_log_likelihood": None
                if not quadratic_fold_rows
                else float(np.mean([row["test_negative_log_likelihood"] for row in quadratic_fold_dicts])),
            },
        },
        "validation": {
            "summary_vs_bsangle": {
                "mean_cross_entropy": None if not run_angle_ce else float(np.mean(run_angle_ce)),
                "mean_l1": None if not run_angle_l1 else float(np.mean(run_angle_l1)),
            },
            "summary_vs_bsenergy": {
                "mean_cross_entropy": None if not run_energy_ce else float(np.mean(run_energy_ce)),
                "mean_l1": None if not run_energy_l1 else float(np.mean(run_energy_l1)),
            },
            "rr_polar_vs_bsangle": {
                "mean_cross_entropy": None if not rr_angle_ce else float(np.mean(rr_angle_ce)),
                "note": "This 1D polar-angle check is independent of the fitted B parameter.",
            },
            "rr_density_normalization": normalization_check,
            "azimuth_harmonics": {
                "summary": azimuth_harmonics["summary"],
                "controls": azimuth_harmonics["controls"],
                "polar_band_consistency": azimuth_harmonics["polar_band_consistency"],
            },
        },
        "diagnostics": {
            "primary_constant_model": "global_pooled",
            "primary_tilt_models": {
                "sin": "b_alpha_sin_sample_weighted",
                "quadratic": "b_alpha_quadratic_sample_weighted",
            },
            "per_run_linear_trends": run_trend_diagnostics,
            "coordinate_frame": coordinate_frame_summary,
            "ingestion": ingestion_diagnostics,
        },
        "notes": [
            "Backscattered second-row exits from summary.dat are the primary data source for fitting B.",
            "The primary constant-B estimate is the pooled trajectory maximum-likelihood solve over all exit trajectories.",
            "BSAngle.dat and BSEnergy.dat are used as consistency checks against the parsed summary trajectories, not as the primary B objective.",
            "The 1D BSAngle marginal does not identify B directly because the azimuthal asymmetry term integrates out over varphi.",
            "The entrance plane is inferred from the summary first-row coordinates instead of trusting the folder tilt label.",
            "A summary.dat file can contain multiple CASINO scan-point result blocks; schema 5 preserves each block and its reported counts while retaining source-file run grouping.",
            "The azimuthal cosine and sine coefficients are independent Fourier diagnostics. They demonstrate the expected oscillation and expose phase errors, but the trajectory NLL remains the primary B estimator.",
            "Polar-band diagnostic B values test whether the cosine amplitude scales with sin(alpha) sin(phi) as required by the Reimer-Riepenhausen equation.",
        ],
    }
    plot_info = _write_per_tilt_comparison_plot(output_path / "b_fit_per_tilt_comparison.png", report)
    objective_plot_info = _write_objective_diagnostics_plot(
        output_path / "b_fit_objective_diagnostics.png",
        objective_rows,
        float(pooled_fit["fitted_b"]),
    )
    run_trend_plot_info = _write_run_trend_plot(output_path / "b_fit_run_trends.png", per_run_rows)
    harmonic_plot_info = _write_azimuth_harmonics_plot(
        output_path / "azimuth_harmonics_by_tilt.png", azimuth_harmonics
    )
    azimuth_histogram_plot_info = _write_azimuth_histograms_plot(
        output_path / "azimuth_histograms_selected_tilts.png", samples
    )
    coordinate_frame_plot_info = _write_coordinate_frame_plot(
        output_path / "coordinate_frame_diagnostics.png", coordinate_frame_rows
    )
    constant_vs_quadratic_rows, constant_vs_quadratic_summary = _build_constant_vs_quadratic_per_tilt_rows(report)
    report["tilt_mode_comparison"] = {
        "constant_vs_quadratic": {
            "summary": constant_vs_quadratic_summary,
            "rows": constant_vs_quadratic_rows,
        }
    }
    report["artifacts"] = {
        "per_tilt_plot_status": plot_info["status"],
        "per_tilt_plot_path": plot_info["path"],
        "per_tilt_plot_reason": plot_info["reason"],
        "objective_plot_status": objective_plot_info["status"],
        "objective_plot_path": objective_plot_info["path"],
        "objective_plot_reason": objective_plot_info["reason"],
        "run_trend_plot_status": run_trend_plot_info["status"],
        "run_trend_plot_path": run_trend_plot_info["path"],
        "run_trend_plot_reason": run_trend_plot_info["reason"],
        "azimuth_harmonics_plot_status": harmonic_plot_info["status"],
        "azimuth_harmonics_plot_path": harmonic_plot_info["path"],
        "azimuth_harmonics_plot_reason": harmonic_plot_info["reason"],
        "azimuth_histograms_plot_status": azimuth_histogram_plot_info["status"],
        "azimuth_histograms_plot_path": azimuth_histogram_plot_info["path"],
        "azimuth_histograms_plot_reason": azimuth_histogram_plot_info["reason"],
        "coordinate_frame_plot_status": coordinate_frame_plot_info["status"],
        "coordinate_frame_plot_path": coordinate_frame_plot_info["path"],
        "coordinate_frame_plot_reason": coordinate_frame_plot_info["reason"],
        "objective_curve_csv": str(output_path / "objective_curve.csv"),
        "constant_vs_quadratic_per_tilt_csv": str(output_path / "constant_vs_quadratic_per_tilt.csv"),
        "constant_vs_quadratic_per_tilt_json": str(output_path / "constant_vs_quadratic_per_tilt.json"),
        "azimuth_harmonics_per_run_csv": str(output_path / "azimuth_harmonics_per_run.csv"),
        "azimuth_harmonics_per_tilt_csv": str(output_path / "azimuth_harmonics_per_tilt.csv"),
        "azimuth_harmonics_per_polar_band_csv": str(output_path / "azimuth_harmonics_per_polar_band.csv"),
        "azimuth_harmonics_summary_json": str(output_path / "azimuth_harmonics_summary.json"),
        "coordinate_frame_diagnostics_csv": str(output_path / "coordinate_frame_diagnostics.csv"),
        "scan_point_summary_csv": str(output_path / "scan_point_summary.csv"),
    }

    _write_csv(output_path / "exit_samples.csv", sample_rows)
    _write_csv(output_path / "run_summary.csv", run_rows)
    _write_csv(output_path / "scan_point_summary.csv", scan_point_rows)
    _write_csv(output_path / "per_run_fit.csv", per_run_rows)
    _write_csv(output_path / "per_tilt_fit.csv", per_tilt_rows)
    _write_csv(output_path / "leave_one_repeat_out.csv", fold_dicts)
    _write_csv(output_path / "b_alpha_sin_per_run.csv", law_per_run_rows)
    _write_csv(output_path / "b_alpha_sin_per_tilt.csv", law_per_tilt_rows)
    _write_csv(output_path / "b_alpha_sin_leave_one_repeat_out.csv", law_fold_dicts)
    _write_csv(output_path / "b_alpha_quadratic_per_run.csv", quadratic_per_run_rows)
    _write_csv(output_path / "b_alpha_quadratic_per_tilt.csv", quadratic_per_tilt_rows)
    _write_csv(output_path / "b_alpha_quadratic_leave_one_repeat_out.csv", quadratic_fold_dicts)
    _write_csv(output_path / "constant_vs_quadratic_per_tilt.csv", constant_vs_quadratic_rows)
    _write_csv(output_path / "objective_curve.csv", objective_rows)
    _write_csv(output_path / "azimuth_harmonics_per_run.csv", azimuth_harmonics["per_run"])
    _write_csv(output_path / "azimuth_harmonics_per_tilt.csv", azimuth_harmonics["per_tilt"])
    _write_csv(output_path / "azimuth_harmonics_per_polar_band.csv", azimuth_harmonics["per_polar_band"])
    _write_csv(output_path / "coordinate_frame_diagnostics.csv", coordinate_frame_rows)
    (output_path / "azimuth_harmonics_summary.json").write_text(
        json.dumps(
            {
                **azimuth_harmonics,
                "coordinate_frame": coordinate_frame_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_path / "b_fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_path / "b_fit_gui_summary.json").write_text(json.dumps(_build_gui_summary(report), indent=2), encoding="utf-8")
    (output_path / "constant_vs_quadratic_per_tilt.json").write_text(
        json.dumps(
            {
                "summary": constant_vs_quadratic_summary,
                "rows": constant_vs_quadratic_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown_summary(output_path / "b_fit_summary.md", report, per_run_rows, per_tilt_rows)

    return report
