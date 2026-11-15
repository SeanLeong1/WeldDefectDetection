"""Compact CASINO boundary histories and audit local-surface emission.

No simulation is launched and no source file is changed. Intermediate collisions
are read once and discarded; the first and final rows of each history are retained.
The surface-conditioned analysis deliberately differs from CASINO's global exit
class labels. Its selection counts and geometric assumptions remain in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from ..atomic_io import write_json_atomic

from .casino_emission_fit import (
    _SCAN_POINT_RESULTS_RE, _TRAJECTORY_RE, _parse_collision_row, _parse_run_id,
    _sha256_file,
)

SCHEMA = 1
TYPES = {"Simple": 0, "Backscattered": 1, "Transmitted": 2}


def _write_json(path: Path, value) -> None:
    write_json_atomic(path, value)


def compact_summary(source: Path, destination: Path) -> dict:
    """Stream a plain-text summary into a compressed, pickle-free boundary cache.

    Memory grows with the number of histories, never with collision count.
    Existing caches are reused only after matching the source content hash.
    """
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("The compact output must differ from the source.")
    if destination.exists():
        with np.load(destination, allow_pickle=False) as cached:
            meta = json.loads(str(cached["metadata"]))
        if (meta.get("schema") == SCHEMA and meta.get("source") == str(source)
                and meta.get("sha256") == _sha256_file(source)):
            return meta

    first_rows, final_rows, blocks, scan_ids, trajectory_ids, kinds, row_counts = ([] for _ in range(7))
    block, scan_id, trajectory_id = 0, 0, None
    first = final = None
    kind, row_count, waiting_type = -1, 0, False
    digest = hashlib.sha256()
    header, reported = [], {}

    def finish():
        if trajectory_id is None:
            return
        blocks.append(block)
        scan_ids.append(scan_id)
        trajectory_ids.append(trajectory_id)
        kinds.append(kind)
        row_counts.append(row_count)
        first_rows.append(first if first is not None else [math.nan] * 8)
        final_rows.append(final if final is not None else [math.nan] * 8)

    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            line = raw.decode("utf-8", errors="replace")
            stripped = line.strip()
            if trajectory_id is None and block == 0 and len(header) < 200:
                header.append(stripped)
            match = _SCAN_POINT_RESULTS_RE.match(stripped)
            if match:
                finish()
                trajectory_id, first, final = None, None, None
                row_count, waiting_type = 0, False
                block += 1
                scan_id = int(match.group("scan_point"))
                reported[block] = {"scan_point_id": scan_id, "source_line": line_number}
                continue
            match = _TRAJECTORY_RE.match(stripped)
            if match:
                finish()
                trajectory_id = int(match.group(1))
                first = final = None
                kind, row_count, waiting_type = -1, 0, False
                continue
            if stripped == '"Type : "':
                waiting_type = True
                continue
            if waiting_type and stripped:
                kind = TYPES.get(stripped.strip('"'), -1)
                waiting_type = False
                continue
            for label in ("nb backscattered", "nb transmitted", "BackScattering Coefficient"):
                if stripped.startswith('"' + label + '"'):
                    try:
                        reported.setdefault(block, {})[label] = float(stripped.split('"', 2)[-1].strip())
                    except ValueError:
                        pass
            if trajectory_id is None:
                continue
            row = _parse_collision_row(line)
            if row is not None:
                values = [row.x_nm, row.y_nm, row.z_nm, row.rx, row.ry, row.rz,
                          row.energy_kev, row.segment_length_nm]
                if first is None:
                    first = values
                final = values
                row_count += 1
    finish()
    meta = {
        "schema": SCHEMA, "source": str(source), "source_bytes": source.stat().st_size,
        "sha256": digest.hexdigest(), "trajectory_count": len(blocks),
        "collision_rows_read": sum(row_counts), "max_rows_per_history": max(row_counts, default=0),
        "stored_rows_per_history": "entrance and final only", "header": header,
        "reported_scan_counts": {str(key): value for key, value in reported.items()},
        "columns": ["x_nm", "y_nm", "z_nm", "rx", "ry", "rz", "energy_kev", "segment_length_nm"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(meta),
                            first=np.asarray(first_rows, dtype=np.float64).reshape(-1, 8),
                            final=np.asarray(final_rows, dtype=np.float64).reshape(-1, 8),
                            block=np.asarray(blocks, dtype=np.int32),
                            scan_id=np.asarray(scan_ids, dtype=np.int32),
                            trajectory_id=np.asarray(trajectory_ids, dtype=np.int64),
                            kind=np.asarray(kinds, dtype=np.int8),
                            row_count=np.asarray(row_counts, dtype=np.int64))
    temporary.replace(destination)
    return meta


def _plane(points: np.ndarray, beam: np.ndarray):
    center = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    if normal @ -beam < 0:
        normal = -normal
    return center, normal, singular


def surface_frame(first: np.ndarray, final: np.ndarray, has_exit: np.ndarray,
                  plane_tolerance_nm: float = 1.0):
    """Fit each landing patch, refining the plane with nearby boundary endpoints.

    Entrance spots span nanometres, while exit points span micrometres. Printed
    coordinate rounding can bias an entrance-only slope when extrapolated. The
    refinement admits endpoints within 10 nm + 0.0005 * radius of the seed plane,
    then iterates a 1 nm residual gate. It must retain the entrance points and
    change the normal within a bounded allowance for entrance-plane uncertainty.
    The 0.1 degree floor is widened only when residual scatter relative to the
    spot extent warrants it, up to 0.5 degrees. This engineering allowance is
    not a calibrated confidence interval. All tolerances are reported.
    """
    if len(first) < 3 or not np.isfinite(first[:, :6]).all():
        raise ValueError("At least three finite entrance points/directions are required.")
    beam = first[:, 3:6].mean(axis=0)
    beam_norm = np.linalg.norm(beam)
    if beam_norm <= 1e-12:
        raise ValueError("Entrance directions do not identify an incident beam.")
    beam /= beam_norm
    center, normal, singular = _plane(first[:, :3], beam)
    if singular[1] <= max(singular[0] * 1e-8, 1e-10):
        raise ValueError("Entrance points do not identify a plane (rank below two).")
    seed_normal = normal.copy()
    # Printed coordinates can have ~0.1 nm steps over a ~5 nm landing spot.
    # A fixed 0.1 degree cap can reject a much better plane supported by distant
    # exits. Use a conservative slope-noise proxy, while preserving a hard cap
    # and the independent entrance-residual and nominal-incidence checks.
    slope_noise = float(singular[2] / singular[1] / np.sqrt(max(len(first)-3, 1)))
    refinement_limit = float(np.clip(np.degrees(4*slope_noise), .1, .5))
    exits = final[has_exit, :3]
    distances = np.linalg.norm(exits - center, axis=1)
    near = np.abs((exits - center) @ normal) <= 10.0 + 0.0005 * distances
    refinement_count = 0
    if near.sum() >= 20:
        for _ in range(3):
            candidate_center, candidate_normal, _ = _plane(
                np.concatenate([first[:, :3], exits[near]]), beam)
            angle_change = np.degrees(np.arccos(np.clip(candidate_normal @ seed_normal, -1, 1)))
            entry_error = np.quantile(np.abs((first[:, :3] - candidate_center) @ candidate_normal), .99)
            if angle_change > refinement_limit or entry_error > plane_tolerance_nm:
                break
            center, normal = candidate_center, candidate_normal
            refinement_count = int(near.sum())
            near = np.abs((exits - center) @ normal) <= plane_tolerance_nm
            if near.sum() < 20:
                break
    tangent = beam - (beam @ normal) * normal
    if np.linalg.norm(tangent) < 1e-4:
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = axis - (axis @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    second = np.cross(normal, tangent)
    tilt = float(np.degrees(np.arccos(np.clip(normal @ -beam, -1, 1))))
    return center, normal, tangent, second, {
        "measured_tilt_deg": tilt,
        "entrance_only_tilt_deg": float(np.degrees(np.arccos(np.clip(seed_normal @ -beam, -1, 1)))),
        "normal": normal.tolist(), "plane_origin_nm": center.tolist(),
        "entrance_residual_p99_nm": float(np.quantile(np.abs((first[:, :3] - center) @ normal), .99)),
        "plane_tolerance_nm": plane_tolerance_nm,
        "refinement_endpoint_count": refinement_count,
        "refinement_max_normal_change_deg": refinement_limit,
        "refinement_slope_noise_proxy_rad": slope_noise,
        "refinement_guard_method": "four_times_slope_noise_proxy_clipped_0.1_to_0.5_deg",
    }


def select_surface_exits(cache: Path, nominal_tilt: float, repeat: int,
                         plane_tolerance_nm: float = 1.0):
    """Return angles for the declared incident face, independent of global class.

    Off-nominal landing patches remain in diagnostics and are not relabelled as
    a new independent simulation. Off-face exits remain counted, rather than
    silently becoming observations from an infinite-plane emission model.
    """
    with np.load(cache, allow_pickle=False) as data:
        arrays = {name: data[name] for name in ("first", "final", "block", "kind", "row_count")}
        metadata = json.loads(str(data["metadata"]))
    records, samples = [], []
    for block in np.unique(arrays["block"]):
        mask = arrays["block"] == block
        first, final = arrays["first"][mask], arrays["final"][mask]
        kinds, counts = arrays["kind"][mask], arrays["row_count"][mask]
        has_exit = (counts >= 2) & np.isin(kinds, [1, 2]) & np.isfinite(final).all(axis=1)
        record = {"source": metadata["source"], "block": int(block),
                  "nominal_tilt_deg": nominal_tilt, "repeat": repeat,
                  "primary_histories_exported": len(first)}
        reported = metadata["reported_scan_counts"].get(str(int(block)), {})
        deltas = {}
        for label, code in (("nb backscattered", 1), ("nb transmitted", 2)):
            if label in reported:
                deltas[label] = int(np.sum(kinds == code)) - int(reported[label])
        count_ok = all(abs(delta) <= 1 for delta in deltas.values())
        record["reported_count_deltas"] = deltas
        record["count_status"] = ("not_reported" if not deltas else
                                  "exact_match" if all(delta == 0 for delta in deltas.values()) else
                                  "within_one_exported_history" if count_ok else "mismatch")
        record["unknown_history_types"] = int(np.sum(kinds < 0))
        record["incident_energy_kev_range"] = [float(np.nanmin(first[:, 6])), float(np.nanmax(first[:, 6]))]
        try:
            center, normal, tangent, second, frame = surface_frame(first, final, has_exit, plane_tolerance_nm)
        except ValueError as exc:
            record.update(status="rejected_frame", reason=str(exc))
            records.append(record)
            continue
        record.update(frame)
        unit = final[:, 3:6].copy()
        norms = np.linalg.norm(unit, axis=1)
        unit /= np.where(norms > 0, norms, 1)[:, None]
        mu = np.clip(unit @ normal, -1, 1)
        residual = np.abs((final[:, :3] - center) @ normal)
        on_face = has_exit & (residual <= plane_tolerance_nm)
        selected = on_face & (mu > 0) & (norms > 0) & (final[:, 6] > .05)
        angle_match = abs(frame["measured_tilt_deg"] - nominal_tilt) <= .1
        flat = frame["entrance_residual_p99_nm"] <= plane_tolerance_nm
        record.update({
            "status": "eligible" if angle_match and flat and count_ok and not record["unknown_history_types"] else "excluded_landing_patch",
            "nominal_angle_match": angle_match,
            "global_backscattered": int(np.sum(kinds == 1)),
            "global_transmitted": int(np.sum(kinds == 2)),
            "same_surface_outward": int(selected.sum()),
            "recovered_global_transmitted": int(np.sum(selected & (kinds == 2))),
            "global_backscattered_off_face": int(np.sum(has_exit & (kinds == 1) & ~on_face)),
            "off_face_exits": int(np.sum(has_exit & ~on_face)),
            "on_face_nonoutward": int(np.sum(on_face & (mu <= 0))),
            "class_direction_counts": {
                name: {"global_z_positive": int(np.sum(has_exit & (kinds == code) & (unit[:, 2] > 0))),
                       "global_z_negative": int(np.sum(has_exit & (kinds == code) & (unit[:, 2] < 0)))}
                for name, code in (("Backscattered", 1), ("Transmitted", 2))},
            "surface_yield_per_exported_history": float(selected.sum() / len(first)),
        })
        records.append(record)
        if record["status"] == "eligible":
            # Columns: alpha, mu=cos(phi), varphi, repeat, nominal incidence.
            samples.append(np.column_stack([
                np.full(selected.sum(), np.radians(frame["measured_tilt_deg"])),
                mu[selected], np.arctan2(unit[selected] @ second, unit[selected] @ tangent),
                np.full(selected.sum(), repeat), np.full(selected.sum(), nominal_tilt),
            ]))
    return np.concatenate(samples) if samples else np.empty((0, 5)), records


def fit_scalar_b(samples: np.ndarray) -> dict:
    if not len(samples):
        raise ValueError("No selected surface exits.")
    alpha, mu, azimuth = samples[:, :3].T
    if not np.isfinite(samples[:, :3]).all() or np.any((mu <= 0) | (mu > 1)):
        raise ValueError("Likelihood inputs must be finite outward directions; no clipping is allowed.")
    if np.max(np.abs(np.sin(alpha))) < 1e-4:
        return {"b": None, "identifiable": False, "n": len(samples), "nll": float(-np.log(mu / np.pi).mean())}
    a = np.sin(alpha) * np.sqrt(np.maximum(1 - mu * mu, 0)) * np.cos(azimuth)
    bound = .999 / np.max(np.abs(np.sin(alpha)))
    result = minimize_scalar(lambda b: -np.log1p(b * a).mean(), bounds=(-bound, bound), method="bounded")
    b = float(result.x)
    information = float(np.sum((a / (1 + b * a)) ** 2))
    return {"b": b, "identifiable": True, "n": len(samples),
            "nll": float(-np.log(mu / np.pi).mean() + result.fun),
            "conditional_se": float(1 / np.sqrt(information)),
            "max_abs_b_sin_alpha": float(abs(b) * np.max(np.abs(np.sin(alpha)))),
            "optimizer_success": bool(result.success)}


def _rr_nll(samples, b):
    alpha, mu, azimuth = samples[:, :3].T
    bracket = 1 + b * np.sin(alpha) * np.sqrt(np.maximum(1 - mu * mu, 0)) * np.cos(azimuth)
    if np.any(bracket <= 0):
        return math.inf
    return float(-np.mean(np.log(mu / np.pi) + np.log(bracket)))


def _histogram(samples):
    counts, _, _ = np.histogram2d(samples[:, 1], samples[:, 2], bins=(12, 24), range=((0, 1), (-np.pi, np.pi)))
    # Fixed half-count prior, selected before examining held-out scores.
    return (counts + .5) / (len(samples) + .5 * counts.size)


def _hist_nll(samples, mass):
    mi = np.minimum((samples[:, 1] * 12).astype(int), 11)
    ai = np.clip(((samples[:, 2] + np.pi) / (2 * np.pi) * 24).astype(int), 0, 23)
    return float(-np.log(mass[mi, ai] / (2 * np.pi / 288)).mean())


def fit_sine_b(samples, max_alpha):
    """Fit B=b0+b1*sin(alpha), constrained over the continuous declared domain."""
    s = np.sin(samples[:, 0])
    a = s * np.sqrt(np.maximum(1 - samples[:, 1] ** 2, 0)) * np.cos(samples[:, 2])
    smax = np.sin(max_alpha)
    def extrema(beta):
        candidates = [0., smax]
        if abs(beta[1]) > 1e-15:
            vertex = -beta[0] / (2 * beta[1])
            if 0 < vertex < smax:
                candidates.append(vertex)
        x = np.asarray(candidates)
        return float(np.max(np.abs(beta[0] * x + beta[1] * x * x)))
    def objective(beta):
        bracket = 1 + (beta[0] + beta[1] * s) * a
        return -np.log(bracket).mean() if np.all(bracket > 0) else 1e10
    result = minimize(objective, [0, 0], method="SLSQP", constraints=[{
        "type": "ineq", "fun": lambda beta: .999 - extrema(beta)}],
        options={"ftol": 1e-11, "maxiter": 200})
    if not result.success or extrema(result.x) > .999 + 1e-7:
        raise ValueError(f"Sine fit failed its optimization/domain gate: {result.message}")
    return {"b0": float(result.x[0]), "b1": float(result.x[1]),
            "optimizer_success": bool(result.success),
            "pooled_nll": float(result.fun - np.log(samples[:, 1] / np.pi).mean()),
            "domain_max_deg": float(np.degrees(max_alpha)),
            "max_abs_b_sin_alpha": extrema(result.x),
            "status": "candidate_only_not_renderer_calibration"}


def analyze_samples(samples: np.ndarray) -> dict:
    if not len(samples):
        raise ValueError("No eligible surface exits; inspect landing-patch diagnostics.")
    rows, folds = [], []
    for nominal in sorted(np.unique(samples[:, 4])):
        subset = samples[samples[:, 4] == nominal]
        fit = fit_scalar_b(subset)
        run_b = [fit_scalar_b(subset[subset[:, 3] == r])["b"] for r in np.unique(subset[:, 3])]
        run_b = [b for b in run_b if b is not None]
        bands = []
        phi_deg = np.degrees(np.arccos(subset[:, 1]))
        for lo, hi in ((15, 30), (30, 45), (45, 60), (60, 75)):
            part = subset[(phi_deg >= lo) & (phi_deg < hi)]
            if len(part) >= 100:
                bands.append({"polar_band_deg": [lo, hi], **fit_scalar_b(part)})
        az = subset[:, 2]
        a1 = float(2 * abs(np.mean(np.exp(1j * az))))
        a2 = float(2 * abs(np.mean(np.exp(2j * az))))
        usable = [b for b in bands if b["b"] is not None]
        heterogeneity = None
        if len(usable) > 1:
            weights = np.array([1 / b["conditional_se"] ** 2 for b in usable])
            estimates = np.array([b["b"] for b in usable])
            mean = np.average(estimates, weights=weights)
            heterogeneity = float(np.sum(weights * (estimates - mean) ** 2) / (len(usable) - 1))
        reasons = []
        if not fit["identifiable"]:
            reasons.append("B unidentifiable at normal incidence")
        else:
            if fit["max_abs_b_sin_alpha"] >= .95:
                reasons.append("near positivity boundary")
            if heterogeneity is not None and heterogeneity > 25:
                reasons.append("polar bands disagree under one scalar B")
            if a2 > a1:
                reasons.append("second harmonic exceeds first")
            if usable and any(b["b"] - 1.96 * b["conditional_se"] > 0 for b in usable) and any(b["b"] + 1.96 * b["conditional_se"] < 0 for b in usable):
                reasons.append("polar bands require opposite signs")
        caution = (nominal < 10 or (heterogeneity is not None and heterogeneity > 2.5)
                   or (len(run_b) > 1 and np.std(run_b, ddof=1) > .15))
        rows.append({"nominal_tilt_deg": float(nominal), **fit,
                     "run_b_range": [min(run_b), max(run_b)] if run_b else None,
                     "first_harmonic": a1, "second_harmonic": a2,
                     "polar_reduced_chi_square": heterogeneity, "polar_band_fits": bands,
                     "screening_decision": "DO_NOT_USE" if reasons else ("PROVISIONAL" if caution else "PASS_SCREENING"),
                     "reasons": reasons,
                     "polar_mean_mu2": float(np.mean(subset[:, 1] ** 2)),
                     "lambertian_expected_mu2": .5})
        for repeat in np.unique(subset[:, 3]):
            train, test = subset[subset[:, 3] != repeat], subset[subset[:, 3] == repeat]
            if not len(train):
                continue
            train_fit = fit_scalar_b(train)
            folds.append({"nominal_tilt_deg": float(nominal), "heldout_repeat": int(repeat),
                          "test_count": len(test), "training_count": len(train),
                          "lambertian_nll": _rr_nll(test, 0),
                          "scalar_b_nll": _rr_nll(test, train_fit["b"] or 0),
                          "angular_histogram_nll": _hist_nll(test, _histogram(train))})

    # Select no model using these scores: report candidates and held-out evidence.
    max_alpha = float(samples[:, 0].max())
    smooth = fit_sine_b(samples, max_alpha)
    model_folds = []
    for mode, column in (("repeat", 3), ("tilt", 4)):
        for heldout in sorted(np.unique(samples[:, column])):
            train = samples[samples[:, column] != heldout]
            test = samples[samples[:, column] == heldout]
            if not len(train):
                continue
            constant = fit_scalar_b(train)
            # Constrain constant on the test domain too, without using outcomes.
            bound = .999 / np.sin(max_alpha) if max_alpha >= 1e-4 else 0.
            constant_b = float(np.clip(constant["b"] or 0, -bound, bound))
            fitted = fit_sine_b(train, max_alpha)
            law = fitted["b0"] + fitted["b1"] * np.sin(test[:, 0])
            model_folds.append({"heldout_unit": mode, "heldout_value": float(heldout),
                "training_count": len(train), "test_count": len(test),
                "interpolation": bool(mode == "tilt" and train[:, 4].min() < heldout < train[:, 4].max()),
                "constant_b": constant_b, "sine_b0": fitted["b0"], "sine_b1": fitted["b1"],
                "lambertian_nll": _rr_nll(test, 0),
                "constant_nll": _rr_nll(test, constant_b), "sine_nll": _rr_nll(test, law)})
    return {"per_tilt": rows, "repeat_holdouts": folds,
            "pooled_constant": fit_scalar_b(samples), "pooled_sine_candidate": smooth,
            "model_holdouts": model_folds,
            "selected_exit_count": len(samples),
            "scope": "same-entrance-surface exits from retained finite-box simulations; conditional on geometric selection",
            "histogram": {"mu_bins": 12, "azimuth_bins": 24, "prior_count_per_bin": .5,
                          "measure": "dOmega = dmu dvarphi", "heldout_unit": "source repeat"},
            "automatic_renderer_change": False}


def run_reanalysis(cas_root: Path, output: Path, plane_tolerance_nm: float = 1.0):
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(cas_root.rglob("*summary.dat"))
    if not sources:
        raise ValueError(f"No summary.dat exports found under {cas_root}")
    all_samples, all_blocks, manifests, seen_hashes, seen_ids = [], [], [], set(), set()
    for index, source in enumerate(sources, 1):
        run_id = source.parent.name
        if run_id in seen_ids:
            raise ValueError(f"Repeated run id {run_id}: choose one canonical campaign root.")
        seen_ids.add(run_id)
        nominal, repeat = _parse_run_id(run_id)
        cache = output / "compact" / (run_id + ".npz")
        meta = compact_summary(source, cache)
        if meta["sha256"] in seen_hashes:
            raise ValueError(f"Duplicate source content must not count as an independent repeat: {source}")
        seen_hashes.add(meta["sha256"])
        sample, blocks = select_surface_exits(cache, nominal, repeat, plane_tolerance_nm)
        all_samples.append(sample)
        all_blocks.extend(blocks)
        manifests.append({"run_id": run_id, "source": str(source.resolve()),
                          "source_bytes": meta["source_bytes"], "source_sha256": meta["sha256"],
                          "compact_bytes": cache.stat().st_size, "compact": str(cache.resolve()),
                          "histories": meta["trajectory_count"], "collision_rows": meta["collision_rows_read"]})
        print(f"[{index}/{len(sources)}] {run_id}: {len(sample)} selected exits, {cache.stat().st_size / 1e6:.2f} MB compact", flush=True)
    samples = np.concatenate(all_samples)
    np.savez_compressed(output / "selected_surface_exits.npz", samples=samples,
                        columns=np.array(["alpha_rad", "cos_phi", "varphi_rad", "repeat", "nominal_tilt_deg"]))
    _write_json(output / "source_manifest.json", manifests)
    _write_json(output / "surface_selection.json", all_blocks)
    report = analyze_samples(samples)
    _write_json(output / "surface_b_report.json", report)
    _write_json(output / "analysis_provenance.json", {
        "script": str(Path(__file__).resolve()), "script_sha256": _sha256_file(__file__),
        "numpy_version": np.__version__, "plane_tolerance_nm": plane_tolerance_nm,
        "source_manifest_sha256": _sha256_file(output / "source_manifest.json"),
        "renderer_changed": False, "input_policy": "same_entry_surface_all_exit_classes_v1",
    })
    np.savez_compressed(output / "angular_reference.npz",
                        tilts=np.array(sorted(np.unique(samples[:, 4]))),
                        probability_mass=np.array([_histogram(samples[samples[:, 4] == t]) for t in sorted(np.unique(samples[:, 4]))]),
                        mu_edges=np.linspace(0, 1, 13), azimuth_edges=np.linspace(-np.pi, np.pi, 25))
    lines = ["# Surface-conditioned CASINO B reanalysis", "",
             "Existing summaries were streamed; no simulation was launched and original files were preserved.", "",
             "Each scan point has its own plane. Exits must lie on that entrance plane, point outward, and exceed 50 eV, regardless of the global Backscattered/Transmitted label. Off-nominal landing patches are excluded and listed separately.", "",
             f"Plane residual tolerance: {plane_tolerance_nm:g} nm. Screening is conditional on this finite-specimen surface selection; it is not full physical validation.", "",
             "| Tilt | Selected exits | B | Polar heterogeneity | Screening |",
             "|---:|---:|---:|---:|---|"]
    for row in report["per_tilt"]:
        b = "unidentifiable" if row["b"] is None else f"{row['b']:.6f}"
        h = row["polar_reduced_chi_square"]
        lines.append(f"| {row['nominal_tilt_deg']:g} | {row['n']} | {b} | {'NA' if h is None else f'{h:.2f}'} | {row['screening_decision']} |")
    lines += ["", "## Repeat-held-out angular scores", "",
              "Mean NLL per solid angle, averaged equally across repeat folds within each tilt. Lower is better. Histogram bins/prior were fixed in advance. These are same-tilt repeat holdouts, not unseen-tilt validation.", "",
              "| Tilt | B=0 | One fitted B per tilt | Angular histogram |", "|---:|---:|---:|---:|"]
    for tilt in sorted(np.unique(samples[:, 4])):
        folds = [r for r in report["repeat_holdouts"] if r["nominal_tilt_deg"] == tilt]
        if folds:
            scores = [np.mean([r[k] for r in folds]) for k in ("lambertian_nll", "scalar_b_nll", "angular_histogram_nll")]
            lines.append(f"| {tilt:g} | {scores[0]:.6f} | {scores[1]:.6f} | {scores[2]:.6f} |")
    lines += ["", "## Pooled constant and tilt-dependent candidates", "",
              f"Pooled constant B: {report['pooled_constant']['b']:.6f}. Sine candidate: B(alpha) = {report['pooled_sine_candidate']['b0']:.6f} + {report['pooled_sine_candidate']['b1']:.6f} sin(alpha). Neither is automatically adopted.", "",
              "| Holdout grouping | B=0 NLL | Constant NLL | Sine NLL |", "|---|---:|---:|---:|"]
    for mode in ("repeat", "tilt"):
        group = [f for f in report["model_holdouts"] if f["heldout_unit"] == mode]
        if not group:
            lines.append(f"| {mode}: insufficient independent groups | NA | NA | NA |")
            continue
        score = [np.mean([f[k] for f in group]) for k in ("lambertian_nll", "constant_nll", "sine_nll")]
        lines.append(f"| Equal-weight {mode} folds | {score[0]:.6f} | {score[1]:.6f} | {score[2]:.6f} |")
    lines += ["", "The tilt-holdout JSON distinguishes interpolation from endpoint extrapolation. Lower average NLL does not override failed per-tilt adequacy checks."]
    raw_bytes = sum(m["source_bytes"] for m in manifests)
    compact_bytes = sum(m["compact_bytes"] for m in manifests)
    lines += ["", "## Storage and interpretation", "",
              f"{len(manifests)} source summaries: {raw_bytes:,} bytes. Compressed entrance/final-row caches: {compact_bytes:,} bytes ({raw_bytes / compact_bytes:.1f} times smaller). Intermediate collisions are not needed for this angular analysis.", "",
              "The angular reference is an analysis artifact, not a renderer calibration. Its final all-run table must not be evaluated as held-out evidence; use the fold scores above. No extrapolation, energy transfer or finite-size correction is established here.", "",
              "Detailed source hashes, recovered transmitted exits, off-face losses, frames, per-band fits and scores are in the adjacent JSON files. Existing frozen fits and datasets were not replaced."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cas-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plane-tolerance-nm", type=float, default=1.)
    args = parser.parse_args()
    if args.plane_tolerance_nm <= 0 or not math.isfinite(args.plane_tolerance_nm):
        parser.error("--plane-tolerance-nm must be finite and positive")
    run_reanalysis(args.cas_root, args.output_dir, args.plane_tolerance_nm)


if __name__ == "__main__":
    main()
