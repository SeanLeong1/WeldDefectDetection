from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import default_slice_runtime, load_config_bundle
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..geometry.defect_injection import parse_defect_plan, validate_defect_plan
from ..renderer.scan_engine import ScanEngine
from .casino_adapter import run_casino_probe
from .casino_calibration import run_casino_calibration
from .regime_validation import run_regime_sweep_validation
from .standards import evaluate_standards_report, load_standards_defaults


def _beam_overrides_for_regime(regime: str) -> dict[str, float]:
    if regime == "low":
        return {
            "accelerating_voltage_kv": 45.0,
            "beam_current_ma": 90.0,
            "focusing_current_norm": 0.35,
            "travel_speed_mm_min": 1600.0,
        }
    if regime == "high":
        return {
            "accelerating_voltage_kv": 150.0,
            "beam_current_ma": 650.0,
            "focusing_current_norm": 0.85,
            "travel_speed_mm_min": 450.0,
        }
    return {
        "accelerating_voltage_kv": 90.0,
        "beam_current_ma": 320.0,
        "focusing_current_norm": 0.60,
        "travel_speed_mm_min": 900.0,
    }


def _write_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "regime",
                "defect_type",
                "size_mm",
                "depth_mm",
                "threshold_mm",
                "required_for_level_b_eval",
                "detected",
                "pass_detection_requirement",
                "compo_contrast",
                "topo_contrast",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("regime"),
                    row.get("defect_type"),
                    row.get("size_mm"),
                    row.get("depth_mm"),
                    row.get("threshold_mm"),
                    row.get("required_for_level_b_eval"),
                    row.get("detected"),
                    row.get("pass_detection_requirement"),
                    row.get("compo_contrast"),
                    row.get("topo_contrast"),
                ]
            )
    return str(path)


def _save_threshold_plot(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "skipped", "reason": "No threshold rows to plot", "path": None}
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception as exc:
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}", "path": None}

    sizes = np.array([float(r.get("size_mm", 0.0)) for r in rows], dtype=np.float64)
    detected = np.array([1 if r.get("detected") else 0 for r in rows], dtype=np.float64)
    thresholds = np.array([float(r.get("threshold_mm", 0.0)) for r in rows], dtype=np.float64)

    fig = Figure(figsize=(7, 4))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.scatter(sizes, detected, c=np.where(detected > 0.5, "tab:green", "tab:red"), alpha=0.75)
    if np.any(np.isfinite(thresholds)):
        ax.axvline(float(np.nanmean(thresholds)), linestyle="--", color="tab:blue", label="Mean threshold")
    ax.set_xlabel("Defect size (mm)")
    ax.set_ylabel("Detected (1/0)")
    ax.set_title("Near-threshold detection proxy")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    if np.any(np.isfinite(thresholds)):
        ax.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return {"status": "ok", "reason": None, "path": str(path)}


def _run_threshold_slices(
    *,
    config_dir: str | Path,
    calibration_path: str | None,
    t_mm: float,
    s_mm: float,
    contrast_threshold: float,
    topo_contrast_threshold: float | None = None,
) -> dict[str, Any]:
    cfg = load_config_bundle(config_dir)
    std_defaults = load_standards_defaults(config_dir)
    pore_thr = min(0.4 * s_mm, 3.0, max(std_defaults.pore_diameter_floor_mm, 1e-6))
    undercut_thr = min(0.05 * t_mm, 0.5)
    plans = [
        f"keyhole_pore(diameter_mm={0.8 * pore_thr:.3f},depth_mm=0.35)",
        f"keyhole_pore(diameter_mm={1.0 * pore_thr:.3f},depth_mm=0.35)",
        f"keyhole_pore(diameter_mm={1.2 * pore_thr:.3f},depth_mm=0.35)",
        f"intrinsic_pore(diameter_mm={0.8 * pore_thr:.3f},depth_mm=0.35)",
        f"intrinsic_pore(diameter_mm={1.0 * pore_thr:.3f},depth_mm=0.35)",
        f"undercut(groove_depth_mm={0.8 * undercut_thr:.3f},toe_angle_deg=34.0)",
        f"undercut(groove_depth_mm={1.0 * undercut_thr:.3f},toe_angle_deg=34.0)",
        f"undercut(groove_depth_mm={1.2 * undercut_thr:.3f},toe_angle_deg=34.0)",
        "crack_solidification(length_mm=0.8,width_mm=0.08,depth_mm=0.30)",
        "crack_liquidation(length_mm=0.8,width_mm=0.08,depth_mm=0.22)",
        "lack_of_fusion(extent_mm=0.8,depth_mm=0.4,irregularity=0.35)",
    ]

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    controlled_sample = sample_parameters(
        cfg,
        seed=61000,
        max_defects=0,
        randomize_defect_location=False,
    )
    controlled_slice = default_slice_runtime(cfg.slice)
    for regime in ("low", "mid", "high"):
        for plan_text in plans:
            defects = parse_defect_plan(plan_text)
            validate_defect_plan(cfg.defect, defects)
            beam = replace(controlled_sample.beam, **_beam_overrides_for_regime(regime))
            beam = replace(
                beam,
                e0_kev=beam.accelerating_voltage_kv,
                power_w=(beam.accelerating_voltage_kv * 1000.0) * (beam.beam_current_ma / 1000.0),
            )
            sampled = replace(
                controlled_sample,
                beam=beam,
                defects=defects,
                slice=controlled_slice,
                randomize_defect_location=False,
            )
            render = ScanEngine(image_size=128, calibration_path=calibration_path).run(sampled)
            std = evaluate_standards_report(
                defects=render.defects,
                compo=render.channels.compo,
                topo=render.channels.topo,
                t_mm=t_mm,
                s_mm=s_mm,
                pore_floor_mm=std_defaults.pore_diameter_floor_mm,
                contrast_threshold=contrast_threshold,
                topo_contrast_threshold=topo_contrast_threshold,
                confidence_z=std_defaults.confidence_z,
            )
            for row in std["rows"]:
                out_row = dict(row)
                out_row["regime"] = regime
                rows.append(out_row)
                grouped[regime].append(out_row)

    summary = {}
    for regime, g_rows in grouped.items():
        req = [r for r in g_rows if r["required_for_level_b_eval"]]
        req_n = len(req)
        req_k = int(sum(1 for r in req if r["detected"]))
        summary[regime] = {
            "required_count": req_n,
            "detected_required_count": req_k,
            "rate": float(req_k / req_n) if req_n else 0.0,
        }
    return {
        "status": "ok",
        "rows": rows,
        "summary_by_regime": summary,
        "settings": {
            "t_mm": float(t_mm),
            "s_mm": float(s_mm),
            "contrast_threshold": float(contrast_threshold),
            "compo_contrast_threshold": float(contrast_threshold),
            "topo_contrast_threshold": float(
                contrast_threshold
                if topo_contrast_threshold is None
                else topo_contrast_threshold
            ),
            "controlled_seed": int(controlled_sample.seed),
            "randomize_defect_location": False,
            "slice": {
                "voxel_depth_mm": float(controlled_slice.voxel_depth_mm),
                "slice_offset_mm": float(controlled_slice.slice_offset_mm),
                "slice_tilt_deg": float(controlled_slice.slice_tilt_deg),
                "slice_rotation_deg": float(controlled_slice.slice_rotation_deg),
            },
        },
    }


def run_casino_comparison(
    *,
    config_dir: str | Path = "config",
    casino_exe: str | None = None,
    run_root: str | Path | None = None,
    run_budget: str = "medium",
    calibrate: bool = False,
    calibration_cases: int = 18,
    timeout_s: float = 240.0,
    regime_cases: int = 12,
    t_mm: float | None = None,
    s_mm: float | None = None,
    contrast_threshold: float | None = None,
    topo_contrast_threshold: float | None = None,
) -> dict:
    """
    CASINO-backed validation and standards-aligned reporting.

    The CASINO executable is expected to be a CLI adapter that accepts:
      --input <case.json> --output <casino_output.json>
    """
    cfg_dir = load_config_bundle(config_dir).config_dir
    standards = load_standards_defaults(cfg_dir)
    t_val = float(t_mm if t_mm is not None else standards.thickness_t_mm)
    s_val = float(s_mm if s_mm is not None else standards.throat_s_mm)
    contrast_thr = float(
        contrast_threshold
        if contrast_threshold is not None
        else standards.contrast_detection_threshold
    )
    topo_contrast_thr = float(
        topo_contrast_threshold
        if topo_contrast_threshold is not None
        else (
            contrast_threshold
            if contrast_threshold is not None
            else standards.topo_contrast_detection_threshold
        )
    )
    if run_root is None:
        run_root = Path("data") / "output" / "validation_reports"
    report_root = Path(run_root).resolve() / f"casino_validation_{int(time.time())}"
    report_root.mkdir(parents=True, exist_ok=True)

    budget_cases = {"fast": 6, "medium": 12, "high": 24}
    regime_cases_effective = int(budget_cases.get(run_budget, regime_cases))

    probe = run_casino_probe(casino_exe=casino_exe)
    calibration_result = {
        "status": "skipped",
        "message": "Calibration disabled",
        "calibration_file": None,
    }
    if calibrate:
        calibration_result = run_casino_calibration(
            config_dir=cfg_dir,
            casino_exe=casino_exe,
            run_root=report_root / "calibration",
            num_cases=max(1, calibration_cases),
            timeout_s=timeout_s,
        )
    calibration_file = calibration_result.get("calibration_file")

    threshold_eval = _run_threshold_slices(
        config_dir=cfg_dir,
        calibration_path=calibration_file,
        t_mm=t_val,
        s_mm=s_val,
        contrast_threshold=contrast_thr,
        topo_contrast_threshold=topo_contrast_thr,
    )
    regime_eval = run_regime_sweep_validation(
        config_dir=cfg_dir,
        run_root=report_root / "regimes",
        cases_per_regime=regime_cases_effective,
        calibration_path=calibration_file,
        t_mm=t_val,
        s_mm=s_val,
        contrast_threshold=contrast_thr,
        topo_contrast_threshold=topo_contrast_thr,
    )

    threshold_csv = _write_threshold_csv(
        report_root / "threshold_slices.csv",
        threshold_eval["rows"],
    )
    threshold_plot = _save_threshold_plot(
        report_root / "threshold_slices.png",
        threshold_eval["rows"],
    )

    status = "ok"
    if not probe.get("available"):
        status = "partial"
    if calibration_result.get("status") in {"failed", "diagnostic"}:
        status = "partial"
    if regime_eval.get("status") != "ok":
        status = "partial"

    out = {
        "status": status,
        "casino_probe": probe,
        "calibration": calibration_result,
        "standards_threshold_slices": threshold_eval,
        "regime_validation": regime_eval,
        "settings": {
            "run_budget": run_budget,
            "regime_cases": regime_cases_effective,
            "t_mm": t_val,
            "s_mm": s_val,
            "contrast_threshold": contrast_thr,
            "compo_contrast_threshold": contrast_thr,
            "topo_contrast_threshold": topo_contrast_thr,
            "run_root": str(report_root),
        },
        "artifacts": {
            "threshold_csv": threshold_csv,
            "threshold_plot": threshold_plot,
            "regime_report": regime_eval.get("report_path"),
        },
    }
    report_json = report_root / "validation_report.json"
    report_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["artifacts"]["validation_report"] = str(report_json)
    return {
        **out,
        "relative_error_compo": None,
        "relative_error_topo": None,
    }
