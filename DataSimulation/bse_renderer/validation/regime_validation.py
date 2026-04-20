from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import load_config_bundle
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..geometry.defect_injection import parse_defect_plan, validate_defect_plan
from ..renderer.scan_engine import ScanEngine
from .standards import evaluate_standards_report, load_standards_defaults


_REGIMES = {
    "low": {
        "accelerating_voltage_kv": 45.0,
        "beam_current_ma": 90.0,
        "focusing_current_norm": 0.35,
        "travel_speed_mm_min": 1600.0,
    },
    "mid": {
        "accelerating_voltage_kv": 90.0,
        "beam_current_ma": 320.0,
        "focusing_current_norm": 0.60,
        "travel_speed_mm_min": 900.0,
    },
    "high": {
        "accelerating_voltage_kv": 150.0,
        "beam_current_ma": 650.0,
        "focusing_current_norm": 0.85,
        "travel_speed_mm_min": 450.0,
    },
}

_CRITICAL_DEFECT_PLANS = [
    "keyhole_pore(diameter_mm=0.5,depth_mm=0.45)",
    "intrinsic_pore(diameter_mm=0.5,depth_mm=0.45)",
    "crack_solidification(length_mm=1.0,width_mm=0.08,depth_mm=0.35)",
    "crack_liquidation(length_mm=1.0,width_mm=0.08,depth_mm=0.25)",
    "lack_of_fusion(extent_mm=1.0,depth_mm=0.5,irregularity=0.35)",
    "undercut(groove_depth_mm=0.5,toe_angle_deg=34.0)",
]


def _session_state(index: int, total: int, rng: np.random.Generator) -> dict[str, Any]:
    denom = max(total - 1, 1)
    progress = float(np.clip(index / denom, 0.0, 1.0))
    mismatch_amp = 0.02 + 0.06 * progress
    mismatch = {
        "A": float(rng.normal(0.0, mismatch_amp)),
        "B": float(rng.normal(0.0, mismatch_amp)),
        "C": float(rng.normal(0.0, mismatch_amp)),
        "D": float(rng.normal(0.0, mismatch_amp)),
    }
    return {
        "progress": progress,
        "thermal_baseline": float(0.01 * progress),
        "quadrant_mismatch": mismatch,
    }


def _beam_with_regime(sampled_beam, regime: str):
    ov = _REGIMES[regime]
    beam = replace(sampled_beam, **ov)
    return replace(
        beam,
        e0_kev=beam.accelerating_voltage_kv,
        power_w=(beam.accelerating_voltage_kv * 1000.0) * (beam.beam_current_ma / 1000.0),
    )


def run_regime_sweep_validation(
    *,
    config_dir: str | Path,
    run_root: str | Path,
    cases_per_regime: int = 12,
    calibration_path: str | None = None,
    t_mm: float | None = None,
    s_mm: float | None = None,
    contrast_threshold: float | None = None,
) -> dict[str, Any]:
    cfg = load_config_bundle(config_dir)
    standards = load_standards_defaults(config_dir)
    t_val = float(t_mm if t_mm is not None else standards.thickness_t_mm)
    s_val = float(s_mm if s_mm is not None else standards.throat_s_mm)
    contrast_thr = float(
        contrast_threshold
        if contrast_threshold is not None
        else standards.contrast_detection_threshold
    )
    base = Path(run_root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    regime_summary: dict[str, dict[str, Any]] = {}
    total_cases = max(1, int(cases_per_regime))
    t0 = time.time()

    for regime in ("low", "mid", "high"):
        rng = np.random.default_rng(4000 + len(rows))
        detection_pass_rates: list[float] = []
        compo_means: list[float] = []
        topo_means: list[float] = []
        regime_rows: list[dict[str, Any]] = []

        for i in range(total_cases):
            defect_text = _CRITICAL_DEFECT_PLANS[i % len(_CRITICAL_DEFECT_PLANS)]
            defects = parse_defect_plan(defect_text)
            validate_defect_plan(cfg.defect, defects)
            sampled = sample_parameters(cfg, seed=50000 + i + 100 * len(rows), max_defects=0)
            sampled = replace(
                sampled,
                beam=_beam_with_regime(sampled.beam, regime),
                defects=defects,
            )
            session = _session_state(i, total_cases, rng)
            run_start = time.time()
            render = ScanEngine(image_size=128, calibration_path=calibration_path).run(
                sampled,
                session_state=session,
            )
            runtime_s = float(time.time() - run_start)
            std = evaluate_standards_report(
                defects=render.defects,
                compo=render.channels.compo,
                topo=render.channels.topo,
                t_mm=t_val,
                s_mm=s_val,
                pore_floor_mm=standards.pore_diameter_floor_mm,
                contrast_threshold=contrast_thr,
                confidence_z=standards.confidence_z,
            )
            req_rows = [r for r in std["rows"] if r["required_for_level_b_eval"]]
            pass_rate = float(
                np.mean([1.0 if r["pass_detection_requirement"] else 0.0 for r in req_rows])
            ) if req_rows else 1.0
            detection_pass_rates.append(pass_rate)
            compo_means.append(float(np.mean(render.channels.compo)))
            topo_means.append(float(np.mean(render.channels.topo)))
            case_row = {
                "regime": regime,
                "index": i,
                "defect_plan": defect_text,
                "runtime_s": runtime_s,
                "session_progress": float(session["progress"]),
                "compo_mean": float(np.mean(render.channels.compo)),
                "topo_mean": float(np.mean(render.channels.topo)),
                "standards": std,
            }
            rows.append(case_row)
            regime_rows.append(case_row)

        regime_summary[regime] = {
            "num_cases": len(regime_rows),
            "mean_runtime_s": float(np.mean([r["runtime_s"] for r in regime_rows])),
            "mean_compo": float(np.mean(compo_means)) if compo_means else 0.0,
            "mean_topo": float(np.mean(topo_means)) if topo_means else 0.0,
            "mean_pass_rate_required_defects": float(np.mean(detection_pass_rates))
            if detection_pass_rates else 0.0,
        }

    elapsed = float(time.time() - t0)
    throughput = float(len(rows) / elapsed) if elapsed > 0 else 0.0
    payload = {
        "status": "ok",
        "settings": {
            "cases_per_regime": total_cases,
            "t_mm": t_val,
            "s_mm": s_val,
            "contrast_threshold": contrast_thr,
            "calibration_path": calibration_path,
        },
        "regime_summary": regime_summary,
        "operational_metrics": {
            "total_cases": len(rows),
            "elapsed_s": elapsed,
            "throughput_cases_per_s": throughput,
        },
        "cases": rows,
    }
    out_path = base / "regime_sweep_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_path"] = str(out_path)
    return payload
