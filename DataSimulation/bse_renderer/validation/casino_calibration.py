from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import load_config_bundle
from ..domain_randomisation.parameter_sampler import sample_parameters
from ..geometry.defect_injection import parse_defect_plan, validate_defect_plan
from ..renderer.calibration import save_renderer_calibration
from ..renderer.scan_engine import ScanEngine
from .casino_adapter import run_casino_case, run_casino_probe


_CALIBRATION_DEFECTS = [
    "keyhole_pore(diameter_mm=0.5,depth_mm=0.4)",
    "intrinsic_pore(diameter_mm=0.5,depth_mm=0.4)",
    "crack_solidification(length_mm=0.8,width_mm=0.08,depth_mm=0.3)",
    "lack_of_fusion(extent_mm=0.8,depth_mm=0.4,irregularity=0.3)",
    "undercut(groove_depth_mm=0.5,toe_angle_deg=34.0)",
]


def _regime_overrides(regime: str) -> dict[str, float]:
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


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("A calibration fit requires at least three finite paired cases.")
    if np.unique(x).size < 3 or float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        raise ValueError("Calibration inputs require three distinct signal levels.")
    coeff = np.polyfit(x, y, deg=1)
    return float(coeff[0]), float(coeff[1])


def run_casino_calibration(
    *,
    config_dir: str | Path,
    casino_exe: str | None = None,
    run_root: str | Path,
    num_cases: int = 18,
    timeout_s: float = 240.0,
) -> dict[str, Any]:
    probe = run_casino_probe(casino_exe=casino_exe)
    run_base = Path(run_root).resolve()
    if run_base.exists() and any(run_base.iterdir()):
        raise FileExistsError("Calibration requires a new empty run directory.")
    run_base.mkdir(parents=True, exist_ok=True)

    if not probe.get("available"):
        return {
            "status": "skipped",
            "probe": probe,
            "message": probe.get("message", "CASINO unavailable"),
            "calibration_file": None,
            "cases": [],
        }

    cfg = load_config_bundle(config_dir)
    regimes = ["low", "mid", "high"]
    rows: list[dict[str, Any]] = []
    x_compo: list[float] = []
    y_compo: list[float] = []
    x_topo: list[float] = []
    y_topo: list[float] = []
    cases_to_run = max(1, int(num_cases))

    for i in range(cases_to_run):
        regime = regimes[i % len(regimes)]
        defect_text = _CALIBRATION_DEFECTS[i % len(_CALIBRATION_DEFECTS)]
        sampled = sample_parameters(cfg, seed=12000 + i, max_defects=0)
        beam = replace(sampled.beam, **_regime_overrides(regime))
        beam = replace(
            beam,
            e0_kev=beam.accelerating_voltage_kv,
            power_w=(beam.accelerating_voltage_kv * 1000.0) * (beam.beam_current_ma / 1000.0),
        )
        defects = parse_defect_plan(defect_text)
        validate_defect_plan(cfg.defect, defects)
        sampled = replace(sampled, beam=beam, defects=defects)

        render = ScanEngine(image_size=128, render_model_mode="formula_only", emission_b_mode="lambertian").run(sampled)
        fast_compo = float(np.mean(render.channels.compo))
        fast_topo = float(np.mean(render.channels.topo))

        case_dir = run_base / "cases" / f"calib_{i:03d}"
        case_payload = {
            "seed": int(sampled.seed),
            "regime": regime,
            "beam": {
                "e0_kev": float(sampled.beam.e0_kev),
                "beam_current_ma": float(sampled.beam.beam_current_ma),
                "focusing_current_norm": float(sampled.beam.focusing_current_norm),
                "travel_speed_mm_min": float(sampled.beam.travel_speed_mm_min),
            },
            "material": {
                "name": sampled.material.name,
                "z": float(sampled.material.z),
                "rho": float(sampled.material.rho),
            },
            "defects": defects,
        }
        casino = run_casino_case(
            case=case_payload,
            casino_exe=casino_exe,
            run_dir=case_dir,
            timeout_s=timeout_s,
        )
        row: dict[str, Any] = {
            "index": i,
            "regime": regime,
            "defect_plan": defect_text,
            "fast_compo": fast_compo,
            "fast_topo": fast_topo,
            "casino_status": casino.get("status"),
            "casino_message": casino.get("message"),
            "work_dir": casino.get("work_dir"),
            "provenance": casino.get("provenance"),
        }
        if casino.get("status") == "ok" and casino.get("calibration_eligible") is True:
            cas_compo = float(casino["derived"]["compo"])
            cas_topo = float(casino["derived"]["topo"])
            row["casino_compo"] = cas_compo
            row["casino_topo"] = cas_topo
            x_compo.append(fast_compo)
            y_compo.append(cas_compo)
            x_topo.append(fast_topo)
            y_topo.append(cas_topo)
        rows.append(row)

    calibration_dir = run_base / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "calibration_cases.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    result = {"status": "failed", "probe": probe, "calibration_file": None,
              "num_requested_cases": cases_to_run, "num_successful_cases": len(x_compo), "cases": rows}
    try:
        if len(x_compo) < 6:
            raise ValueError("At least six independently generated external cases are required; mock/unknown cases are diagnostic only.")
        provenance_hashes = [(row.get("provenance") or {}).get("parsed_artifact_sha256")
                             for row in rows if row.get("casino_compo") is not None]
        if None in provenance_hashes or len(set(provenance_hashes)) != len(provenance_hashes):
            raise ValueError("Calibration cases require distinct hashed external source artifacts.")
        coefficients = {}
        quality = {}
        for name, xx, yy in (("compo", x_compo, y_compo), ("topo", x_topo, y_topo)):
            x, y = np.asarray(xx), np.asarray(yy)
            scale, bias = _fit_linear(x, y)
            residuals = []
            for i in range(len(x)):
                keep = np.arange(len(x)) != i
                a, b = _fit_linear(x[keep], y[keep])
                residuals.append(a*x[i] + b - y[i])
            nrmse = float(np.sqrt(np.mean(np.square(residuals))) / max(float(np.std(y)), 1e-8))
            if not np.isfinite(nrmse) or nrmse > 0.25 or scale <= 0:
                raise ValueError(f"{name}: held-out normalized RMSE {nrmse:.4g} or nonpositive scale fails the calibration gate.")
            coefficients[name] = {"scale": scale, "bias": bias}
            quality[name] = {"leave_one_case_out_nrmse": nrmse, "maximum_nrmse": 0.25,
                             "input_domain": [float(x.min()), float(x.max())]}
        path = save_renderer_calibration(
            calibration_dir / "renderer_calibration.json",
            compo_scale=coefficients["compo"]["scale"], compo_bias=coefficients["compo"]["bias"],
            topo_scale=coefficients["topo"]["scale"], topo_bias=coefficients["topo"]["bias"],
            metadata={"source": "casino_calibration", "backend": "external", "quality_passed": True,
                      "num_successful_cases": len(x_compo), "quality": quality,
                      "case_provenance": [row["provenance"] for row in rows if row.get("casino_compo") is not None]})
        result.update(status="ok", calibration_file=path, coefficients=coefficients, quality=quality,
                      message="External calibration passed held-out case checks.")
    except ValueError as exc:
        result["message"] = str(exc)
    return result
