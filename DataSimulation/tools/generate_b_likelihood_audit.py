from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

DATA_SIMULATION_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_SIMULATION_ROOT))

from bse_renderer.validation.casino_emission_fit import (  # noqa: E402
    _domain_feasible_interval,
    _pooled_negative_log_likelihood,
    _pooled_nll_derivative,
    _pooled_nll_second_derivative,
)


def _load_emission_components(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cos_phi: list[float] = []
    alpha_rad: list[float] = []
    phi_rad: list[float] = []
    varphi_rad: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cos_phi.append(float(row["cos_phi"]))
            alpha_rad.append(math.radians(float(row["tilt_deg"])))
            phi_rad.append(float(row["phi_rad"]))
            varphi_rad.append(float(row["varphi_rad"]))
    if not cos_phi:
        raise ValueError(f"No exit samples found in {path}")
    cos_values = np.asarray(cos_phi, dtype=np.float64)
    alpha_values = np.asarray(alpha_rad, dtype=np.float64)
    phi_values = np.asarray(phi_rad, dtype=np.float64)
    varphi_values = np.asarray(varphi_rad, dtype=np.float64)
    asymmetry = np.sin(alpha_values) * np.sin(phi_values) * np.cos(varphi_values)
    return cos_values, asymmetry, alpha_values, varphi_values


def build_audit(exit_samples_csv: Path, fit_report_json: Path) -> dict[str, object]:
    cos_phi, asymmetry, alpha_rad, varphi_rad = _load_emission_components(exit_samples_csv)
    report = json.loads(fit_report_json.read_text(encoding="utf-8"))
    fit = report["fit"]["global_pooled"]
    fitted_b = float(fit["fitted_b"])
    sample_count = int(asymmetry.size)

    nll_zero = float(_pooled_negative_log_likelihood(cos_phi, asymmetry, 0.0))
    nll_fit = float(_pooled_negative_log_likelihood(cos_phi, asymmetry, fitted_b))
    score_zero = float(_pooled_nll_derivative(asymmetry, 0.0))
    score_fit = float(_pooled_nll_derivative(asymmetry, fitted_b))
    curvature_fit = float(_pooled_nll_second_derivative(asymmetry, fitted_b))
    standard_error = float(math.sqrt(1.0 / (sample_count * curvature_fit)))
    domain_lower, domain_upper = _domain_feasible_interval(np.degrees(alpha_rad))

    reference_values = sorted(
        {
            float(max(domain_lower * 0.999, -1.0)),
            -0.5,
            fitted_b,
            0.0,
            0.5,
            float(min(domain_upper * 0.999, 1.0)),
        }
    )
    objective_points = [
        {
            "b": value,
            "mean_negative_log_likelihood": float(
                _pooled_negative_log_likelihood(cos_phi, asymmetry, value)
            ),
            "derivative": float(_pooled_nll_derivative(asymmetry, value)),
            "second_derivative": float(_pooled_nll_second_derivative(asymmetry, value)),
            "minimum_bracket": float(np.min(1.0 + value * asymmetry)),
        }
        for value in reference_values
    ]

    mean_asymmetry = float(np.mean(asymmetry))
    return {
        "schema_version": 1,
        "sources": {
            "exit_samples_csv": str(exit_samples_csv.resolve()),
            "fit_report_json": str(fit_report_json.resolve()),
        },
        "model": {
            "density": "p_i(B) = cos(phi_i)/pi * (1 + B A_i)",
            "asymmetry_factor": "A_i = sin(alpha_i) sin(phi_i) cos(varphi_i)",
            "mean_nll": "J(B) = -mean(log(cos(phi_i)/pi) + log(1 + B A_i))",
            "derivative": "J'(B) = -mean(A_i/(1 + B A_i))",
            "second_derivative": "J''(B) = mean(A_i^2/(1 + B A_i)^2) >= 0",
            "conditioning": (
                "This is the directional likelihood conditional on an electron being backscattered. "
                "It does not fit backscatter yield, transmitted counts, or exit energy."
            ),
            "zero_tilt_identifiability": (
                "At alpha=0, A_i=0 for every exit, so zero-tilt exits add no information about B."
            ),
        },
        "normalization": {
            "solid_angle_element": "dOmega = sin(phi) dphi dvarphi",
            "baseline_integral": "integral cos(phi)/pi dOmega = 1",
            "asymmetry_integral": "integral cos(phi) sin(phi) cos(varphi) dOmega = 0",
            "conclusion": "The density is normalized for every feasible B; no B-dependent normalizer is omitted.",
        },
        "data": {
            "sample_count": sample_count,
            "mean_asymmetry_factor": mean_asymmetry,
            "fraction_negative_asymmetry": float(np.mean(asymmetry < 0.0)),
            "fraction_positive_asymmetry": float(np.mean(asymmetry > 0.0)),
            "mean_cos_varphi": float(np.mean(np.cos(varphi_rad))),
        },
        "fit": {
            "fitted_b": fitted_b,
            "solution_method": fit["solution_method"],
            "mean_nll_at_zero": nll_zero,
            "mean_nll_at_fit": nll_fit,
            "mean_nll_improvement_vs_zero": float(nll_zero - nll_fit),
            "total_log_likelihood_improvement_vs_zero": float(sample_count * (nll_zero - nll_fit)),
            "twice_log_likelihood_improvement_vs_zero": float(2.0 * sample_count * (nll_zero - nll_fit)),
            "nll_derivative_at_zero": score_zero,
            "nll_derivative_at_fit": score_fit,
            "mean_nll_second_derivative_at_fit": curvature_fit,
            "trajectory_independence_approx_standard_error": standard_error,
            "trajectory_independence_approx_95pct_ci": [
                float(fitted_b - 1.96 * standard_error),
                float(fitted_b + 1.96 * standard_error),
            ],
            "equal_source_run_weight_fitted_b": float(report["fit"]["global_equal_run"]["fitted_b"]),
        },
        "feasibility": {
            "condition": "1 + B A_i > 0, with global sufficient bound |B| <= 1/max|sin(alpha)|",
            "domain_interval": [float(domain_lower), float(domain_upper)],
            "solver_interval": [
                float(fit["solver_interval"]["lower"]),
                float(fit["solver_interval"]["upper"]),
            ],
            "observed_minimum_bracket_at_fit": float(np.min(1.0 + fitted_b * asymmetry)),
            "all_angle_minimum_bracket_at_fit": float(fit["domain_min_density_margin"]),
        },
        "sign_explanation": {
            "identity_at_zero": "J'(0) = -mean(A)",
            "observed_mean_a": mean_asymmetry,
            "observed_derivative_at_zero": score_zero,
            "conclusion": (
                "The positive NLL derivative at B=0 means moving toward negative B lowers NLL. "
                "Under this frame, negative B increases density for exits with cos(varphi)<0, "
                "opposite the projected into-specimen beam direction."
            ),
            "coordinate_dependence": (
                "Reversing the in-plane reference t1 shifts varphi by pi and reverses the fitted B sign. "
                "The sign is meaningful only together with the stated azimuth convention."
            ),
        },
        "objective_points": objective_points,
        "uncertainty_warning": (
            "The curvature standard error treats all exit trajectories as independent. "
            "CASINO exits are nested in scan points and source runs, so run-level or clustered "
            "uncertainty is required before using this interval as final inferential evidence."
        ),
    }


def write_audit(audit: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "likelihood_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    data = audit["data"]
    fit = audit["fit"]
    feasibility = audit["feasibility"]
    points = audit["objective_points"]
    lines = [
        "# Stage 3: constant-B likelihood audit",
        "",
        "## Model and normalization",
        "",
        "For trajectory `i`, define `A_i = sin(alpha_i) sin(phi_i) cos(varphi_i)`. Then",
        "",
        "`p_i(B) = cos(phi_i)/pi * (1 + B A_i)`.",
        "",
        "With `dOmega = sin(phi) dphi dvarphi`, the Lambertian term integrates to one and the",
        "`cos(varphi)` term integrates to zero. Therefore the density is normalized without a",
        "B-dependent normalizing constant.",
        "",
        "## Objective",
        "",
        "`J(B) = -mean[log(cos(phi_i)/pi) + log(1 + B A_i)]`",
        "",
        "`J'(B) = -mean[A_i/(1 + B A_i)]`",
        "",
        "`J''(B) = mean[A_i^2/(1 + B A_i)^2] >= 0`",
        "",
        "The nonnegative second derivative makes the feasible one-parameter objective convex.",
        "An interior maximum-likelihood estimate solves `J'(B)=0`.",
        "This likelihood is conditional on an electron being backscattered. It fits exit direction only,",
        "not backscatter yield, transmitted counts, or exit energy. At `alpha=0`, every `A_i=0`, so",
        "zero-tilt exits add no information about B.",
        "",
        "## Why the estimate is negative",
        "",
        f"Across `{data['sample_count']}` exits, `mean(A) = {data['mean_asymmetry_factor']:.12f}`.",
        f"Consequently `J'(0) = -mean(A) = {fit['nll_derivative_at_zero']:.12f}`.",
        "Because this derivative is positive, moving left from zero lowers the NLL. The convex",
        f"objective reaches its interior minimum at `B = {fit['fitted_b']:.12f}`.",
        "",
        "Negative B is valid because B is a signed azimuthal harmonic amplitude, not a probability.",
        "Under the audited coordinate convention, it increases density around `cos(varphi)<0`,",
        "opposite the projected into-specimen beam direction.",
        "Reversing `t1` would shift every azimuth by pi and reverse the fitted B sign. Therefore the",
        "sign is meaningful only together with the declared azimuth convention.",
        "",
        "## Numerical result",
        "",
        f"- Mean NLL at `B=0`: `{fit['mean_nll_at_zero']:.12f}`",
        f"- Mean NLL at fitted B: `{fit['mean_nll_at_fit']:.12f}`",
        f"- Mean NLL improvement: `{fit['mean_nll_improvement_vs_zero']:.12f}`",
        f"- Score at fitted B: `{fit['nll_derivative_at_fit']:.3e}`",
        f"- Curvature at fitted B: `{fit['mean_nll_second_derivative_at_fit']:.12f}`",
        f"- Curvature SE, assuming independent trajectories: `{fit['trajectory_independence_approx_standard_error']:.12f}`",
        f"- Approximate 95% interval under that assumption: `{fit['trajectory_independence_approx_95pct_ci']}`",
        f"- Equal-source-run-weight estimate: `{fit['equal_source_run_weight_fitted_b']:.12f}`",
        "",
        "## Feasibility",
        "",
        f"- All-angle feasible interval: `{feasibility['domain_interval']}`",
        f"- Solver interval with safety margin: `{feasibility['solver_interval']}`",
        f"- Smallest observed bracket at the fit: `{feasibility['observed_minimum_bracket_at_fit']:.12f}`",
        f"- Worst-case all-angle bracket margin at the fit: `{feasibility['all_angle_minimum_bracket_at_fit']:.12f}`",
        "",
        "## Objective checkpoints",
        "",
        "| B | Mean NLL | Derivative | Curvature | Minimum observed bracket |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        lines.append(
            f"| {point['b']:.12f} | {point['mean_negative_log_likelihood']:.12f} | "
            f"{point['derivative']:.12f} | {point['second_derivative']:.12f} | "
            f"{point['minimum_bracket']:.12f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical limitation",
            "",
            str(audit["uncertainty_warning"]),
            "",
            "The difference between sample-weighted and equal-source-run-weight estimates shows",
            "that the final B recommendation must be based on per-run and per-tilt reliability, not",
            "only the narrow trajectory-level curvature interval.",
        ]
    )
    (output_dir / "likelihood_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the fitted constant-B likelihood mathematically.")
    parser.add_argument("--exit-samples", required=True, type=Path)
    parser.add_argument("--fit-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    audit = build_audit(args.exit_samples, args.fit_report)
    write_audit(audit, args.output_dir)
    fit = audit["fit"]
    print(f"B_hat       = {fit['fitted_b']:.12f}")
    print(f"J_prime(0)  = {fit['nll_derivative_at_zero']:.12f}")
    print(f"J_prime(B)  = {fit['nll_derivative_at_fit']:.3e}")
    print(f"wrote       = {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
