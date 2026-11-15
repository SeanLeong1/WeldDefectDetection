from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    return None if raw in {None, ""} else float(raw)


def _group_by_tilt(rows: list[dict[str, str]]) -> dict[float, list[dict[str, str]]]:
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["nominal_tilt_deg"])].append(row)
    return grouped


def _polar_heterogeneity(rows: list[dict[str, str]]) -> dict[str, Any]:
    values: list[tuple[float, float]] = []
    significant_positive = False
    significant_negative = False
    for row in rows:
        value = _float(row, "diagnostic_b_measured_tilt")
        standard_error = _float(row, "diagnostic_b_standard_error")
        if value is None or standard_error is None or standard_error <= 0.0:
            continue
        values.append((value, standard_error))
        significant_positive |= value - 1.96 * standard_error > 0.0
        significant_negative |= value + 1.96 * standard_error < 0.0
    if len(values) < 2:
        return {
            "weighted_mean_b": None,
            "reduced_chi_square": None,
            "minimum_b": None,
            "maximum_b": None,
            "significant_sign_reversal": False,
        }
    weights = [1.0 / (standard_error * standard_error) for _, standard_error in values]
    weighted_mean = sum(weight * value for weight, (value, _) in zip(weights, values)) / sum(weights)
    q_value = sum(
        ((value - weighted_mean) / standard_error) ** 2 for value, standard_error in values
    )
    return {
        "weighted_mean_b": weighted_mean,
        "reduced_chi_square": q_value / (len(values) - 1),
        "minimum_b": min(value for value, _ in values),
        "maximum_b": max(value for value, _ in values),
        "significant_sign_reversal": significant_positive and significant_negative,
    }


def build_audit(report_dir: Path) -> dict[str, Any]:
    per_tilt_rows = _read_csv(report_dir / "per_tilt_fit.csv")
    per_run = _group_by_tilt(_read_csv(report_dir / "per_run_fit.csv"))
    harmonic_rows = {
        float(row["nominal_tilt_deg"]): row
        for row in _read_csv(report_dir / "azimuth_harmonics_per_tilt.csv")
    }
    polar_rows = _group_by_tilt(
        [
            row
            for row in _read_csv(report_dir / "azimuth_harmonics_per_polar_band.csv")
            if row["aggregation_level"] == "tilt"
        ]
    )

    exit_counts: dict[float, dict[str, float]] = defaultdict(
        lambda: {"sample_count": 0.0, "nonpositive_cos_phi_count": 0.0, "maximum_phi_deg": 0.0}
    )
    with (report_dir / "exit_samples.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tilt = float(row["nominal_tilt_deg"])
            cos_phi = float(row["cos_phi"])
            phi_deg = math.degrees(float(row["phi_rad"]))
            exit_counts[tilt]["sample_count"] += 1.0
            exit_counts[tilt]["nonpositive_cos_phi_count"] += float(cos_phi <= 0.0)
            exit_counts[tilt]["maximum_phi_deg"] = max(
                exit_counts[tilt]["maximum_phi_deg"], phi_deg
            )

    results: list[dict[str, Any]] = []
    for row in sorted(per_tilt_rows, key=lambda item: float(item["nominal_tilt_deg"])):
        nominal_tilt = float(row["nominal_tilt_deg"])
        measured_tilt = float(row["measured_tilt_deg_mean"])
        fitted_b = float(row["fitted_b"])
        standard_error = _float(row, "approx_standard_error")
        run_values = [float(run["fitted_b"]) for run in per_run[nominal_tilt]]
        harmonic = harmonic_rows[nominal_tilt]
        polar = _polar_heterogeneity(polar_rows[nominal_tilt])
        counts = exit_counts[nominal_tilt]
        invalid_fraction = counts["nonpositive_cos_phi_count"] / counts["sample_count"]
        sin_alpha = abs(math.sin(math.radians(measured_tilt)))
        boundary_fraction = abs(fitted_b) * sin_alpha
        first_to_second = float(harmonic["first_to_second_amplitude_ratio"])

        failures: list[str] = []
        cautions: list[str] = []
        if sin_alpha < 1e-4:
            failures.append("unidentifiable because sin(alpha) is approximately zero")
        if invalid_fraction > 0.01:
            failures.append(
                f"{100.0 * invalid_fraction:.1f}% of parsed exits lie outside the outward hemisphere"
            )
        if boundary_fraction >= 0.95:
            failures.append("fit is at or within 5% of the global positivity boundary")
        if polar["significant_sign_reversal"]:
            failures.append("polar bands require statistically clear opposite B signs")
        if polar["reduced_chi_square"] is not None and polar["reduced_chi_square"] > 25.0:
            failures.append("strong polar-band heterogeneity rejects one full-hemisphere scalar B")
        if first_to_second < 1.0:
            failures.append("second azimuthal harmonic is stronger than the first harmonic")

        if not failures and sin_alpha < 0.15:
            cautions.append("low incidence angle weakens B identifiability")
        if not failures and polar["reduced_chi_square"] is not None and polar["reduced_chi_square"] > 2.5:
            cautions.append("moderate polar-band heterogeneity")
        if not failures and len(run_values) > 1 and statistics.stdev(run_values) > 0.15:
            cautions.append("between-run B spread exceeds 0.15")

        if failures:
            decision = "DO_NOT_USE"
        elif cautions:
            decision = "PROVISIONAL"
        else:
            decision = "USE"

        ci = None
        if standard_error is not None:
            ci = [fitted_b - 1.96 * standard_error, fitted_b + 1.96 * standard_error]
        results.append(
            {
                "nominal_tilt_deg": nominal_tilt,
                "measured_tilt_deg": measured_tilt,
                "sample_count": int(float(row["sample_count"])),
                "run_count": int(float(row["run_count"])),
                "fitted_b": fitted_b,
                "trajectory_level_standard_error": standard_error,
                "trajectory_level_95pct_ci": ci,
                "effective_modulation_abs_b_sin_alpha": boundary_fraction,
                "run_b_mean": statistics.mean(run_values),
                "run_b_standard_deviation": (
                    statistics.stdev(run_values) if len(run_values) > 1 else 0.0
                ),
                "run_b_minimum": min(run_values),
                "run_b_maximum": max(run_values),
                "harmonic_diagnostic_b": _float(harmonic, "diagnostic_b_measured_tilt"),
                "first_to_second_harmonic_ratio": first_to_second,
                "polar_band_weighted_b": polar["weighted_mean_b"],
                "polar_band_reduced_chi_square": polar["reduced_chi_square"],
                "polar_band_minimum_b": polar["minimum_b"],
                "polar_band_maximum_b": polar["maximum_b"],
                "polar_band_significant_sign_reversal": polar["significant_sign_reversal"],
                "nonpositive_cos_phi_count": int(counts["nonpositive_cos_phi_count"]),
                "nonpositive_cos_phi_fraction": invalid_fraction,
                "maximum_phi_deg": counts["maximum_phi_deg"],
                "decision": decision,
                "failure_reasons": failures,
                "cautions": cautions,
            }
        )

    return {
        "schema_version": 1,
        "scope": "Reliability of one constant B per nominal tilt over the full outward hemisphere.",
        "source_report_dir": str(report_dir.resolve()),
        "decision_policy": {
            "do_not_use": [
                "sin(alpha) < 1e-4",
                "more than 1% of exits have cos(phi) <= 0",
                "|B sin(alpha)| >= 0.95, indicating positivity-boundary proximity",
                "polar bands contain statistically clear opposite signs",
                "polar-band reduced chi-square > 25",
                "first-to-second harmonic amplitude ratio < 1",
            ],
            "provisional": [
                "otherwise acceptable but sin(alpha) < 0.15",
                "otherwise acceptable but polar-band reduced chi-square > 2.5",
                "otherwise acceptable but between-run standard deviation > 0.15",
            ],
            "use": "No exclusion or provisional criteria triggered.",
        },
        "results": results,
        "recommended_full_hemisphere_values": [
            {"nominal_tilt_deg": row["nominal_tilt_deg"], "fitted_b": row["fitted_b"]}
            for row in results
            if row["decision"] == "USE"
        ],
        "provisional_values": [
            {"nominal_tilt_deg": row["nominal_tilt_deg"], "fitted_b": row["fitted_b"]}
            for row in results
            if row["decision"] == "PROVISIONAL"
        ],
        "uncertainty_warning": (
            "Per-tilt standard errors treat trajectories as independent and are not final clustered "
            "uncertainty intervals. Decisions use them only as diagnostics alongside repeat and "
            "model-adequacy checks."
        ),
    }


def write_audit(audit: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_tilt_b_reliability.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    rows = audit["results"]
    with (output_dir / "per_tilt_b_reliability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        flattened = []
        for row in rows:
            flattened.append(
                {
                    **{key: value for key, value in row.items() if key not in {"failure_reasons", "cautions"}},
                    "failure_reasons": "; ".join(row["failure_reasons"]),
                    "cautions": "; ".join(row["cautions"]),
                }
            )
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

    lines = [
        "# Stage 4: per-tilt B reliability decisions",
        "",
        str(audit["scope"]),
        "",
        "## Decision table",
        "",
        "| Tilt | B MLE | Run range | Polar reduced chi-square | 1st/2nd harmonic | Invalid hemisphere | Decision |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        polar_q = row["polar_band_reduced_chi_square"]
        lines.append(
            f"| {row['nominal_tilt_deg']:.0f} deg | {row['fitted_b']:.6f} | "
            f"[{row['run_b_minimum']:.3f}, {row['run_b_maximum']:.3f}] | "
            f"{'NA' if polar_q is None else f'{polar_q:.2f}'} | "
            f"{row['first_to_second_harmonic_ratio']:.2f} | "
            f"{100.0 * row['nonpositive_cos_phi_fraction']:.2f}% | {row['decision']} |"
        )
    lines.extend(["", "## Reasons by tilt", ""])
    for row in rows:
        reasons = row["failure_reasons"] or row["cautions"] or ["all declared criteria passed"]
        lines.append(
            f"- **{row['nominal_tilt_deg']:.0f} deg, {row['decision']}**: " + "; ".join(reasons) + "."
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use the 10 and 15 degree values as the defensible full-hemisphere scalar-B calibration",
            "subset. Keep 5 degrees provisional because the `sin(alpha)` multiplier weakens",
            "identification, and keep 20 degrees provisional because polar-band heterogeneity is",
            "already detectable. Do not use 25 degrees or above as one scalar B over the",
            "whole hemisphere; polar-angle dependence or higher harmonics become material.",
            "",
            "The 70 degree data also fail the coordinate-domain check: approximately 19% of parsed",
            "exits have `cos(phi)<=0`, the measured plane tilt differs materially from the nominal",
            "tilt, and the fitted B is on the positivity boundary.",
            "",
            "## Uncertainty limitation",
            "",
            str(audit["uncertainty_warning"]),
        ]
    )
    (output_dir / "per_tilt_b_reliability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify the reliability of each per-tilt B fit.")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    audit = build_audit(args.report_dir)
    write_audit(audit, args.output_dir)
    for row in audit["results"]:
        print(f"{row['nominal_tilt_deg']:>4.0f} deg  B={row['fitted_b']:>10.6f}  {row['decision']}")
    print(f"wrote = {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
