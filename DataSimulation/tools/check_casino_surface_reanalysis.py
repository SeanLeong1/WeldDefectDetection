"""Check geometric tolerance sensitivity using compact caches, and plot B results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
from bse_renderer.validation.casino_compact import select_surface_exits, fit_scalar_b
from bse_renderer.validation.casino_emission_fit import _parse_run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    report = json.loads((root / "surface_b_report.json").read_text())
    baseline = {r["nominal_tilt_deg"]: r for r in report["per_tilt"]}
    results = []
    for tolerance in (.25, .5, 2.):
        parts = []
        for cache in sorted((root / "compact").glob("*.npz")):
            nominal, repeat = _parse_run_id(cache.stem)
            sample, _ = select_surface_exits(cache, nominal, repeat, tolerance)
            parts.append(sample)
        samples = np.concatenate(parts)
        rows = []
        for tilt in sorted(np.unique(samples[:, 4])):
            fitted = fit_scalar_b(samples[samples[:, 4] == tilt])
            b_delta = None if fitted["b"] is None else fitted["b"] - baseline[tilt]["b"]
            rows.append({"tilt": float(tilt), **fitted,
                         "count_change_from_1nm": fitted["n"] - baseline[tilt]["n"],
                         "b_change_from_1nm": b_delta})
        results.append({"plane_tolerance_nm": tolerance, "selected_exits": len(samples), "per_tilt": rows})
        print(f"Tolerance {tolerance:g} nm: {len(samples)} exits; largest |delta B| = "
              f"{max(abs(r['b_change_from_1nm'] or 0) for r in rows):.8g}", flush=True)
    (root / "tolerance_sensitivity.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    historical = json.loads((PROJECT / "data/output/casino_reruns/parametric_model_tilt_304ss_60kev_combined_0to70_fit/reports/b_fit_schema5_stage4/per_tilt_b_reliability.json").read_text())
    old_rows = historical["results"]
    fig, (ax, lower) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    old = [r for r in old_rows if r["nominal_tilt_deg"] > 0]
    old_b_key = "fitted_b" if "fitted_b" in old[0] else "b"
    ax.plot([r["nominal_tilt_deg"] for r in old], [r[old_b_key] for r in old], "o--", color="#b54740", label="Historical global-class selection")
    new = [r for r in report["per_tilt"] if r["b"] is not None]
    x = np.array([r["nominal_tilt_deg"] for r in new])
    y = np.array([r["b"] for r in new])
    errors = np.array([[r["b"] - r["run_b_range"][0] for r in new],
                       [r["run_b_range"][1] - r["b"] for r in new]])
    ax.errorbar(x, y, yerr=np.maximum(errors, 0), fmt="o-", capsize=4, color="#126c69",
                label="Corrected surface selection (bars: repeat range)")
    ax.axhline(0, color="#999999", lw=.8)
    ax.set_ylabel("Signed B coefficient")
    ax.set_title("Correcting which surface the electrons leave changes the fitted B")
    ax.legend(loc="lower left")
    ax.grid(alpha=.2)
    blocks = json.loads((root / "surface_selection.json").read_text())
    percentages = []
    for tilt in x:
        rows = [r for r in blocks if r["nominal_tilt_deg"] == tilt and r["status"] == "eligible"]
        percentages.append(100 * sum(r["recovered_global_transmitted"] for r in rows) / sum(r["same_surface_outward"] for r in rows))
    lower.bar(x, percentages, width=3, color="#537895")
    lower.set_xlabel("Local incidence angle (degrees)")
    lower.set_ylabel("Selected exits labelled\nTransmitted by CASINO (%)")
    lower.set_xticks([0, 5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70])
    lower.grid(axis="y", alpha=.2)
    fig.savefig(root / "b_before_after.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
