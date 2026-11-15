from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


Row = dict[str, object]


@dataclass(frozen=True)
class FitPaths:
    report_root: Path

    @property
    def b_fit_root(self) -> Path:
        return self.report_root / "b_fit"

    @property
    def report_json(self) -> Path:
        return self.b_fit_root / "b_fit_report.json"

    @property
    def gui_summary_json(self) -> Path:
        return self.b_fit_root / "b_fit_gui_summary.json"

    @property
    def per_run_csv(self) -> Path:
        return self.b_fit_root / "per_run_fit.csv"

    @property
    def per_tilt_csv(self) -> Path:
        return self.b_fit_root / "per_tilt_fit.csv"

    @property
    def sin_per_tilt_csv(self) -> Path:
        return self.b_fit_root / "b_alpha_sin_per_tilt.csv"

    @property
    def quadratic_per_tilt_csv(self) -> Path:
        return self.b_fit_root / "b_alpha_quadratic_per_tilt.csv"

    @property
    def constant_vs_quadratic_csv(self) -> Path:
        return self.b_fit_root / "constant_vs_quadratic_per_tilt.csv"

    @property
    def constant_loro_csv(self) -> Path:
        return self.b_fit_root / "leave_one_repeat_out.csv"

    @property
    def sin_loro_csv(self) -> Path:
        return self.b_fit_root / "b_alpha_sin_leave_one_repeat_out.csv"

    @property
    def quadratic_loro_csv(self) -> Path:
        return self.b_fit_root / "b_alpha_quadratic_leave_one_repeat_out.csv"

    @property
    def objective_curve_csv(self) -> Path:
        return self.b_fit_root / "objective_curve.csv"

    @property
    def exit_samples_csv(self) -> Path:
        return self.b_fit_root / "exit_samples.csv"


@dataclass(frozen=True)
class ConstantModeSpec:
    key: str
    label: str
    target_b: float
    report_constant_b: float | None
    gui_constant_b: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create six report-ready graphs for the CASINO emission-B calibration story."
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        required=True,
        help="Root directory containing reports/b_fit and related artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write graphs into. Defaults to <report-root>/b_calibration_graphs_6_stage.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path, numeric_columns: Iterable[str]) -> list[Row]:
    numeric = set(numeric_columns)
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: Row = {}
            for key, value in raw.items():
                if key in numeric:
                    row[key] = float(value) if value not in (None, "") else float("nan")
                else:
                    row[key] = value
            rows.append(row)
    return rows


def sort_rows(rows: list[Row], keys: list[str]) -> list[Row]:
    return sorted(rows, key=lambda row: tuple(float(row[key]) for key in keys))


def values(rows: list[Row], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def mean(items: list[float]) -> float:
    return float(sum(items) / len(items)) if items else float("nan")


def std(items: list[float]) -> float:
    if not items:
        return float("nan")
    arr = np.array(items, dtype=float)
    return float(arr.std(ddof=0))


def mean_abs(rows: list[Row], key: str) -> float:
    return mean([abs(float(row[key])) for row in rows])


def read_exit_sample_arrays(path: Path) -> dict[str, np.ndarray]:
    run_id: list[str] = []
    nominal_tilt_deg: list[float] = []
    tilt_deg: list[float] = []
    phi_rad: list[float] = []
    varphi_rad: list[float] = []
    cos_phi: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_id.append(str(row["run_id"]))
            nominal_tilt_deg.append(float(row["nominal_tilt_deg"]))
            tilt_deg.append(float(row["tilt_deg"]))
            phi_rad.append(float(row["phi_rad"]))
            varphi_rad.append(float(row["varphi_rad"]))
            cos_phi.append(float(row["cos_phi"]))

    tilt = np.asarray(tilt_deg, dtype=np.float64)
    phi = np.asarray(phi_rad, dtype=np.float64)
    varphi = np.asarray(varphi_rad, dtype=np.float64)
    asym_factor = np.sin(np.radians(tilt)) * np.sin(phi) * np.cos(varphi)
    return {
        "run_id": np.asarray(run_id, dtype=object),
        "nominal_tilt_deg": np.asarray(nominal_tilt_deg, dtype=np.float64),
        "tilt_deg": tilt,
        "phi_rad": phi,
        "varphi_rad": varphi,
        "cos_phi": np.clip(np.asarray(cos_phi, dtype=np.float64), 1e-12, None),
        "asym_factor": asym_factor,
    }


def _build_b_grid(tilt_deg_values: np.ndarray, grid_points: int = 2001) -> np.ndarray:
    max_sin_alpha = float(np.max(np.abs(np.sin(np.radians(tilt_deg_values)))))
    if max_sin_alpha <= 1e-12:
        limit = 1.0
    else:
        limit = min(2.0, 0.999 / max_sin_alpha)
    return np.linspace(-limit, limit, max(101, int(grid_points)), dtype=np.float64)


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


def _sample_weighted_nll(cos_phi: np.ndarray, asym_factor: np.ndarray, b_values: np.ndarray | float) -> np.ndarray | float:
    b = np.atleast_1d(np.asarray(b_values, dtype=np.float64))
    prob = cos_phi[:, np.newaxis] * (1.0 + asym_factor[:, np.newaxis] * b[np.newaxis, :]) / math.pi
    prob = np.clip(prob, 1e-12, None)
    nll = -np.mean(np.log(prob), axis=0)
    if np.isscalar(b_values):
        return float(nll[0])
    return nll


def _sample_weighted_derivative(asym_factor: np.ndarray, b_values: np.ndarray | float) -> np.ndarray | float:
    b = np.atleast_1d(np.asarray(b_values, dtype=np.float64))
    denom = np.clip(1.0 + asym_factor[:, np.newaxis] * b[np.newaxis, :], 1e-12, None)
    deriv = -np.mean(asym_factor[:, np.newaxis] / denom, axis=0)
    if np.isscalar(b_values):
        return float(deriv[0])
    return deriv


def _sample_weighted_curvature(asym_factor: np.ndarray, b_values: np.ndarray | float) -> np.ndarray | float:
    b = np.atleast_1d(np.asarray(b_values, dtype=np.float64))
    denom = np.clip(1.0 + asym_factor[:, np.newaxis] * b[np.newaxis, :], 1e-12, None)
    curvature = np.mean((asym_factor[:, np.newaxis] ** 2) / (denom**2), axis=0)
    if np.isscalar(b_values):
        return float(curvature[0])
    return curvature


def _build_run_groups(run_ids: np.ndarray) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for idx, run_id in enumerate(run_ids.tolist()):
        grouped.setdefault(str(run_id), []).append(idx)
    return [np.asarray(indices, dtype=np.int32) for _, indices in sorted(grouped.items())]


def _equal_run_curves(
    cos_phi: np.ndarray,
    asym_factor: np.ndarray,
    run_groups: list[np.ndarray],
    b_values: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    b = np.atleast_1d(np.asarray(b_values, dtype=np.float64))
    run_nlls: list[np.ndarray] = []
    run_derivs: list[np.ndarray] = []
    run_curvatures: list[np.ndarray] = []
    for indices in run_groups:
        cos_run = cos_phi[indices][:, np.newaxis]
        asym_run = asym_factor[indices][:, np.newaxis]
        prob = cos_run * (1.0 + asym_run * b[np.newaxis, :]) / math.pi
        prob = np.clip(prob, 1e-12, None)
        denom = np.clip(1.0 + asym_run * b[np.newaxis, :], 1e-12, None)
        run_nlls.append(-np.mean(np.log(prob), axis=0))
        run_derivs.append(-np.mean(asym_run / denom, axis=0))
        run_curvatures.append(np.mean((asym_run**2) / (denom**2), axis=0))

    objective = np.vstack(run_nlls).mean(axis=0)
    derivative = np.vstack(run_derivs).mean(axis=0)
    curvature = np.vstack(run_curvatures).mean(axis=0)
    if np.isscalar(b_values):
        return float(objective[0]), float(derivative[0]), float(curvature[0])
    return objective, derivative, curvature


def fit_constant_from_arrays(
    exit_samples: dict[str, np.ndarray],
    *,
    mode_key: str,
    mask: np.ndarray | None = None,
    grid_points: int = 2001,
) -> dict[str, object]:
    if mask is None:
        mask = np.ones_like(exit_samples["tilt_deg"], dtype=bool)
    cos_phi = exit_samples["cos_phi"][mask]
    asym_factor = exit_samples["asym_factor"][mask]
    tilt_deg = exit_samples["tilt_deg"][mask]
    run_ids = exit_samples["run_id"][mask]

    b_grid = _build_b_grid(tilt_deg, grid_points=grid_points)
    if mode_key == "equal_run":
        run_groups = _build_run_groups(run_ids)
        objective_grid, derivative_grid, curvature_grid = _equal_run_curves(
            cos_phi,
            asym_factor,
            run_groups,
            b_grid,
        )
        objective_fn = lambda b: _equal_run_curves(cos_phi, asym_factor, run_groups, b)[0]
        derivative_fn = lambda b: _equal_run_curves(cos_phi, asym_factor, run_groups, b)[1]
        curvature_fn = lambda b: _equal_run_curves(cos_phi, asym_factor, run_groups, b)[2]
    elif mode_key == "sample_weighted":
        objective_grid = _sample_weighted_nll(cos_phi, asym_factor, b_grid)
        derivative_grid = _sample_weighted_derivative(asym_factor, b_grid)
        curvature_grid = _sample_weighted_curvature(asym_factor, b_grid)
        objective_fn = lambda b: _sample_weighted_nll(cos_phi, asym_factor, b)
        derivative_fn = lambda b: _sample_weighted_derivative(asym_factor, b)
        curvature_fn = lambda b: _sample_weighted_curvature(asym_factor, b)
    else:
        raise ValueError(f"Unknown constant fit mode '{mode_key}'.")

    objective_grid = np.asarray(objective_grid, dtype=np.float64)
    derivative_grid = np.asarray(derivative_grid, dtype=np.float64)
    curvature_grid = np.asarray(curvature_grid, dtype=np.float64)
    observed_interval = _observed_feasible_interval(asym_factor)

    grid_idx = int(np.argmin(objective_grid))
    grid_fitted_b = float(b_grid[grid_idx])
    grid_objective = float(objective_grid[grid_idx])
    solver_lower = float(b_grid[0])
    solver_upper = float(b_grid[-1])

    lower_derivative = float(derivative_fn(solver_lower))
    upper_derivative = float(derivative_fn(solver_upper))
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
            derivative_mid = float(derivative_fn(mid))
            fitted_b = mid
            if abs(derivative_mid) <= 1e-12 or abs(hi - lo) <= 1e-10:
                break
            if derivative_mid > 0.0:
                hi = mid
            else:
                lo = mid

    return {
        "fitted_b": float(fitted_b),
        "negative_log_likelihood": float(objective_fn(fitted_b)),
        "derivative_at_fit": float(derivative_fn(fitted_b)),
        "curvature_at_fit": float(curvature_fn(fitted_b)),
        "grid_fitted_b": grid_fitted_b,
        "grid_negative_log_likelihood": grid_objective,
        "solution_method": solution_method,
        "observed_feasible_interval": observed_interval,
        "b_grid": b_grid,
        "objective_grid": objective_grid,
        "derivative_grid": derivative_grid,
        "curvature_grid": curvature_grid,
        "sample_count": int(cos_phi.size),
        "run_count": int(len(np.unique(run_ids))),
    }


def determine_constant_mode(report: dict, gui_summary: dict | None) -> ConstantModeSpec:
    fit = report["fit"]
    gui_constant_b: float | None = None
    if gui_summary is not None:
        constant_model = (gui_summary.get("model_comparison") or {}).get("constant_b")
        if isinstance(constant_model, dict) and "fitted_b" in constant_model:
            gui_constant_b = float(constant_model["fitted_b"])

    report_constant = fit.get("global_pooled") or fit.get("global_sample_weighted")
    report_constant_b = None if report_constant is None else float(report_constant["fitted_b"])
    equal_run = fit.get("global_equal_run")
    equal_run_b = None if equal_run is None else float(equal_run["fitted_b"])

    if gui_constant_b is not None and equal_run_b is not None and abs(gui_constant_b - equal_run_b) <= 1e-9:
        return ConstantModeSpec(
            key="equal_run",
            label="Recomputed equal-run solve (same family as the GUI summary constant)",
            target_b=gui_constant_b,
            report_constant_b=report_constant_b,
            gui_constant_b=gui_constant_b,
        )
    if report_constant_b is not None:
        return ConstantModeSpec(
            key="sample_weighted",
            label="Primary pooled/sample-weighted solve",
            target_b=report_constant_b,
            report_constant_b=report_constant_b,
            gui_constant_b=gui_constant_b,
        )
    if equal_run_b is not None:
        return ConstantModeSpec(
            key="equal_run",
            label="Equal-run solve",
            target_b=equal_run_b,
            report_constant_b=report_constant_b,
            gui_constant_b=gui_constant_b,
        )
    raise ValueError("Could not determine a constant-B model from the fit report.")


def prepare_data(paths: FitPaths) -> dict[str, object]:
    report = read_json(paths.report_json)
    gui_summary = read_json(paths.gui_summary_json) if paths.gui_summary_json.exists() else None
    per_run = sort_rows(
        read_csv_rows(
            paths.per_run_csv,
            [
                "nominal_tilt_deg",
                "tilt_deg",
                "repeat_id",
                "sample_count",
                "fitted_b",
                "negative_log_likelihood",
            ],
        ),
        ["nominal_tilt_deg", "repeat_id"],
    )
    per_tilt = sort_rows(
        read_csv_rows(
            paths.per_tilt_csv,
            [
                "nominal_tilt_deg",
                "measured_tilt_deg_mean",
                "run_count",
                "sample_count",
                "fitted_b",
                "negative_log_likelihood",
            ],
        ),
        ["nominal_tilt_deg"],
    )
    sin_per_tilt = sort_rows(
        read_csv_rows(
            paths.sin_per_tilt_csv,
            [
                "nominal_tilt_deg",
                "measured_tilt_deg_mean",
                "observed_fitted_b",
                "predicted_b",
                "residual_b",
                "sample_count",
                "run_count",
            ],
        ),
        ["nominal_tilt_deg"],
    )
    quadratic_per_tilt = sort_rows(
        read_csv_rows(
            paths.quadratic_per_tilt_csv,
            [
                "nominal_tilt_deg",
                "measured_tilt_deg_mean",
                "observed_fitted_b",
                "predicted_b",
                "residual_b",
                "sample_count",
                "run_count",
            ],
        ),
        ["nominal_tilt_deg"],
    )
    constant_vs_quadratic = sort_rows(
        read_csv_rows(
            paths.constant_vs_quadratic_csv,
            [
                "nominal_tilt_deg",
                "measured_tilt_deg_mean",
                "observed_fitted_b",
                "constant_b",
                "constant_abs_residual_b",
                "quadratic_b",
                "quadratic_abs_residual_b",
                "quadratic_minus_constant_abs_residual_b",
                "sample_count",
                "run_count",
            ],
        ),
        ["nominal_tilt_deg"],
    )
    constant_loro = sort_rows(
        read_csv_rows(
            paths.constant_loro_csv,
            [
                "holdout_repeat_id",
                "fitted_b",
                "train_objective",
                "test_negative_log_likelihood",
                "train_run_count",
                "test_run_count",
                "train_sample_count",
                "test_sample_count",
            ],
        ),
        ["holdout_repeat_id"],
    )
    sin_loro = sort_rows(
        read_csv_rows(
            paths.sin_loro_csv,
            [
                "holdout_repeat_id",
                "fitted_b0",
                "fitted_b1",
                "train_objective",
                "test_negative_log_likelihood",
            ],
        ),
        ["holdout_repeat_id"],
    )
    quadratic_loro = sort_rows(
        read_csv_rows(
            paths.quadratic_loro_csv,
            [
                "holdout_repeat_id",
                "fitted_b0",
                "fitted_b1",
                "fitted_b2",
                "train_objective",
                "test_negative_log_likelihood",
            ],
        ),
        ["holdout_repeat_id"],
    )

    constant_mode = determine_constant_mode(report, gui_summary)
    exit_samples = read_exit_sample_arrays(paths.exit_samples_csv)
    selected_constant_fit = fit_constant_from_arrays(
        exit_samples,
        mode_key=constant_mode.key,
        grid_points=2001,
    )

    cutoff_rows: list[dict[str, float]] = []
    for cutoff in (30.0, 40.0, 45.0, 50.0, 60.0, 70.0):
        mask = exit_samples["nominal_tilt_deg"] <= cutoff
        cutoff_fit = fit_constant_from_arrays(
            exit_samples,
            mode_key=constant_mode.key,
            mask=mask,
            grid_points=2001,
        )
        cutoff_rows.append(
            {
                "max_nominal_tilt_deg": cutoff,
                "fitted_b": float(cutoff_fit["fitted_b"]),
                "negative_log_likelihood": float(cutoff_fit["negative_log_likelihood"]),
                "sample_count": float(cutoff_fit["sample_count"]),
                "run_count": float(cutoff_fit["run_count"]),
            }
        )

    asymmetry_rows: list[dict[str, float]] = []
    nominal_tilts = np.unique(exit_samples["nominal_tilt_deg"])
    for tilt in nominal_tilts:
        mask = exit_samples["nominal_tilt_deg"] == tilt
        asym = exit_samples["asym_factor"][mask]
        asymmetry_rows.append(
            {
                "nominal_tilt_deg": float(tilt),
                "mean_a": float(np.mean(asym)),
                "mean_abs_a": float(np.mean(np.abs(asym))),
                "positive_fraction": float(np.mean(asym > 0.0)),
                "sample_count": float(mask.sum()),
            }
        )

    fit = report["fit"]
    quadratic_params = fit.get("b_alpha_quadratic_sample_weighted") or fit["b_alpha_quadratic_equal_run"]
    sin_params = fit.get("b_alpha_sin_sample_weighted") or fit["b_alpha_sin_equal_run"]

    mean_abs_residuals = {
        "constant": mean_abs(constant_vs_quadratic, "constant_abs_residual_b"),
        "sin": mean_abs(sin_per_tilt, "residual_b"),
        "quadratic": mean_abs(quadratic_per_tilt, "residual_b"),
    }
    mean_holdout_nll = {
        "constant": mean([float(row["test_negative_log_likelihood"]) for row in constant_loro]),
        "sin": mean([float(row["test_negative_log_likelihood"]) for row in sin_loro]),
        "quadratic": mean([float(row["test_negative_log_likelihood"]) for row in quadratic_loro]),
    }

    return {
        "report": report,
        "gui_summary": gui_summary,
        "constant_mode": constant_mode,
        "selected_constant_fit": selected_constant_fit,
        "exit_samples": exit_samples,
        "per_run": per_run,
        "per_tilt": per_tilt,
        "sin_per_tilt": sin_per_tilt,
        "quadratic_per_tilt": quadratic_per_tilt,
        "constant_vs_quadratic": constant_vs_quadratic,
        "constant_loro": constant_loro,
        "sin_loro": sin_loro,
        "quadratic_loro": quadratic_loro,
        "cutoff_rows": cutoff_rows,
        "asymmetry_rows": asymmetry_rows,
        "mean_abs_residuals": mean_abs_residuals,
        "mean_holdout_nll": mean_holdout_nll,
        "sin_params": sin_params,
        "quadratic_params": quadratic_params,
    }


def save_constant_solve(data: dict[str, object], output_dir: Path) -> Path:
    constant_mode: ConstantModeSpec = data["constant_mode"]
    fit = data["selected_constant_fit"]
    b_grid = np.asarray(fit["b_grid"], dtype=np.float64)
    objective = np.asarray(fit["objective_grid"], dtype=np.float64)
    derivative = np.asarray(fit["derivative_grid"], dtype=np.float64)
    curvature = np.asarray(fit["curvature_grid"], dtype=np.float64)
    fitted_b = float(fit["fitted_b"])
    lower, upper = fit["observed_feasible_interval"]

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.4), sharex=True)
    for ax in axes:
        if np.isfinite(lower):
            ax.axvline(lower, color="#999999", linestyle="--", linewidth=1.0)
        if np.isfinite(upper):
            ax.axvline(upper, color="#999999", linestyle="--", linewidth=1.0)
        ax.axvline(fitted_b, color="#E15759", linewidth=1.4, linestyle="-")

    axes[0].plot(b_grid, objective, color="#4E79A7", linewidth=2.0)
    axes[0].scatter([fitted_b], [float(fit["negative_log_likelihood"])], color="#E15759", zorder=3)
    axes[0].set_title(f"Constant-B Solve Used by {constant_mode.label}")
    axes[0].set_ylabel("NLL(B)")
    axes[0].text(
        0.99,
        0.98,
        (
            f"fitted B = {fitted_b:.6f}\n"
            f"method = {fit['solution_method']}\n"
            f"sample count = {fit['sample_count']:,}\n"
            f"run count = {fit['run_count']}"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#333333",
    )

    axes[1].plot(b_grid, derivative, color="#F28E2B", linewidth=2.0)
    axes[1].axhline(0.0, color="#444444", linewidth=1.0)
    axes[1].set_ylabel("dNLL/dB")

    axes[2].plot(b_grid, curvature, color="#59A14F", linewidth=2.0)
    axes[2].set_ylabel("d²NLL/dB²")
    axes[2].set_xlabel("B")
    axes[2].text(
        0.99,
        0.02,
        "Dashed lines mark the observed feasible interval.",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#444444",
    )

    fig.tight_layout()
    output_path = output_dir / "01_constant_b_solve.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_b_vs_tilt(data: dict[str, object], output_dir: Path) -> Path:
    per_run = data["per_run"]
    per_tilt = data["per_tilt"]
    sin_per_tilt = data["sin_per_tilt"]
    quadratic_per_tilt = data["quadratic_per_tilt"]
    fit = data["selected_constant_fit"]
    constant_mode: ConstantModeSpec = data["constant_mode"]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    jitter_offsets = {1.0: -0.45, 2.0: -0.15, 3.0: 0.15, 4.0: 0.35, 5.0: 0.55}
    x = np.array(
        [
            float(row["nominal_tilt_deg"]) + jitter_offsets.get(float(row["repeat_id"]), 0.0)
            for row in per_run
        ],
        dtype=float,
    )
    sample_counts = values(per_run, "sample_count")
    sizes = 28 + 18 * np.sqrt(sample_counts / sample_counts.max())
    ax.scatter(
        x,
        values(per_run, "fitted_b"),
        s=sizes,
        alpha=0.55,
        color="#4E79A7",
        edgecolors="white",
        linewidths=0.5,
        label="Per-run fitted B",
    )
    ax.plot(
        values(per_tilt, "nominal_tilt_deg"),
        values(per_tilt, "fitted_b"),
        color="#1F1F1F",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label="Observed per-tilt fitted B",
    )
    ax.plot(
        values(sin_per_tilt, "nominal_tilt_deg"),
        values(sin_per_tilt, "predicted_b"),
        color="#F28E2B",
        linewidth=2.0,
        linestyle="--",
        label="Sin fit",
    )
    ax.plot(
        values(quadratic_per_tilt, "nominal_tilt_deg"),
        values(quadratic_per_tilt, "predicted_b"),
        color="#59A14F",
        linewidth=2.4,
        label="Quadratic fit",
    )
    ax.axhline(
        float(fit["fitted_b"]),
        color="#B07AA1",
        linewidth=1.8,
        linestyle=":",
        label=f"{constant_mode.label}: B = {float(fit['fitted_b']):.3f}",
    )
    ax.axvspan(-1.5, 2.5, color="#E15759", alpha=0.08)
    zero_row = next(row for row in per_tilt if float(row["nominal_tilt_deg"]) == 0.0)
    ax.annotate(
        "0 degree is weakly constrained\n(B becomes effectively unidentifiable)",
        xy=(0, float(zero_row["fitted_b"])),
        xytext=(7.5, -1.45),
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.0},
        fontsize=8.5,
        color="#444444",
    )
    ax.set_title("Observed Per-Tilt B Shows Why a Single Constant Struggles Across 0-70 Degrees")
    ax.set_xlabel("Nominal tilt (deg)")
    ax.set_ylabel("Fitted B")
    ax.set_xlim(-1, 72)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    output_path = output_dir / "02_b_vs_tilt.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_residuals_by_tilt(data: dict[str, object], output_dir: Path) -> Path:
    per_tilt = data["per_tilt"]
    sin_per_tilt = data["sin_per_tilt"]
    quadratic_per_tilt = data["quadratic_per_tilt"]
    constant_vs_quadratic = data["constant_vs_quadratic"]

    tilts = values(per_tilt, "nominal_tilt_deg")
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(
        tilts,
        values(constant_vs_quadratic, "constant_abs_residual_b"),
        color="#B07AA1",
        linewidth=2.0,
        marker="o",
        label="Constant residual",
    )
    ax.plot(
        tilts,
        np.abs(values(sin_per_tilt, "residual_b")),
        color="#F28E2B",
        linewidth=2.0,
        marker="o",
        linestyle="--",
        label="Sin residual",
    )
    ax.plot(
        tilts,
        np.abs(values(quadratic_per_tilt, "residual_b")),
        color="#59A14F",
        linewidth=2.4,
        marker="o",
        label="Quadratic residual",
    )
    ax.set_title("Residuals by Tilt Show Where the Constant Model Breaks Down")
    ax.set_xlabel("Nominal tilt (deg)")
    ax.set_ylabel("Absolute residual in fitted B")
    ax.set_xlim(-1, 72)

    ax2 = ax.twinx()
    ax2.bar(
        tilts,
        values(per_tilt, "sample_count"),
        width=3.2,
        color="#4E79A7",
        alpha=0.18,
        label="Exit samples",
    )
    ax2.set_ylabel("Exit sample count")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True)
    fig.tight_layout()
    output_path = output_dir / "03_residuals_by_tilt.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_model_selection(data: dict[str, object], output_dir: Path) -> Path:
    mean_holdout_nll = data["mean_holdout_nll"]
    mean_abs_residuals = data["mean_abs_residuals"]
    constant_loro = data["constant_loro"]
    sin_loro = data["sin_loro"]
    quadratic_loro = data["quadratic_loro"]

    models = ["Constant", "Sin", "Quadratic"]
    holdout_means = [
        float(mean_holdout_nll["constant"]),
        float(mean_holdout_nll["sin"]),
        float(mean_holdout_nll["quadratic"]),
    ]
    holdout_stds = [
        std([float(row["test_negative_log_likelihood"]) for row in constant_loro]),
        std([float(row["test_negative_log_likelihood"]) for row in sin_loro]),
        std([float(row["test_negative_log_likelihood"]) for row in quadratic_loro]),
    ]
    residual_means = [
        float(mean_abs_residuals["constant"]),
        float(mean_abs_residuals["sin"]),
        float(mean_abs_residuals["quadratic"]),
    ]
    colors = ["#B07AA1", "#F28E2B", "#59A14F"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].bar(models, holdout_means, yerr=holdout_stds, color=colors, capsize=4)
    axes[0].set_title("Leave-One-Repeat-Out NLL")
    axes[0].set_ylabel("Mean holdout negative log-likelihood")
    nll_floor = min(holdout_means) - 0.01
    nll_ceiling = max(holdout_means) + max(holdout_stds) + 0.01
    axes[0].set_ylim(nll_floor, nll_ceiling)

    axes[1].bar(models, residual_means, color=colors)
    axes[1].set_title("Mean Absolute Per-Tilt Residual")
    axes[1].set_ylabel("|Observed B - model B|")

    fig.suptitle("Holdout Error and Per-Tilt Residuals Compare the Constant and Tilt-Dependent Laws", y=1.02, fontsize=12)
    fig.tight_layout()
    output_path = output_dir / "04_model_selection.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_tilt_cutoff_stability(data: dict[str, object], output_dir: Path) -> Path:
    cutoff_rows = data["cutoff_rows"]
    x = np.array([float(row["max_nominal_tilt_deg"]) for row in cutoff_rows], dtype=float)
    y = np.array([float(row["fitted_b"]) for row in cutoff_rows], dtype=float)
    samples = np.array([float(row["sample_count"]) for row in cutoff_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    sizes = 60 + 120 * np.sqrt(samples / samples.max())
    ax.plot(x, y, color="#4E79A7", linewidth=2.2, marker="o", markersize=5.0)
    ax.scatter(x, y, s=sizes, color="#4E79A7", alpha=0.35)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title("The Constant-B Sign Flips as Higher Tilts Are Included")
    ax.set_xlabel("Maximum nominal tilt included in the fit (deg)")
    ax.set_ylabel("Constant fitted B")
    ax.set_xlim(28, 72)
    ax.annotate(
        "Low and mid tilts alone favor a positive B.\nThe pooled sign turns negative only once high-tilt runs enter.",
        xy=(60, y[list(x).index(60.0)]),
        xytext=(41, 0.55),
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.0},
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout()
    output_path = output_dir / "05_tilt_cutoff_stability.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_asymmetry_by_tilt(data: dict[str, object], output_dir: Path) -> Path:
    rows = data["asymmetry_rows"]
    tilts = np.array([float(row["nominal_tilt_deg"]) for row in rows], dtype=float)
    mean_a = np.array([float(row["mean_a"]) for row in rows], dtype=float)
    mean_abs_a = np.array([float(row["mean_abs_a"]) for row in rows], dtype=float)
    pos_frac = np.array([float(row["positive_fraction"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.0), sharex=True)
    axes[0].plot(tilts, mean_a, color="#E15759", linewidth=2.2, marker="o", label="Mean a_i")
    axes[0].plot(tilts, mean_abs_a, color="#4E79A7", linewidth=2.0, marker="o", linestyle="--", label="Mean |a_i|")
    axes[0].axhline(0.0, color="#444444", linewidth=1.0)
    axes[0].set_ylabel("Asymmetry factor statistics")
    axes[0].set_title("The RR Asymmetry Factor Changes Sign at High Tilt")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(tilts, pos_frac, color="#59A14F", linewidth=2.2, marker="o")
    axes[1].axhline(0.5, color="#444444", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Nominal tilt (deg)")
    axes[1].set_ylabel("Fraction of samples with a_i > 0")
    axes[1].set_ylim(0.2, 0.6)
    axes[1].annotate(
        "Below about 40 deg, positive asymmetry dominates.\nBy 60-70 deg the balance shifts negative.",
        xy=(60, pos_frac[list(tilts).index(60.0)]),
        xytext=(41, 0.31),
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.0},
        fontsize=8.5,
        color="#444444",
    )

    fig.tight_layout()
    output_path = output_dir / "06_asymmetry_by_tilt.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_summary(data: dict[str, object], output_dir: Path, graph_paths: list[Path]) -> tuple[Path, Path]:
    report = data["report"]
    campaign = report["campaign"]
    constant_mode: ConstantModeSpec = data["constant_mode"]
    fit = data["selected_constant_fit"]
    quadratic_params = data["quadratic_params"]

    summary = {
        "campaign": {
            "run_count": campaign["run_count"],
            "exit_sample_count": campaign["exit_sample_count"],
            "nominal_tilt_deg_values": campaign["nominal_tilt_deg_values"],
        },
        "constant_mode": {
            "key": constant_mode.key,
            "label": constant_mode.label,
            "selected_fitted_b": float(fit["fitted_b"]),
            "report_primary_constant_b": constant_mode.report_constant_b,
            "gui_constant_b": constant_mode.gui_constant_b,
        },
        "model_selection": {
            "mean_holdout_nll": data["mean_holdout_nll"],
            "mean_abs_per_tilt_residual": data["mean_abs_residuals"],
        },
        "quadratic": {
            "b0": quadratic_params["b0"],
            "b1": quadratic_params["b1"],
            "b2": quadratic_params["b2"],
            "formula": quadratic_params["formula"],
        },
        "graphs": {path.stem: str(path) for path in graph_paths},
    }
    json_path = output_dir / "b_calibration_graphs_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# B calibration six-stage graph summary",
        "",
        f"- Campaign: {campaign['run_count']} runs, {campaign['exit_sample_count']} exit samples",
        f"- Constant-B mode used for these graphs: {constant_mode.label}",
        f"- Recomputed constant B from exit_samples.csv: {float(fit['fitted_b']):.6f}",
    ]
    if constant_mode.report_constant_b is not None:
        md_lines.append(f"- Report primary constant B: {float(constant_mode.report_constant_b):.6f}")
    if constant_mode.gui_constant_b is not None:
        md_lines.append(f"- GUI summary constant B: {float(constant_mode.gui_constant_b):.6f}")
    md_lines.extend(
        [
            (
                "- Quadratic B(alpha): "
                f"B = {float(quadratic_params['b0']):.6f} + {float(quadratic_params['b1']):.6f} sin(alpha)"
                f" + {float(quadratic_params['b2']):.6f} sin^2(alpha)"
            ),
            (
                "- Mean holdout NLL: "
                f"constant {float(data['mean_holdout_nll']['constant']):.6f}, "
                f"sin {float(data['mean_holdout_nll']['sin']):.6f}, "
                f"quadratic {float(data['mean_holdout_nll']['quadratic']):.6f}"
            ),
            "",
            "## Files",
        ]
    )
    md_lines.extend(f"- `{path.name}`" for path in graph_paths)
    md_path = output_dir / "b_calibration_graphs_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    output_dir = (args.output_dir or (report_root / "b_calibration_graphs_6_stage")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    data = prepare_data(FitPaths(report_root=report_root))

    graph_paths = [
        save_constant_solve(data, output_dir),
        save_b_vs_tilt(data, output_dir),
        save_residuals_by_tilt(data, output_dir),
        save_model_selection(data, output_dir),
        save_tilt_cutoff_stability(data, output_dir),
        save_asymmetry_by_tilt(data, output_dir),
    ]
    summary_json, summary_md = save_summary(data, output_dir, graph_paths)

    print(str(output_dir))
    print(str(summary_json))
    print(str(summary_md))
    for path in graph_paths:
        print(str(path))


if __name__ == "__main__":
    main()
