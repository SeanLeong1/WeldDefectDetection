from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bse_renderer.validation.casino_emission_fit import (
    build_trajectory_angle_audit,
    _build_constant_vs_quadratic_per_tilt_rows,
    _build_coordinate_frame_diagnostics,
    _build_rr_normalization_check,
    _build_azimuth_harmonic_diagnostics,
    _fit_b_from_samples,
    _fit_pooled_b_with_diagnostics,
    _fit_b_alpha_quadratic_from_samples,
    _fit_b_alpha_sin_from_samples,
    _parse_summary_file,
    _harmonic_statistics,
    _pooled_negative_log_likelihood,
    _pooled_nll_derivative,
    _rr_polar_histogram,
    ExitSample,
)


def test_parse_summary_file_extracts_backscattered_second_row(tmp_path: Path):
    summary_path = tmp_path / "30deg-r1_summary.dat"
    summary_path.write_text(
        "\n".join(
            [
                '"BackScattering Coefficient"\t0.3',
                '"nb backscattered"\t1',
                '"nb transmitted"\t1',
                '"Trajectory"\t0',
                '"Type : "',
                "\tBackscattered",
                '"Collisions"',
                '"X"\t"Y"\t"Z"\t"Rx"\t"Ry"\t"Rz"\t"segment_length"\t"Energy"\t"CollisionType"',
                "0\t0\t0\t0\t0\t1\t5\t60\tRegion",
                "1\t2\t3\t0.4\t0.0\t-0.8\t10\t42\tRegion",
                '"Trajectory"\t1',
                '"Type : "',
                "\tSimple",
                '"Collisions"',
                "0\t0\t0\t0\t0\t1\t5\t60\tRegion",
                '"Trajectory"\t2',
                '"Type : "',
                "\tTransmitted",
                '"Collisions"',
                "0\t0\t0\t0\t0\t1\t5\t60\tRegion",
                "1\t2\t3\t0.0\t0.0\t1.0\t10\t55\tRegion",
            ]
        ),
        encoding="utf-8",
    )

    samples, meta = _parse_summary_file(summary_path, "30deg-r1", nominal_tilt_deg=30.0, repeat_id=1)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.run_id == "30deg-r1"
    assert sample.nominal_tilt_deg == 30.0
    assert sample.trajectory_id == 0
    assert sample.exit_energy_kev == 42.0
    assert math.isclose(sample.rx, 0.4)
    assert math.isclose(sample.rz, -0.8)
    assert meta["trajectory_count"] == 3
    assert meta["backscattered_count"] == 1
    assert meta["transmitted_count"] == 1
    assert meta["simple_count"] == 1
    assert meta["reported_backscatter_count"] == 1
    assert meta["reported_transmitted_count"] == 1
    assert math.isclose(float(meta["frame_handedness"]), 1.0, abs_tol=1e-12)
    assert float(meta["frame_normal_norm_error"]) < 1e-12
    assert float(meta["frame_tangent_dot_abs"]) < 1e-12
    assert len(meta["entry_tangent_1"]) == 3
    assert len(meta["entry_tangent_2"]) == 3


def test_parse_summary_file_preserves_multiple_scan_point_blocks(tmp_path: Path):
    summary_path = tmp_path / "20deg-r1_summary.dat"
    summary_path.write_text(
        "\n".join(
            [
                '"Scan Point"\t0',
                '"X"\t0',
                '"Y"\t0',
                '"Initial Energy"\t60',
                '"Scan Point 0 Results : "',
                '"BackScattering Coefficient"\t0.25',
                '"nb backscattered"\t1',
                '"nb transmitted"\t0',
                '"Trajectory"\t0',
                '"Type : "',
                "\tBackscattered",
                '"Collisions"',
                "0\t0\t0\t0\t0\t1\t5\t60\tRegion",
                "1\t2\t3\t0.4\t0.0\t-0.8\t10\t42\tRegion",
                '"Scan Point"\t1',
                '"X"\t1089',
                '"Y"\t0',
                '"Initial Energy"\t60',
                '"Scan Point 1 Results : "',
                '"BackScattering Coefficient"\t0.30',
                '"nb backscattered"\t1',
                '"nb transmitted"\t1',
                '"Trajectory"\t0',
                '"Type : "',
                "\tBackscattered",
                '"Collisions"',
                "10\t0\t0\t0\t0\t1\t5\t60\tRegion",
                "11\t2\t3\t-0.4\t0.0\t-0.8\t10\t41\tRegion",
                '"Trajectory"\t1',
                '"Type : "',
                "\tTransmitted",
                '"Collisions"',
                "10\t0\t0\t0\t0\t1\t5\t60\tRegion",
                "11\t2\t3\t0.0\t0.0\t1.0\t10\t55\tRegion",
            ]
        ),
        encoding="utf-8",
    )

    samples, meta = _parse_summary_file(summary_path, "20deg-r1", nominal_tilt_deg=20.0, repeat_id=1)

    assert len(samples) == 2
    assert [sample.trajectory_id for sample in samples] == [0, 0]
    assert [sample.scan_point_id for sample in samples] == [0, 1]
    assert [sample.scan_point_run_id for sample in samples] == ["20deg-r1#sp0", "20deg-r1#sp1"]
    assert meta["scan_point_count"] == 2
    assert meta["trajectory_count"] == 3
    assert meta["backscattered_count"] == 2
    assert meta["transmitted_count"] == 1
    assert meta["reported_backscatter_count"] == 2
    assert meta["reported_transmitted_count"] == 1
    assert meta["reported_counts_match"] is True
    assert meta["reported_counts_within_export_tolerance"] is True
    assert [row["scan_point_run_id"] for row in meta["scan_point_blocks"]] == [
        "20deg-r1#sp0",
        "20deg-r1#sp1",
    ]
    assert [row["scan_x"] for row in meta["scan_point_blocks"]] == [0.0, 1089.0]
    assert all(row["initial_energy_kev"] == 60.0 for row in meta["scan_point_blocks"])
    assert all(row["counts_match"] is True for row in meta["scan_point_blocks"])
    assert all(row["count_status"] == "exact_match" for row in meta["scan_point_blocks"])


def test_parse_summary_file_classifies_single_missing_exported_trajectory(tmp_path: Path):
    summary_path = tmp_path / "10deg-r1_summary.dat"
    summary_path.write_text(
        "\n".join(
            [
                '"Scan Point 0 Results : "',
                '"BackScattering Coefficient"\t0.10',
                '"nb backscattered"\t2',
                '"nb transmitted"\t0',
                '"Trajectory"\t0',
                '"Type : "',
                "\tBackscattered",
                '"Collisions"',
                "0\t0\t0\t0\t0\t1\t5\t60\tRegion",
                "1\t0\t0\t0.4\t0.0\t-0.8\t10\t42\tRegion",
                '"Trajectory"\t1',
                '"Type : "',
                "\tSimple",
                '"Collisions"',
                "2\t0\t0\t0\t0\t1\t5\t60\tRegion",
                '"Trajectory"\t2',
                '"Type : "',
                "\tSimple",
                '"Collisions"',
                "4\t0\t0\t0\t0\t1\t5\t60\tRegion",
            ]
        ),
        encoding="utf-8",
    )

    _, meta = _parse_summary_file(summary_path, "10deg-r1", nominal_tilt_deg=10.0, repeat_id=1)

    block = meta["scan_point_blocks"][0]
    assert block["backscatter_count_delta"] == -1
    assert block["counts_match"] is False
    assert block["counts_within_export_tolerance"] is True
    assert block["count_status"] == "within_one_trajectory_export_tolerance"


def test_build_trajectory_angle_audit_recomputes_known_geometry(tmp_path: Path):
    summary_path = tmp_path / "30deg-r1_summary.dat"
    sqrt_three = math.sqrt(3.0)
    normal = np.asarray([0.5, 0.0, -sqrt_three / 2.0])
    tangent_1 = np.asarray([sqrt_three / 2.0, 0.0, 0.5])
    tangent_2 = np.asarray([0.0, -1.0, 0.0])
    phi_rad = math.radians(60.0)
    varphi_rad = math.radians(45.0)
    exit_direction = (
        math.cos(phi_rad) * normal
        + math.sin(phi_rad) * math.cos(varphi_rad) * tangent_1
        + math.sin(phi_rad) * math.sin(varphi_rad) * tangent_2
    )
    summary_path.write_text(
        "\n".join(
            [
                '"Scan Point 0 Results : "',
                '"Trajectory"\t0',
                '"Type : "',
                "\tBackscattered",
                '"Collisions"',
                '"X"\t"Y"\t"Z"\t"Rx"\t"Ry"\t"Rz"\t"segment_length"\t"Energy"\t"CollisionType"',
                "0\t0\t0\t0\t0\t1\t1\t60\tRegion",
                f"1\t1\t1\t{exit_direction[0]:.17g}\t{exit_direction[1]:.17g}\t{exit_direction[2]:.17g}\t1\t50\tRegion",
                '"Trajectory"\t1',
                '"Type : "',
                "\tSimple",
                '"Collisions"',
                "0\t1\t0\t0\t0\t1\t1\t60\tRegion",
                '"Trajectory"\t2',
                '"Type : "',
                "\tSimple",
                '"Collisions"',
                f"{tangent_1[0]:.17g}\t0\t{tangent_1[2]:.17g}\t0\t0\t1\t1\t60\tRegion",
            ]
        ),
        encoding="utf-8",
    )

    audit = build_trajectory_angle_audit(
        summary_path=summary_path,
        run_id="30deg-r1",
        nominal_tilt_deg=30.0,
        repeat_id=1,
        trajectory_id=0,
        scan_point_id=0,
        b_value=-0.75,
    )

    assert math.isclose(audit["angles"]["alpha"]["degrees"], 30.0, abs_tol=1e-8)
    assert math.isclose(audit["angles"]["phi"]["degrees"], 60.0, abs_tol=1e-8)
    assert math.isclose(audit["angles"]["varphi"]["degrees"], 45.0, abs_tol=1e-8)
    assert audit["checks"]["unit_direction_reconstruction_l2_error"] < 1e-12
    assert audit["checks"]["parser_phi_abs_error_rad"] < 1e-12
    assert audit["checks"]["parser_varphi_abs_error_rad"] < 1e-12
    assert audit["rr_terms"]["evaluation"] is not None


def _make_rr_harmonic_samples(*, true_b: float, tilt_deg: float, count: int, seed: int) -> list[ExitSample]:
    rng = np.random.default_rng(seed)
    phi_grid = np.linspace(0.04, math.pi / 2.0 - 0.04, 72)
    varphi_grid = np.linspace(-math.pi, math.pi, 144, endpoint=False)
    p_grid, v_grid = np.meshgrid(phi_grid, varphi_grid, indexing="ij")
    weights = np.cos(p_grid) * (
        1.0 + true_b * math.sin(math.radians(tilt_deg)) * np.sin(p_grid) * np.cos(v_grid)
    )
    probabilities = weights.ravel() / weights.sum()
    chosen = rng.choice(len(probabilities), size=count, replace=True, p=probabilities)
    samples: list[ExitSample] = []
    for trajectory_id, (phi_rad, varphi_rad) in enumerate(
        zip(p_grid.ravel()[chosen], v_grid.ravel()[chosen])
    ):
        samples.append(
            ExitSample(
                run_id=f"{tilt_deg:g}deg-r1",
                tilt_deg=tilt_deg,
                nominal_tilt_deg=tilt_deg,
                repeat_id=1,
                trajectory_id=trajectory_id,
                x_nm=0.0,
                y_nm=0.0,
                z_nm=0.0,
                rx=0.0,
                ry=0.0,
                rz=0.0,
                exit_energy_kev=40.0,
                phi_rad=float(phi_rad),
                varphi_rad=float(varphi_rad),
                cos_phi=float(math.cos(phi_rad)),
            )
        )
    return samples


def test_harmonic_diagnostic_recovers_rr_cosine_and_b():
    samples = _make_rr_harmonic_samples(true_b=0.42, tilt_deg=55.0, count=30000, seed=100)

    stats = _harmonic_statistics(samples)

    assert abs(float(stats["diagnostic_b_measured_tilt"]) - 0.42) < 0.05
    assert abs(float(stats["sin_first"])) < 0.03
    assert float(stats["first_amplitude"]) > 3.0 * float(stats["second_amplitude"])
    assert float(stats["cos_first"]) > 0.0


def test_harmonic_tangent_reversal_changes_phase_not_amplitude():
    samples = _make_rr_harmonic_samples(true_b=-0.50, tilt_deg=60.0, count=24000, seed=101)
    original = _harmonic_statistics(samples)
    varphi = np.asarray([sample.varphi_rad for sample in samples], dtype=np.float64)
    reversed_varphi = (varphi + 2.0 * math.pi) % (2.0 * math.pi) - math.pi
    reversed_stats = _harmonic_statistics(samples, varphi_override=reversed_varphi)

    assert math.isclose(float(reversed_stats["cos_first"]), -float(original["cos_first"]), abs_tol=1e-12)
    assert math.isclose(float(reversed_stats["sin_first"]), -float(original["sin_first"]), abs_tol=1e-12)
    assert math.isclose(
        float(reversed_stats["first_amplitude"]), float(original["first_amplitude"]), abs_tol=1e-12
    )


def test_azimuth_harmonic_diagnostics_include_uniform_control():
    samples = _make_rr_harmonic_samples(true_b=0.55, tilt_deg=60.0, count=30000, seed=102)

    diagnostics = _build_azimuth_harmonic_diagnostics(samples)

    original = diagnostics["controls"]["nonzero_tilt_original"]
    uniform = diagnostics["controls"]["uniform_azimuth"]
    assert float(original["first_amplitude"]) > 0.20
    assert float(uniform["first_amplitude"]) < 0.03
    assert abs(float(diagnostics["controls"]["tangent_reversal_cos_sign_check"])) < 1e-12


def test_coordinate_frame_diagnostics_detect_tangent_flip():
    def frame_run(run_id: str, tilt: float, tangent_sign: float) -> SimpleNamespace:
        return SimpleNamespace(
            run_id=run_id,
            repeat_id=1,
            nominal_tilt_deg=tilt,
            tilt_deg=tilt,
            entry_normal_x=0.0,
            entry_normal_y=0.0,
            entry_normal_z=-1.0,
            entry_tangent_1_x=tangent_sign,
            entry_tangent_1_y=0.0,
            entry_tangent_1_z=0.0,
            entry_tangent_2_x=0.0,
            entry_tangent_2_y=-tangent_sign,
            entry_tangent_2_z=0.0,
            frame_normal_norm_error=0.0,
            frame_tangent_1_norm_error=0.0,
            frame_tangent_2_norm_error=0.0,
            frame_normal_dot_tangent_1_abs=0.0,
            frame_normal_dot_tangent_2_abs=0.0,
            frame_tangent_dot_abs=0.0,
            frame_handedness=1.0,
            projected_beam_tangent_norm=0.5,
            weak_azimuth_orientation=False,
        )

    rows, summary = _build_coordinate_frame_diagnostics(
        [frame_run("10deg-r1", 10.0, 1.0), frame_run("20deg-r1", 20.0, 1.0), frame_run("30deg-r1", 30.0, -1.0)]
    )

    assert len(rows) == 3
    assert summary["continuity_comparison_count"] == 2
    assert summary["tangent_flip_count"] == 1
    assert summary["tangent_flip_run_ids"] == ["30deg-r1"]


def test_chunked_likelihood_derivative_matches_grazing_angle_finite_difference():
    rng = np.random.default_rng(103)
    cos_phi = rng.uniform(0.01, 1.0, size=70001)
    cos_phi[::7] = 1e-12
    asymmetry = rng.uniform(-0.75, 0.75, size=70001)
    b_value = -0.48
    step = 1e-5

    finite_difference = (
        float(_pooled_negative_log_likelihood(cos_phi, asymmetry, b_value + step))
        - float(_pooled_negative_log_likelihood(cos_phi, asymmetry, b_value - step))
    ) / (2.0 * step)
    analytical = float(_pooled_nll_derivative(asymmetry, b_value))

    assert math.isclose(analytical, finite_difference, rel_tol=1e-7, abs_tol=1e-9)


def test_fit_b_from_samples_recovers_known_value():
    rng = np.random.default_rng(42)
    true_b = 0.35
    samples: list[ExitSample] = []
    trajectory_id = 0
    runs = [("30deg-r1", 30.0, 1), ("50deg-r1", 50.0, 1), ("70deg-r1", 70.0, 1)]
    phi_grid = np.linspace(0.05, math.pi / 2.0 - 0.05, 48)
    varphi_grid = np.linspace(-math.pi, math.pi, 96, endpoint=False)
    p_grid, v_grid = np.meshgrid(phi_grid, varphi_grid, indexing="ij")

    for run_id, tilt_deg, repeat_id in runs:
        alpha = math.radians(tilt_deg)
        weights = np.cos(p_grid) * (1.0 + true_b * math.sin(alpha) * np.sin(p_grid) * np.cos(v_grid))
        weights = np.clip(weights, 0.0, None)
        probs = weights.ravel() / weights.sum()
        chosen = rng.choice(len(probs), size=4000, replace=True, p=probs)
        phi_vals = p_grid.ravel()[chosen]
        varphi_vals = v_grid.ravel()[chosen]
        for phi_rad, varphi_rad in zip(phi_vals, varphi_vals):
            cos_phi = math.cos(phi_rad)
            samples.append(
                ExitSample(
                    run_id=run_id,
                    tilt_deg=tilt_deg,
                    nominal_tilt_deg=tilt_deg,
                    repeat_id=repeat_id,
                    trajectory_id=trajectory_id,
                    x_nm=0.0,
                    y_nm=0.0,
                    z_nm=0.0,
                    rx=0.0,
                    ry=0.0,
                    rz=0.0,
                    exit_energy_kev=40.0,
                    phi_rad=float(phi_rad),
                    varphi_rad=float(varphi_rad),
                    cos_phi=float(cos_phi),
                )
            )
            trajectory_id += 1

    fitted_b, objective, _, _ = _fit_b_from_samples(samples, grid_points=2001, equal_run_weight=True)
    assert objective > 0.0
    assert abs(fitted_b - true_b) < 0.05


def test_fit_pooled_b_with_diagnostics_agrees_with_grid_and_derivative():
    rng = np.random.default_rng(123)
    true_b = -0.42
    samples: list[ExitSample] = []
    trajectory_id = 0
    runs = [("20deg-r1", 20.0, 1), ("45deg-r1", 45.0, 1), ("65deg-r1", 65.0, 1)]
    phi_grid = np.linspace(0.05, math.pi / 2.0 - 0.05, 52)
    varphi_grid = np.linspace(-math.pi, math.pi, 104, endpoint=False)
    p_grid, v_grid = np.meshgrid(phi_grid, varphi_grid, indexing="ij")

    for run_id, tilt_deg, repeat_id in runs:
        sin_alpha = math.sin(math.radians(tilt_deg))
        weights = np.cos(p_grid) * (1.0 + true_b * sin_alpha * np.sin(p_grid) * np.cos(v_grid))
        weights = np.clip(weights, 0.0, None)
        probs = weights.ravel() / weights.sum()
        chosen = rng.choice(len(probs), size=3500, replace=True, p=probs)
        for phi_rad, varphi_rad in zip(p_grid.ravel()[chosen], v_grid.ravel()[chosen]):
            samples.append(
                ExitSample(
                    run_id=run_id,
                    tilt_deg=tilt_deg,
                    nominal_tilt_deg=tilt_deg,
                    repeat_id=repeat_id,
                    trajectory_id=trajectory_id,
                    x_nm=0.0,
                    y_nm=0.0,
                    z_nm=0.0,
                    rx=0.0,
                    ry=0.0,
                    rz=0.0,
                    exit_energy_kev=40.0,
                    phi_rad=float(phi_rad),
                    varphi_rad=float(varphi_rad),
                    cos_phi=float(math.cos(phi_rad)),
                )
            )
            trajectory_id += 1

    fit, _, objective_grid, derivative_grid, _ = _fit_pooled_b_with_diagnostics(samples, grid_points=2401)

    assert abs(float(fit["fitted_b"]) - true_b) < 0.05
    assert abs(float(fit["root_grid_delta_b"])) < 0.01
    assert abs(float(fit["derivative_at_fitted_b"])) < 1e-6
    assert float(fit["second_derivative_at_fitted_b"]) > 0.0
    assert float(fit["negative_log_likelihood"]) <= float(np.min(objective_grid)) + 1e-4
    assert np.min(np.abs(derivative_grid)) < 1e-3
    assert float(fit["observed_min_density_margin"]) > 0.0
    assert float(fit["domain_min_density_margin"]) > -1e-6
    assert fit["approx_standard_error"] is not None


def test_rr_polar_histogram_is_normalized():
    centers_deg = np.arange(0.0, 91.0, 1.0, dtype=np.float64)
    hist = _rr_polar_histogram(centers_deg)
    assert hist.shape == centers_deg.shape
    assert np.all(hist >= 0.0)
    assert math.isclose(float(hist.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_rr_density_normalization_check_near_one():
    report = _build_rr_normalization_check(np.asarray([0.0, 25.0, 45.0, 70.0], dtype=np.float64), b_value=0.35)
    assert report["max_abs_error"] is not None
    assert float(report["max_abs_error"]) < 5e-3
    assert len(report["rows"]) == 4


def test_fit_b_alpha_sin_from_samples_recovers_known_law():
    rng = np.random.default_rng(7)
    true_b0 = 0.8
    true_b1 = -2.1
    samples: list[ExitSample] = []
    trajectory_id = 0
    runs = [
        ("30deg-r1", 30.0, 1),
        ("40deg-r1", 40.0, 1),
        ("45deg-r1", 45.0, 1),
        ("50deg-r1", 50.0, 1),
        ("60deg-r1", 60.0, 1),
        ("70deg-r1", 70.0, 1),
    ]
    phi_grid = np.linspace(0.05, math.pi / 2.0 - 0.05, 56)
    varphi_grid = np.linspace(-math.pi, math.pi, 112, endpoint=False)
    p_grid, v_grid = np.meshgrid(phi_grid, varphi_grid, indexing="ij")

    for run_id, tilt_deg, repeat_id in runs:
        sin_alpha = math.sin(math.radians(tilt_deg))
        effective_b = true_b0 + true_b1 * sin_alpha
        weights = np.cos(p_grid) * (1.0 + effective_b * sin_alpha * np.sin(p_grid) * np.cos(v_grid))
        weights = np.clip(weights, 0.0, None)
        probs = weights.ravel() / weights.sum()
        chosen = rng.choice(len(probs), size=3500, replace=True, p=probs)
        phi_vals = p_grid.ravel()[chosen]
        varphi_vals = v_grid.ravel()[chosen]
        for phi_rad, varphi_rad in zip(phi_vals, varphi_vals):
            samples.append(
                ExitSample(
                    run_id=run_id,
                    tilt_deg=tilt_deg,
                    nominal_tilt_deg=tilt_deg,
                    repeat_id=repeat_id,
                    trajectory_id=trajectory_id,
                    x_nm=0.0,
                    y_nm=0.0,
                    z_nm=0.0,
                    rx=0.0,
                    ry=0.0,
                    rz=0.0,
                    exit_energy_kev=40.0,
                    phi_rad=float(phi_rad),
                    varphi_rad=float(varphi_rad),
                    cos_phi=float(math.cos(phi_rad)),
                )
            )
            trajectory_id += 1

    fit, prediction_rows, _ = _fit_b_alpha_sin_from_samples(
        samples,
        scalar_grid_points=2001,
        law_search_points=301,
        equal_run_weight=True,
    )
    assert fit["negative_log_likelihood"] > 0.0
    prediction_errors = []
    for row in prediction_rows:
        sin_alpha = math.sin(math.radians(row["measured_tilt_deg"]))
        true_b = true_b0 + true_b1 * sin_alpha
        prediction_errors.append(abs(row["predicted_b"] - true_b))
    assert float(np.mean(prediction_errors)) < 0.1
    assert max(prediction_errors) < 0.15


def test_fit_b_alpha_quadratic_from_samples_recovers_known_law():
    rng = np.random.default_rng(11)
    true_b0 = 0.45
    true_b1 = 0.9
    true_b2 = -2.3
    samples: list[ExitSample] = []
    trajectory_id = 0
    runs = [
        ("30deg-r1", 30.0, 1),
        ("35deg-r1", 35.0, 1),
        ("40deg-r1", 40.0, 1),
        ("45deg-r1", 45.0, 1),
        ("50deg-r1", 50.0, 1),
        ("60deg-r1", 60.0, 1),
        ("70deg-r1", 70.0, 1),
    ]
    phi_grid = np.linspace(0.05, math.pi / 2.0 - 0.05, 60)
    varphi_grid = np.linspace(-math.pi, math.pi, 120, endpoint=False)
    p_grid, v_grid = np.meshgrid(phi_grid, varphi_grid, indexing="ij")

    for run_id, tilt_deg, repeat_id in runs:
        s = math.sin(math.radians(tilt_deg))
        effective_b = true_b0 + true_b1 * s + true_b2 * s * s
        weights = np.cos(p_grid) * (1.0 + effective_b * s * np.sin(p_grid) * np.cos(v_grid))
        weights = np.clip(weights, 0.0, None)
        probs = weights.ravel() / weights.sum()
        chosen = rng.choice(len(probs), size=3200, replace=True, p=probs)
        phi_vals = p_grid.ravel()[chosen]
        varphi_vals = v_grid.ravel()[chosen]
        for phi_rad, varphi_rad in zip(phi_vals, varphi_vals):
            samples.append(
                ExitSample(
                    run_id=run_id,
                    tilt_deg=tilt_deg,
                    nominal_tilt_deg=tilt_deg,
                    repeat_id=repeat_id,
                    trajectory_id=trajectory_id,
                    x_nm=0.0,
                    y_nm=0.0,
                    z_nm=0.0,
                    rx=0.0,
                    ry=0.0,
                    rz=0.0,
                    exit_energy_kev=40.0,
                    phi_rad=float(phi_rad),
                    varphi_rad=float(varphi_rad),
                    cos_phi=float(math.cos(phi_rad)),
                )
            )
            trajectory_id += 1

    fit, prediction_rows, _ = _fit_b_alpha_quadratic_from_samples(
        samples,
        scalar_grid_points=2001,
        equal_run_weight=True,
    )
    assert fit["negative_log_likelihood"] > 0.0
    prediction_errors = []
    for row in prediction_rows:
        s = math.sin(math.radians(row["measured_tilt_deg"]))
        true_b = true_b0 + true_b1 * s + true_b2 * s * s
        prediction_errors.append(abs(row["predicted_b"] - true_b))
    assert float(np.mean(prediction_errors)) < 0.08
    assert max(prediction_errors) < 0.12


def test_build_constant_vs_quadratic_per_tilt_rows_marks_winners():
    report = {
        "fit": {
            "global_equal_run": {
                "fitted_b": 0.2,
            },
            "b_alpha_quadratic_equal_run": {
                "formula": "B(alpha_deg) = b0 + b1 * sin(alpha_deg) + b2 * sin(alpha_deg)^2",
            },
            "per_tilt": [
                {
                    "nominal_tilt_deg": 30.0,
                    "measured_tilt_deg_mean": 30.2,
                    "fitted_b": 0.25,
                    "sample_count": 100,
                    "run_count": 5,
                },
                {
                    "nominal_tilt_deg": 60.0,
                    "measured_tilt_deg_mean": 60.1,
                    "fitted_b": -0.55,
                    "sample_count": 120,
                    "run_count": 5,
                },
            ],
            "b_alpha_quadratic_per_tilt": [
                {
                    "nominal_tilt_deg": 30.0,
                    "predicted_b": 0.24,
                },
                {
                    "nominal_tilt_deg": 60.0,
                    "predicted_b": 0.30,
                },
            ],
        }
    }

    rows, summary = _build_constant_vs_quadratic_per_tilt_rows(report)

    assert len(rows) == 2
    assert summary["tilt_count"] == 2
    assert summary["quadratic_win_count"] == 1
    assert summary["constant_win_count"] == 1
    assert summary["tie_count"] == 0
    assert rows[0]["winner"] == "quadratic"
    assert rows[1]["winner"] == "constant"
