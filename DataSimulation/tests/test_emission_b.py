from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from bse_renderer.renderer.emission_b import (
    EMISSION_B_MODE_CONSTANT,
    EMISSION_B_MODE_LAMBERTIAN,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
    load_emission_b_fit,
    validate_emission_b_mode,
)
from bse_renderer.renderer.scan_engine import ScanEngine


def test_validate_emission_b_mode_defaults_to_lambertian():
    assert validate_emission_b_mode(None) == EMISSION_B_MODE_LAMBERTIAN
    assert validate_emission_b_mode("pooled_constant_fit") == EMISSION_B_MODE_POOLED_CONSTANT_FIT
    assert validate_emission_b_mode("b_alpha_sin") == EMISSION_B_MODE_SIN_FIT
    assert validate_emission_b_mode("quadratic_fit") == EMISSION_B_MODE_QUADRATIC_FIT
    assert validate_emission_b_mode("lambertian") == EMISSION_B_MODE_LAMBERTIAN


def test_scan_engine_lambertian_mode_is_explicit_zero_b():
    engine = ScanEngine(allow_unvalidated_emission_b=True, render_model_mode="formula_only", emission_b_mode="lambertian")
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    b_coeff, meta = engine._resolve_emission_b_coeff(
        normals, material_b=0.19, incidence_tilt_deg=45.0
    )

    assert b_coeff == 0.0
    assert meta["source_type"] == "explicit_uncalibrated_lambertian_fallback"
    assert meta["b_stats"] == {"min": 0.0, "max": 0.0, "mean": 0.0}


def test_load_pooled_constant_emission_b_fit_from_gui_summary(tmp_path: Path):
    path = tmp_path / "b_fit_gui_summary.json"
    path.write_text(
        """
        {
          "model_comparison": {
            "constant_b": {
              "fitted_b": 0.245,
              "negative_log_likelihood": 1.234
            }
          }
        }
        """,
        encoding="utf-8",
    )
    fit = load_emission_b_fit(path, EMISSION_B_MODE_POOLED_CONSTANT_FIT)
    assert fit.emission_b_mode == EMISSION_B_MODE_POOLED_CONSTANT_FIT
    assert math.isclose(float(fit.fitted_b), 0.245)
    values = fit.evaluate(np.asarray([0.0, 30.0, 60.0], dtype=np.float32))
    assert values.shape == (3,)
    assert np.allclose(values, 0.245)


def test_load_pooled_constant_emission_b_fit_from_report(tmp_path: Path):
    path = tmp_path / "b_fit_report.json"
    path.write_text(
        """
        {
          "fit": {
            "global_pooled": {
              "fitted_b": 0.31,
              "negative_log_likelihood": 0.91
            }
          }
        }
        """,
        encoding="utf-8",
    )
    fit = load_emission_b_fit(path, EMISSION_B_MODE_POOLED_CONSTANT_FIT)
    assert fit.emission_b_mode == EMISSION_B_MODE_POOLED_CONSTANT_FIT
    assert math.isclose(float(fit.fitted_b), 0.31)


def test_load_quadratic_emission_b_fit_from_report_prefers_sample_weighted(tmp_path: Path):
    path = tmp_path / "b_fit_report.json"
    path.write_text(
        """
        {
          "fit": {
            "b_alpha_quadratic_equal_run": {
              "b0": 0.9,
              "b1": -1.8,
              "b2": -0.2,
              "formula": "equal"
            },
            "b_alpha_quadratic_sample_weighted": {
              "b0": 0.7,
              "b1": -1.4,
              "b2": -0.1,
              "formula": "sample_weighted"
            }
          }
        }
        """,
        encoding="utf-8",
    )
    fit = load_emission_b_fit(path, EMISSION_B_MODE_QUADRATIC_FIT)
    assert fit.emission_b_mode == EMISSION_B_MODE_QUADRATIC_FIT
    assert math.isclose(float(fit.b0), 0.7)
    assert math.isclose(float(fit.b1), -1.4)
    assert math.isclose(float(fit.b2), -0.1)
    assert fit.formula == "sample_weighted"


def test_load_sine_emission_b_fit_from_gui_summary(tmp_path: Path):
    path = tmp_path / "b_fit_gui_summary.json"
    path.write_text(
        json.dumps(
            {
                "model_comparison": {
                    "b_alpha_sin": {
                        "fit": {
                            "b0": 1.0,
                            "b1": -2.0,
                            "formula": "B(alpha) = 1 - 2 sin(alpha)",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    fit = load_emission_b_fit(path, EMISSION_B_MODE_SIN_FIT)
    assert fit.emission_b_mode == EMISSION_B_MODE_SIN_FIT
    assert np.allclose(fit.evaluate(np.asarray([0.0, 30.0, 90.0])), [1.0, 0.0, -1.0])


def test_scan_engine_sine_emission_b_uses_local_surface_tilt(tmp_path: Path):
    path = tmp_path / "b_fit_gui_summary.json"
    path.write_text(
        json.dumps(
            {
                "model_comparison": {
                    "b_alpha_sin": {"fit": {"b0": 1.0, "b1": -2.0}}
                }
            }
        ),
        encoding="utf-8",
    )
    engine = ScanEngine(allow_unvalidated_emission_b=True, 
        render_model_mode="formula_only",
        emission_b_mode=EMISSION_B_MODE_SIN_FIT,
        emission_b_path=str(path),
    )
    normals = np.zeros((1, 2, 3), dtype=np.float32)
    normals[0, 0, 2] = 1.0
    normals[0, 1, 2] = 0.5
    b_coeff, meta = engine._resolve_emission_b_coeff(
        normals, material_b=0.19, incidence_tilt_deg=0.0
    )
    assert isinstance(b_coeff, np.ndarray)
    assert np.allclose(b_coeff, [[1.0, 1.0 - 2.0 * np.sin(np.radians(60.0))]])
    assert meta["b_stats"]["min"] < meta["b_stats"]["max"]


def test_scan_engine_pooled_constant_emission_b_ignores_incidence_tilt(tmp_path: Path):
    path = tmp_path / "b_fit_gui_summary.json"
    path.write_text(
        json.dumps(
            {
                "model_comparison": {
                    "constant_b": {
                        "fitted_b": 0.28,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    engine = ScanEngine(allow_unvalidated_emission_b=True, 
        render_model_mode="formula_only",
        emission_b_mode=EMISSION_B_MODE_POOLED_CONSTANT_FIT,
        emission_b_path=str(path),
    )
    normals = np.dstack(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.full((2, 2), 0.25, dtype=np.float32),
        ]
    )

    b_coeff, meta = engine._resolve_emission_b_coeff(
        normals,
        material_b=0.19,
        incidence_tilt_deg=47.0,
    )

    assert isinstance(b_coeff, float)
    assert math.isclose(b_coeff, 0.28, abs_tol=1e-6)
    assert meta["mode"] == EMISSION_B_MODE_POOLED_CONSTANT_FIT
    assert meta["source_type"] == "b_fit_gui_summary"
    assert math.isclose(meta["incidence_tilt_deg"], 47.0, abs_tol=1e-9)
    assert math.isclose(meta["b_stats"]["mean"], 0.28, abs_tol=1e-6)


def test_scan_engine_quadratic_emission_b_uses_incidence_tilt(tmp_path: Path):
    path = tmp_path / "b_fit_gui_summary.json"
    path.write_text(
        json.dumps(
            {
                "model_comparison": {
                    "b_alpha_quadratic": {
                        "fit": {
                            "b0": 1.0,
                            "b1": -2.0,
                            "b2": 0.0,
                            "formula": "B(alpha_deg) = b0 + b1 * sin(alpha_deg) + b2 * sin(alpha_deg)^2",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    engine = ScanEngine(allow_unvalidated_emission_b=True, 
        render_model_mode="formula_only",
        emission_b_mode=EMISSION_B_MODE_QUADRATIC_FIT,
        emission_b_path=str(path),
    )
    normals = np.dstack(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.full((2, 2), 0.25, dtype=np.float32),
        ]
    )

    b_coeff, meta = engine._resolve_emission_b_coeff(
        normals,
        material_b=0.19,
        incidence_tilt_deg=30.0,
    )

    assert isinstance(b_coeff, np.ndarray)
    expected = 1.0 - 2.0 * np.sin(np.arccos(0.25))
    assert np.allclose(b_coeff, expected)
    assert meta["mode"] == EMISSION_B_MODE_QUADRATIC_FIT
    assert math.isclose(meta["incidence_tilt_deg"], 30.0, abs_tol=1e-9)
    assert math.isclose(meta["b_stats"]["mean"], expected, abs_tol=1e-6)
