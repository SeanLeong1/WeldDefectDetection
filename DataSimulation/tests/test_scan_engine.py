from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.renderer.scan_engine import ScanEngine


def test_scan_engine_runs():
    bundle = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(bundle, seed=99, max_defects=3)
    result = ScanEngine(image_size=96).run(sampled)
    assert result.channels.compo.shape == (96, 96)
    assert result.channels.topo.shape == (96, 96)
    assert result.channels.tx.shape == (96, 96)
    assert result.channels.ty.shape == (96, 96)
    assert result.defect_mask.shape == (96, 96)
    assert "display_mapping" in result.metadata
    assert "detector_frame" in result.metadata
    assert "section_plane_weight" in result.metadata
    assert "ebw_seam_depth_mm" in result.metadata
    assert "ebw_reinforcement_mm" in result.metadata
    assert result.metadata["ebw_seam_depth_mm"] > 0.0


def test_heightmap_normals_match_analytical_plane():
    pixel_size_mm = 0.05
    y_mm, x_mm = np.indices((41, 47), dtype=np.float64) * pixel_size_mm
    slope_x = 0.32
    slope_y = -0.18
    heightmap = slope_x * x_mm + slope_y * y_mm

    normals = ScanEngine._normals_from_heightmap(heightmap, pixel_size_mm)
    expected = np.asarray([-slope_x, -slope_y, 1.0], dtype=np.float64)
    expected /= np.linalg.norm(expected)

    assert np.allclose(normals, expected, atol=1e-6)


def test_heightmap_normals_are_resolution_invariant():
    slope_x = -0.27
    slope_y = 0.11

    def plane(pixel_size_mm: float, shape: tuple[int, int]) -> np.ndarray:
        y_mm, x_mm = np.indices(shape, dtype=np.float64) * pixel_size_mm
        return slope_x * x_mm + slope_y * y_mm

    coarse = ScanEngine._normals_from_heightmap(plane(0.10, (31, 31)), 0.10)
    fine = ScanEngine._normals_from_heightmap(plane(0.025, (121, 121)), 0.025)

    assert np.allclose(coarse[15, 15], fine[60, 60], atol=1e-6)


def test_heightmap_normals_match_sinusoidal_gradient_away_from_edges():
    pixel_size_mm = 0.01
    height, width = 121, 161
    y_mm, x_mm = np.indices((height, width), dtype=np.float64) * pixel_size_mm
    amplitude_mm = 0.08
    wavelength_mm = 0.8
    heightmap = amplitude_mm * np.sin(2.0 * math.pi * x_mm / wavelength_mm)

    normals = ScanEngine._normals_from_heightmap(heightmap, pixel_size_mm)
    expected_dx = (
        amplitude_mm
        * 2.0
        * math.pi
        / wavelength_mm
        * np.cos(2.0 * math.pi * x_mm / wavelength_mm)
    )
    expected = np.stack(
        [-expected_dx, np.zeros_like(expected_dx), np.ones_like(expected_dx)], axis=-1
    )
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)

    assert np.allclose(normals[2:-2, 2:-2], expected[2:-2, 2:-2], atol=8e-4)


def test_heightmap_normals_reject_nonpositive_spacing():
    with np.testing.assert_raises(ValueError):
        ScanEngine._normals_from_heightmap(np.zeros((3, 3), dtype=np.float32), 0.0)


def test_scan_engine_session_state_changes_output():
    bundle = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(bundle, seed=1234, max_defects=0)
    sampled = replace(
        sampled,
        noise=replace(
            sampled.noise,
            thermal_enabled=True,
            contamination_enabled=True,
        ),
    )
    engine = ScanEngine(image_size=96)

    early = engine.run(
        sampled,
        session_state={
            "progress": 0.0,
            "thermal_baseline": 0.0,
            "quadrant_mismatch": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
        },
    )
    late = engine.run(
        sampled,
        session_state={
            "progress": 1.0,
            "thermal_baseline": 0.01,
            "quadrant_mismatch": {"A": 0.08, "B": -0.08, "C": 0.0, "D": 0.0},
        },
    )
    assert not (early.channels.compo == late.channels.compo).all()
    assert late.metadata["session_progress"] == 1.0
