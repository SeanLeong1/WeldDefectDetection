from __future__ import annotations

import numpy as np

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.models import DetectorRuntime, MaterialPreset
from bse_renderer.physics.bse_yield import eta_z
from bse_renderer.renderer.quadrant_detector import compute_quadrant_signals


def _material() -> MaterialPreset:
    return MaterialPreset(name="test", z=26.0, a=55.845, rho=7.9, b=0.2, n=1.75)


def _detector(sensitivity: dict[str, float] | None = None) -> DetectorRuntime:
    return DetectorRuntime(
        height_mm=12.0,
        radius_mm=12.0,
        inner_radius_mm=4.0,
        outer_radius_mm=16.0,
        quadrants_deg={"A": 0.0, "B": 90.0, "C": 180.0, "D": 270.0},
        sensitivity=sensitivity or {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0},
        psi0_rad=float(np.arctan2(12.0, 12.0)),
    )


def _signals(normals: np.ndarray, detector: DetectorRuntime, **kwargs) -> np.ndarray:
    shape = normals.shape[:2]
    return compute_quadrant_signals(
        normals=normals,
        detector=detector,
        material=_material(),
        material_z_map=np.full(shape, 26.0, dtype=np.float32),
        b_coeff=0.2,
        **kwargs,
    )


def test_flat_surface_has_equal_integrated_quadrants():
    normals = np.zeros((3, 4, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    signals = _signals(normals, _detector())

    assert np.allclose(signals[..., 0], signals[..., 1], rtol=1e-6, atol=1e-8)
    assert np.allclose(signals[..., 1], signals[..., 2], rtol=1e-6, atol=1e-8)
    assert np.allclose(signals[..., 2], signals[..., 3], rtol=1e-6, atol=1e-8)


def test_flat_quadrants_sum_to_full_annulus_integral():
    normals = np.zeros((1, 1, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    detector = _detector()
    signals = _signals(normals, detector)[0, 0]

    phi_inner = np.arctan2(detector.inner_radius_mm, detector.height_mm)
    phi_outer = np.arctan2(detector.outer_radius_mm, detector.height_mm)
    nodes, weights = np.polynomial.legendre.leggauss(40)
    phi = 0.5 * (phi_outer - phi_inner) * nodes + 0.5 * (phi_outer + phi_inner)
    phi_weights = 0.5 * (phi_outer - phi_inner) * weights
    eta = float(eta_z(np.asarray(26.0)))
    expected = (
        2.0
        * np.pi
        * eta
        * np.sum(
            (
                0.58 * np.cos(phi)
                + 0.42 * np.power(np.cos(phi), _material().n)
            )
            * np.sin(phi)
            * phi_weights
        )
    )
    assert np.isclose(float(signals.sum()), float(expected), rtol=2e-5)


def test_quadrant_sensitivity_is_applied_to_only_that_channel():
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    baseline = _signals(normals, _detector())
    sensitive = _signals(
        normals,
        _detector({"A": 1.4, "B": 1.0, "C": 1.0, "D": 1.0}),
    )

    assert np.allclose(sensitive[..., 0], 1.4 * baseline[..., 0], rtol=1e-6)
    assert np.allclose(sensitive[..., 1:], baseline[..., 1:], rtol=1e-6)


def test_rotating_surface_gradient_permutes_integrated_quadrants():
    tilt = np.radians(25.0)
    x_normal = np.asarray([np.sin(tilt), 0.0, np.cos(tilt)], dtype=np.float32)
    y_normal = np.asarray([0.0, np.sin(tilt), np.cos(tilt)], dtype=np.float32)
    x_signals = _signals(x_normal.reshape(1, 1, 3), _detector())[0, 0]
    y_signals = _signals(y_normal.reshape(1, 1, 3), _detector())[0, 0]

    assert np.allclose(y_signals, np.roll(x_signals, 1), rtol=2e-5, atol=1e-7)


def test_annular_quadrature_converges():
    tilt = np.radians(37.0)
    normal = np.asarray([np.sin(tilt), 0.0, np.cos(tilt)], dtype=np.float32)
    normals = normal.reshape(1, 1, 3)
    medium = _signals(
        normals,
        _detector(),
        quadrature_order_phi=6,
        quadrature_order_varphi=8,
    )
    fine = _signals(
        normals,
        _detector(),
        quadrature_order_phi=10,
        quadrature_order_varphi=14,
    )

    assert np.allclose(medium, fine, rtol=2e-5, atol=1e-7)


def test_sampled_detector_center_always_lies_in_annulus():
    bundle = load_config_bundle("config")
    for seed in range(250):
        detector = sample_parameters(bundle, seed=seed, max_defects=0).detector
        assert detector.inner_radius_mm <= detector.radius_mm <= detector.outer_radius_mm
