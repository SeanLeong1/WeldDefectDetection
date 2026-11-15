import numpy as np
import pytest

from bse_renderer.physics.emission_model import emission_distribution, solid_angle_integral


def test_lambert_mode_positive():
    phi = np.linspace(0, np.pi / 2, 16)
    varphi = np.linspace(0, 2 * np.pi, 16)
    p, v = np.meshgrid(phi, varphi, indexing="ij")
    d = emission_distribution(p, v, alpha_rad=0.3, b_coeff=0.0)
    assert np.all(d >= 0.0)


def test_solid_angle_integral_shape():
    alpha = np.zeros((8, 8), dtype=np.float32)
    b = np.full((8, 8), 0.2, dtype=np.float32)
    integ = solid_angle_integral(
        phi_min=0.1,
        phi_max=0.7,
        varphi_min=-0.5,
        varphi_max=0.5,
        alpha_rad=alpha,
        b_coeff=b,
    )
    assert integ.shape == (8, 8)
    assert np.all(integ >= 0.0)


def test_invalid_b_density_raises_instead_of_clipping():
    with pytest.raises(ValueError, match="global density positivity"):
        emission_distribution(
            phi_rad=np.pi / 2.0,
            varphi_rad=np.pi,
            alpha_rad=np.pi / 2.0,
            b_coeff=1.1,
        )


def test_phi_outside_outward_hemisphere_raises():
    with pytest.raises(ValueError, match="outward hemisphere"):
        emission_distribution(
            phi_rad=np.pi / 2.0 + 0.01,
            varphi_rad=0.0,
            alpha_rad=0.0,
            b_coeff=0.0,
        )
