import numpy as np

from bse_renderer.physics.bse_yield import eta_oblique, eta_void, eta_z


def test_eta_z_bounds():
    z = np.array([6, 13, 26, 29, 79], dtype=np.float64)
    eta = eta_z(z)
    assert eta.shape == z.shape
    assert np.all(eta >= 0.0)
    assert np.all(eta <= 1.0)


def test_eta_oblique_increases_with_angle():
    eta0 = np.array([0.2, 0.3], dtype=np.float64)
    z = np.array([26.0, 26.0], dtype=np.float64)
    alpha_small = np.radians(5.0)
    alpha_large = np.radians(25.0)
    e_small = eta_oblique(eta0, alpha_small, z)
    e_large = eta_oblique(eta0, alpha_large, z)
    assert np.all(e_large >= e_small)


def test_eta_void():
    assert eta_void() == 0.0
    z = eta_void((4, 5))
    assert z.shape == (4, 5)
    assert float(z.sum()) == 0.0

