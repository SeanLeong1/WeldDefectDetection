import numpy as np

from bse_renderer.renderer.image_formation import form_channels


def test_image_formation_outputs():
    q = np.zeros((32, 32, 4), dtype=np.float32)
    q[..., 0] = 1.0
    q[..., 1] = 2.0
    q[..., 2] = 1.0
    q[..., 3] = 2.0
    channels = form_channels(q)
    assert channels.compo.shape == (32, 32)
    assert channels.topo.shape == (32, 32)
    assert channels.rx.shape == (32, 32)
    assert np.allclose(channels.compo, 6.0)
    assert np.allclose(channels.topo, 0.0)
