import numpy as np

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.geometry.surface_mesh import build_surface_from_config
from bse_renderer.renderer.quadrant_detector import compute_quadrant_signals


def test_quadrant_shape_and_nonnegative():
    from pathlib import Path

    bundle = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(bundle, seed=7, max_defects=0)
    surface = build_surface_from_config(sampled.beam, image_size=64)
    q = compute_quadrant_signals(
        normals=surface.normals,
        detector=sampled.detector,
        material=sampled.material,
        material_z_map=surface.material_z,
    )
    assert q.shape == (64, 64, 4)
    assert np.all(q >= 0.0)

