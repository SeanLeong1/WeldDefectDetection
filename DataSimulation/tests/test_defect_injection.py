from pathlib import Path

import numpy as np

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.geometry.defect_injection import apply_defect_plan, parse_defect_plan
from bse_renderer.geometry.surface_mesh import build_surface_from_config


def test_apply_defects_mutates_surface():
    from pathlib import Path

    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=123, max_defects=3)
    surface = build_surface_from_config(sampled.beam, image_size=128)
    before = surface.heightmap_mm.copy()
    defects = apply_defect_plan(surface, sampled.defects, np.random.default_rng(123))
    if defects:
        assert not np.allclose(before, surface.heightmap_mm)
        assert surface.defect_mask.any()
    else:
        assert not surface.defect_mask.any()


def test_defects_populate_section_plane_fields():
    cfg = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(cfg, seed=123, max_defects=0)
    surface = build_surface_from_config(sampled.beam, image_size=128)
    defects = apply_defect_plan(
        surface,
        parse_defect_plan("intrinsic_pore(diameter_mm=1.0,depth_mm=0.8);lack_of_fusion(extent_mm=2.0,depth_mm=0.7,irregularity=0.4)"),
        np.random.default_rng(123),
    )
    assert defects
    assert float(surface.section_void_fraction.max()) > 0.0
    assert float(surface.section_yield_modifier.min()) < 1.0
    assert not np.allclose(surface.section_relief_mm, 0.0)
