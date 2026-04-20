from pathlib import Path

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


def test_scan_engine_session_state_changes_output():
    bundle = load_config_bundle(Path(__file__).resolve().parents[1] / "config")
    sampled = sample_parameters(bundle, seed=1234, max_defects=0)
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
