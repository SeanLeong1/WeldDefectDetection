from pathlib import Path

from bse_renderer.config_loader import load_config_bundle


def test_load_config_bundle():
    root = Path(__file__).resolve().parents[1]
    bundle = load_config_bundle(root / "config")
    assert bundle.beam.accelerating_voltage_kv.minimum == 30
    assert bundle.material.default_material in bundle.material.presets
    assert set(bundle.detector.quadrants_deg) == {"A", "B", "C", "D"}
    assert "keyhole_pore" in bundle.defect.probabilities
    assert bundle.slice.voxel_depth_mm.default > 0.0
    assert bundle.slice.slice_tilt_deg.maximum >= bundle.slice.slice_tilt_deg.default
