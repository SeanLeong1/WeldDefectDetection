import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.batch_generator import BatchGenerator, BatchOptions
from bse_renderer.domain_randomisation.dataset_forms import generate_topo_sequence_dataset, _freeze_generation_provenance
from bse_renderer.geometry.surface_mesh import _initial_normals
from bse_renderer.physics.emission_model import emission_distribution, solid_angle_integral
from bse_renderer.renderer.calibration import RendererCalibration, ChannelCalibration, GeometryResponseProfile, load_renderer_calibration
from bse_renderer.renderer.emission_b import load_emission_b_fit, qualify_emission_b_fit
from bse_renderer.renderer.quadrant_detector import _integrated_quadrant_directional_term
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.validation.casino_adapter import parse_casino_case_output
from bse_renderer.validation.casino_calibration import _fit_linear, run_casino_calibration
from bse_renderer.validation.casino_emission_fit import _sample_emission_components
from bse_renderer.validation.standards import _expand_mask


def test_local_detector_integral_matches_independent_ray_dot_products():
    alpha = np.radians(30.)
    normal = np.array([np.sin(alpha), 0, np.cos(alpha)])
    actual = _integrated_quadrant_directional_term(normal[None, None], np.array([[alpha]]),
        np.zeros((1, 1)), 0., np.radians(35), np.radians(45), 0., 1., 1., 12, 16)
    # For B=0, n=1, gain=1 the blend reduces exactly to n dot ray.
    lo, hi = np.radians([35, 45])
    az_integral_cos = 2 * np.sin(np.pi / 4)
    polar_sin2 = (hi-lo)/2 - (np.sin(2*hi)-np.sin(2*lo))/4
    polar_sincos = (np.sin(hi)**2-np.sin(lo)**2)/2
    expected = np.sin(alpha)*az_integral_cos*polar_sin2 + np.cos(alpha)*(np.pi/2)*polar_sincos
    assert float(actual.item()) == pytest.approx(expected, rel=1e-10)


def test_back_facing_detector_patch_contributes_zero():
    alpha = np.radians(80.)
    normal = np.array([np.sin(alpha), 0, np.cos(alpha)])[None, None]
    actual = _integrated_quadrant_directional_term(normal, np.array([[alpha]]), np.zeros((1, 1)),
        np.pi, np.radians(60), np.radians(70), 0., 1., 1., 8, 12)
    assert actual.item() == 0


@pytest.mark.parametrize("b,alpha", [(2., 60.), (np.nan, 30.), (.2, np.nan)])
def test_global_density_gate_even_when_sample_is_positive(b, alpha):
    with pytest.raises(ValueError):
        emission_distribution(0., 0., np.radians(alpha), b)


def test_hemisphere_integral_conserves_cosine_law():
    integral = solid_angle_integral(0., np.pi/2, -np.pi, np.pi, np.radians(60), .8)
    assert integral.item() == pytest.approx(np.pi, rel=1e-7)


def test_mesh_normals_use_physical_spacing():
    spacing = .01
    height = np.tile(np.arange(8)*spacing*np.tan(np.radians(30)), (7, 1))
    assert np.allclose(_initial_normals(height, spacing), [-.5, 0., np.sqrt(.75)], atol=1e-7)


def test_proxy_dilation_does_not_wrap():
    mask = np.zeros((7, 7), bool); mask[0, 0] = True
    expanded = _expand_mask(mask, radius=2)
    assert not expanded[-1].any() and not expanded[:, -1].any()
    assert expanded[0, 2] and expanded[2, 0]


def test_injected_calibration_obeys_render_mode():
    base = RendererCalibration.identity()
    cal = replace(base, compo=ChannelCalibration(3, 2), surrogate=replace(base.surrogate,
        geometry_profiles={"test": GeometryResponseProfile("test", yield_scale=4)}))
    assert ScanEngine(calibration=cal, render_model_mode="formula_only").calibration == base
    restricted = ScanEngine(calibration=cal, render_model_mode="formula_casino").calibration
    assert restricted.compo.scale == 3 and not restricted.surrogate.geometry_profiles


def test_mock_and_incomplete_calibration_are_not_usable(tmp_path):
    output = tmp_path / "casino_output.json"
    output.write_text('{}')
    with pytest.raises(ValueError, match="four quadrants"):
        parse_casino_case_output(tmp_path)
    output.write_text(json.dumps({"quadrants": dict.fromkeys('ABCD', 1), "meta": {"backend": "mock"}}))
    parsed = parse_casino_case_output(tmp_path)
    assert parsed["provenance"]["backend"] == "mock" and not parsed["calibration_eligible"]
    output.write_text(json.dumps({"quadrants": dict.fromkeys('ABCD', float('nan'))}))
    with pytest.raises(ValueError, match="finite"):
        parse_casino_case_output(tmp_path)
    with pytest.raises(ValueError):
        _fit_linear(np.array([1]), np.array([2]))


def test_failed_live_calibration_does_not_write_identity(monkeypatch, tmp_path):
    import bse_renderer.validation.casino_calibration as module
    monkeypatch.setattr(module, "run_casino_probe", lambda **kwargs: {"available": True})
    monkeypatch.setattr(module, "run_casino_case", lambda **kwargs: {"status": "diagnostic", "calibration_eligible": False})
    result = run_casino_calibration(config_dir="config", run_root=tmp_path, num_cases=1)
    assert result["status"] == "failed" and result["calibration_file"] is None
    assert not (tmp_path / "calibration/renderer_calibration.json").exists()


def test_legacy_fit_rejects_nonoutward_population():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="outward"):
        _sample_emission_components([SimpleNamespace(cos_phi=-.1)])


def test_renderer_requires_qualified_b_and_checks_full_domain(tmp_path, bind_qualification):
    path = tmp_path / "fit.json"
    raw = {"fit": {"global_pooled": {"fitted_b": .5}}}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="qualification"):
        ScanEngine(emission_b_mode="pooled_constant_fit", emission_b_path=str(path))
    raw["renderer_qualification"] = {"pooled_constant_fit": {
        "surface_selection_passed": True, "adequacy_passed": True, "repeat_holdout_passed": True,
        "tilt_holdout_passed": True, "beats_lambertian": True, "tilt_domain_deg": [10, 30], "source_sha256": "unit-fixture", "energy_domain_kev": [60, 60], "material_z": 26}}
    path.write_text(json.dumps(raw))
    bind_qualification(path, raw)
    fit = qualify_emission_b_fit(load_emission_b_fit(path, "pooled_constant_fit"))
    assert fit.evaluate(20).item() == .5
    with pytest.raises(ValueError, match="extrapolation"):
        fit.evaluate(40)
    raw["fit"]["global_pooled"]["fitted_b"] = 3.
    bind_qualification(path, raw)
    with pytest.raises(ValueError, match="positivity"):
        qualify_emission_b_fit(load_emission_b_fit(path, "pooled_constant_fit"))


def test_unseen_sequence_generation_rejects_overlap_before_writing(tmp_path):
    with pytest.raises(ValueError, match="stride"):
        generate_topo_sequence_dataset(config_dir="config", output_dir=tmp_path/'out', frame_size=64, stride_px=16)
    assert not (tmp_path/'out').exists()


def test_batch_guard_and_complete_source_archive(tmp_path):
    bundle = load_config_bundle('config')
    with pytest.raises(ValueError, match="workers"):
        BatchGenerator(bundle, BatchOptions(1, tmp_path, workers=2))
    engine = ScanEngine()
    provenance = _freeze_generation_provenance(tmp_path, config_dir='config', engine=engine,
        seed=1, pixel_size_mm=.05, generation_parameters={})
    assert 'DataSimulation/bse_renderer/geometry/voxel_volume.py' in provenance['source_hashes']
    assert (tmp_path / provenance['source_archive']).is_file()
    assert provenance['configuration_snapshot'] and provenance['dependency_versions']['numpy']
    with pytest.raises(FileExistsError):
        BatchGenerator(bundle, BatchOptions(1, tmp_path)).run()
