import json
from dataclasses import replace
import numpy as np
import pytest
from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.domain_randomisation.dataset_forms import _set_regime
from bse_renderer.renderer.angular_table import AngularTable
from bse_renderer.renderer.emission_b import load_emission_b_fit, qualify_emission_b_fit
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.validation.emission_qualification import mean_interval


def test_fixed_energy_survives_every_regime():
    sample = sample_parameters(load_config_bundle('config'), seed=1)
    for regime in ('low','mid','high'):
        actual = _set_regime(sample, regime, 60.)
        assert actual.beam.e0_kev == actual.beam.accelerating_voltage_kv == 60.
        assert actual.beam.power_w == pytest.approx(60*actual.beam.beam_current_ma)
    for bad in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            _set_regime(sample, 'low', bad)


def test_angular_table_normalization_interpolation_and_support():
    mass = np.ones((2,12,24))/288
    mass[1] *= np.linspace(.5,1.5,12)[:,None]
    table = AngularTable(np.array([0,60]), mass)
    mu, az = np.meshgrid((np.arange(12)+.5)/12, (np.arange(24)+.5)*2*np.pi/24-np.pi, indexing='ij')
    for alpha in (0, 10, 30, 60):
        density = table.density(mu,az,alpha)
        assert np.min(density) >= 0
        assert density.sum()*2*np.pi/288 == pytest.approx(1)
    assert np.allclose(table.density(mu,az,30), .5*(table.density(mu,az,0)+table.density(mu,az,60)))
    assert table.density(-.1,0,0) == 0
    assert table.density(.5,-np.pi,30) == table.density(.5,np.pi,30)
    with pytest.raises(ValueError, match='extrapolation'):
        table.density(.5,0,61)
    with pytest.raises(ValueError, match='mass one'):
        AngularTable(np.array([0,60]), mass*2)


def fixture_raw():
    return {'fit': {'global_pooled': {'fitted_b': .5}}, 'renderer_qualification': {'pooled_constant_fit': {
        'surface_selection_passed': True, 'adequacy_passed': True, 'repeat_holdout_passed': True,
        'tilt_holdout_passed': True, 'beats_lambertian': True, 'tilt_domain_deg': [0,60],
        'source_sha256': 'fixture', 'energy_domain_kev': [60,60], 'material_z': 25.4147}}}


def test_manual_flags_and_tampered_evidence_fail(tmp_path, bind_qualification):
    path = tmp_path/'fit.json'
    raw = fixture_raw()
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='hashed evidence'):
        qualify_emission_b_fit(load_emission_b_fit(path,'pooled_constant_fit'))
    bind_qualification(path,raw)
    qualify_emission_b_fit(load_emission_b_fit(path,'pooled_constant_fit'))
    (tmp_path/'fresh_validation.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        qualify_emission_b_fit(load_emission_b_fit(path,'pooled_constant_fit'))


def test_model_tampering_and_local_material_rejected(tmp_path, bind_qualification):
    path=tmp_path/'fit.json'
    raw=fixture_raw()
    bind_qualification(path,raw)
    engine=ScanEngine(image_size=32,render_view_mode='top_surface',emission_b_mode='pooled_constant_fit',emission_b_path=str(path))
    sample=_set_regime(sample_parameters(load_config_bundle('config'),seed=1,max_defects=0),'mid',60)
    with pytest.raises(ValueError,match='Local material'):
        engine.run(sample)
    raw['fit']['global_pooled']['fitted_b']=.6
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='bind model'):
        qualify_emission_b_fit(load_emission_b_fit(path,'pooled_constant_fit'))


def test_group_intervals_require_evidence():
    assert not mean_interval([.1])['passed']
    assert not mean_interval([-.1,.2,-.3,.1])['passed']
    assert mean_interval([.1,.11,.12,.11])['passed']


def test_stage_gate_stops_before_creating_dataset(tmp_path):
    from bse_renderer.validation.campaign_execution import execute_stage
    with pytest.raises(ValueError,match='Run domain-pilot'):
        execute_stage(tmp_path,'generate')
    assert not (tmp_path/'datasets').exists()


def test_resume_checks_artifact_hash(tmp_path):
    from bse_renderer.validation.campaign_execution import verify_receipt
    from bse_renderer.domain_randomisation.dataset_forms import _sha256_path
    path=tmp_path/'artifact.json'
    path.write_text('{"value":1}')
    receipt={'artifacts':{path.name:_sha256_path(path)}}
    verify_receipt(tmp_path,receipt)
    path.write_text('{"value":2}')
    with pytest.raises(ValueError,match='artifact changed'):
        verify_receipt(tmp_path,receipt)


def test_angular_density_integral_renderer_scaling():
    from bse_renderer.renderer.quadrant_detector import _integrated_quadrant_directional_term
    normal=np.array([[[0.,0.,1.]]])
    kwargs=dict(normals=normal,alpha=np.zeros((1,1)),tilt_azimuth=np.zeros((1,1)),center_rad=0.,phi_inner=0.,phi_outer=np.pi/2,
                b_coeff=0.,material_n=1.,directional_gain=1.,order_phi=24,order_varphi=24)
    expected=_integrated_quadrant_directional_term(**kwargs)
    actual=_integrated_quadrant_directional_term(**kwargs,angular_density=lambda mu,az,alpha:mu/np.pi)
    assert actual == pytest.approx(expected,abs=1e-10)


def test_chunk_history_cap_before_writing(tmp_path):
    from bse_renderer.validation.casino_chunk_ingest import ingest_chunk
    source=tmp_path/'summary.dat'
    source.write_text('"Trajectory 1"\n'*1001)
    with pytest.raises(ValueError,match='1000'):
        ingest_chunk(source,tmp_path/'out',tilt_deg=10,repeat=1,seed=100)
    assert not (tmp_path/'out').exists()


def test_release_requires_numeric_fresh_validation(tmp_path,bind_qualification):
    from bse_renderer.validation.qualification_release import export_qualified_model
    raw=fixture_raw()
    candidate=tmp_path/'candidate.json'
    bind_qualification(candidate,raw)
    with pytest.raises(ValueError,match='improve held-out likelihood'):
        export_qualified_model(candidate,tmp_path/'qualification.json',tmp_path/'fresh_validation.json',tmp_path/'out',mode='pooled_constant_fit')
    assert not (tmp_path/'out').exists()
