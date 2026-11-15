import json
from dataclasses import replace

import numpy as np
import pytest

from assumed_campaign import initialize
from bse_renderer.config_loader import load_config_bundle
from bse_renderer.domain_randomisation.dataset_forms import _sha256_path, _set_regime
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.renderer.emission_b import load_emission_b_fit, qualify_emission_b_fit
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.validation.campaign_execution import require_model, execute_stage


def make_campaign(tmp_path):
    source = tmp_path/'original.json'
    source.write_text(json.dumps({'model_comparison': {'constant_b': {'fitted_b': .92}}}))
    freeze = tmp_path/'freeze.json'
    freeze.write_text(json.dumps({'candidate_models_path': str(source), 'candidate_models_sha256': _sha256_path(source)}))
    root = tmp_path/'synthetic'
    initialize(root, freeze)
    return root, root/'model/assumed_constant.json'


def test_explicit_assumption_never_becomes_qualification(tmp_path):
    root, path = make_campaign(tmp_path)
    mode, _ = require_model(root)
    assert mode == 'assumed_constant'
    fit = load_emission_b_fit(path, mode)
    assert np.allclose(fit.evaluate([0, 30, 60]), .92)
    assert fit.qualified_tilt_domain is None
    with pytest.raises(ValueError, match='engineering assumption'):
        qualify_emission_b_fit(fit)
    with pytest.raises(ValueError, match='assumed domain'):
        fit.evaluate(65)
    with pytest.raises(ValueError, match='Prerequisite stage'):
        execute_stage(root, 'generate')
    assert not (root/'datasets').exists()


def test_assumption_tampering_and_false_labels_rejected(tmp_path):
    root, path = make_campaign(tmp_path)
    raw = json.loads(path.read_text())
    raw['physical_qualification'] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='changed'):
        require_model(root)
    with pytest.raises(ValueError, match='explicitly unqualified'):
        load_emission_b_fit(path, 'assumed_constant')
    raw['physical_qualification'] = False
    raw['fitted_b'] = .8
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='differs'):
        load_emission_b_fit(path, 'assumed_constant')


def test_assumed_render_has_constant_and_provenance(tmp_path):
    _, path = make_campaign(tmp_path)
    engine = ScanEngine(image_size=32, render_model_mode='formula_only', render_view_mode='top_surface',
                        emission_b_mode='assumed_constant', emission_b_path=str(path))
    sample = _set_regime(sample_parameters(load_config_bundle('config'), seed=124, max_defects=0), 'low', 60.)
    result = engine.run(sample)
    assert np.isfinite(result.channels.topo).all()
    assert np.ptp(result.channels.topo) > 0
    assert result.metadata['emission_b']['b_stats']['mean'] == pytest.approx(.92)
    assert result.metadata['emission_b']['evidence_status'] == 'engineering_assumption'
    assert result.metadata['emission_b']['assumption']['physical_qualification'] is False
    assert not result.metadata['emission_b_diagnostic_only']
    with pytest.raises(ValueError, match='60 keV'):
        engine.run(replace(sample, beam=replace(sample.beam, e0_kev=70.)))


def test_ordinary_fit_does_not_get_assumption_permissions(tmp_path):
    _, path = make_campaign(tmp_path)
    with pytest.raises(ValueError):
        ScanEngine(emission_b_mode='pooled_constant_fit', emission_b_path=str(path))
    source = path.parent/'original_candidates.json'
    with pytest.raises(ValueError, match='qualification'):
        ScanEngine(emission_b_mode='pooled_constant_fit', emission_b_path=str(source))
    with pytest.raises(ValueError, match='explicitly unqualified'):
        ScanEngine(emission_b_mode='assumed_constant', emission_b_path=str(source))
