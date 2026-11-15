import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bse_renderer.renderer.emission_b import load_emission_b_fit, qualify_emission_b_fit
from bse_renderer.validation.regularized_angular import (
    RegularizedAngularFit, angular_grid, fit_regularized, table_nll,
)
from bse_renderer.validation.casino_compact import _rr_nll


def synthetic_runs(seed, count=4000):
    rng = np.random.default_rng(seed)
    truth = RegularizedAngularFit(np.array([-.8, .2, .2, -.1]), np.ones(4), 4, .001,
                                  np.array([0., 20., 40., 60.]), {})
    table = truth.table()
    runs = []
    for i, tilt in enumerate(table.tilts_deg):
        bins = rng.choice(288, size=count, p=table.probability_mass[i].ravel())
        mu = (bins//24+rng.random(count))/12
        az = (bins%24+rng.random(count))*2*np.pi/24-np.pi
        runs.append(np.column_stack([np.full(count, np.radians(tilt)), mu, az,
                                     np.ones(count), np.full(count, tilt)]))
    return runs, table


def test_fit_recovers_polar_change_on_new_samples():
    training, truth = synthetic_runs(14)
    check, _ = synthetic_runs(24)
    fit = fit_regularized(training, feature_count=4, ridge=.001)
    table = fit.table()
    assert np.mean([_rr_nll(x, 0)-table_nll(x, table) for x in check]) > .04
    assert np.max(np.sum(abs(table.probability_mass-truth.probability_mass), axis=(1, 2))) < .07


def test_normalization_symmetry_interpolation_and_support():
    runs, _ = synthetic_runs(10, 500)
    table = fit_regularized(runs, feature_count=8, ridge=.01).table()
    mu, az, _ = angular_grid()
    for tilt in np.linspace(0, 60, 127):
        p = table.density(mu, az, tilt)
        assert p.min() > 0
        assert p.sum()*2*np.pi/288 == pytest.approx(1., abs=1e-12)
        assert np.allclose(p, table.density(mu, -az, tilt))
    assert np.allclose(table.probability_mass[0], table.probability_mass[0, :, :1])
    with pytest.raises(ValueError, match='extrapolation'):
        table.density(.5, 0, 60.1)
    with pytest.raises(ValueError, match='extrapolation'):
        fit_regularized(runs, feature_count=4, ridge=.01).table([0, 70])


def test_strong_regularization_and_malformed_data():
    runs, _ = synthetic_runs(12, 500)
    table = fit_regularized(runs, feature_count=4, ridge=1e6).table()
    _, _, base = angular_grid()
    assert np.max(abs(table.probability_mass-base.reshape(12, 24))) < 1e-7
    for bad in ([], [np.empty((0, 5))], [np.zeros((10, 4))]):
        with pytest.raises(ValueError):
            fit_regularized(bad, feature_count=4, ridge=.001)


def test_candidate_load_preserves_metadata_and_stays_unqualified(tmp_path):
    runs, _ = synthetic_runs(3, 100)
    table = fit_regularized(runs, feature_count=4, ridge=.01).table()
    path = tmp_path/'candidate.json'
    path.write_text(json.dumps({'angular_table': table.to_dict()}))
    loaded = load_emission_b_fit(path, 'angular_table')
    assert loaded.angular_table.to_dict() == table.to_dict()
    assert loaded.angular_table.prior_count_per_bin is None
    with pytest.raises(ValueError):
        qualify_emission_b_fit(loaded)


def test_development_repeat_split_is_disjoint():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/casino'))
    from develop_angular_model import split_runs
    runs, _ = synthetic_runs(4, 10)
    for i, run in enumerate(runs):
        run[:, 3] = i+1
    groups = split_runs(runs)
    assert [len(groups[k]) for k in ('fit', 'selection', 'check')] == [2, 1, 1]
    assert groups['check'][0] is runs[3]
    runs[3][:, 3] = 11
    with pytest.raises(ValueError, match='1..4'):
        split_runs(runs)


def test_confirmation_rejects_changed_evidence(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/casino'))
    from check_angular_confirmation import verify_candidate
    from prepare_qualification_sims import sha
    from bse_renderer.renderer.emission_b import model_identity
    runs, _ = synthetic_runs(9, 30)
    table = fit_regularized(runs, feature_count=4, ridge=.01).table()
    evidence = tmp_path/'evidence.json'
    evidence.write_text('{}')
    candidate = tmp_path/'candidate.json'
    candidate.write_text(json.dumps({'angular_table': table.to_dict(), 'code_evidence': [],
        'evidence': [{'path': str(evidence), 'sha256': sha(evidence)}]}))
    freeze = tmp_path/'freeze.json'
    freeze.write_text(json.dumps({'candidate_path': str(candidate), 'candidate_sha256': sha(candidate),
        'model_identity': model_identity(load_emission_b_fit(candidate, 'angular_table'))}))
    verify_candidate(freeze)
    evidence.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='evidence hash changed'):
        verify_candidate(freeze)


def test_detector_diagnostic_serializes_and_integrates(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/casino'))
    from develop_angular_model import detector_check
    from prepare_qualification_sims import sha
    _, table = synthetic_runs(1, 10)
    record = tmp_path/'normal.json'
    record.write_text(json.dumps({'metadata': {'operating_domain': {'detector_polar_deg': [0, 90]},
        'detector': {'quadrants_deg': {'A': 0, 'B': 90, 'C': 180, 'D': 270}}}}))
    (tmp_path/'summary.json').write_text(json.dumps({'cases': [{'id': 'normal', 'record_sha256': sha(record)}]}))
    runs, _ = synthetic_runs(2, 100)
    report = detector_check([runs[0]], table, tmp_path)
    json.dumps(report, allow_nan=False)
    for row in report['rows']:
        assert row['predicted'] == pytest.approx(.25, abs=1e-12)
