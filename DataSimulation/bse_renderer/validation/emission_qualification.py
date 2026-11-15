"""Evidence-led model comparison; incomplete evidence never issues a certificate."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import t, norm

from .casino_compact import (analyze_samples, fit_scalar_b, fit_sine_b, _histogram,
                            _rr_nll, _write_json, select_surface_exits)
from ..domain_randomisation.dataset_forms import _sha256_path
from ..renderer.angular_table import AngularTable

def mean_interval(values, alpha=.05):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return {'mean': float(values.mean()) if len(values) and np.isfinite(values).all() else None,
                'ci95': None, 'passed': False, 'n_groups': len(values)}
    mean = float(values.mean())
    half = float(t.ppf(1-alpha/2, len(values)-1)*values.std(ddof=1)/math.sqrt(len(values)))
    return {'mean': mean, 'ci95': [mean-half, mean+half], 'passed': bool(mean-half > 0), 'n_groups': len(values)}

def fit_table(samples):
    knots = np.unique(samples[:, 4])
    return AngularTable(knots, np.array([_histogram(samples[samples[:, 4] == k]) for k in knots]))

def table_nll(samples, model):
    # Nominal tilt defines the experimental knot; measured deviations are reported separately.
    p = model.density(samples[:, 1], samples[:, 2], samples[:, 4])
    return float(-np.log(np.maximum(p, np.finfo(float).tiny)).mean())

def _global_directions(mu, az, alpha):
    a = np.radians(alpha)
    v = np.sqrt(np.maximum(0, 1-mu*mu))
    return np.stack([mu*np.sin(a)+v*np.cos(az)*np.cos(a), v*np.sin(az),
                     mu*np.cos(a)-v*np.cos(az)*np.sin(a)], axis=-1)

def detector_acceptance(samples, scalar_b, sine, table, geometries):
    """Fold-wise empirical probabilities versus numerical angular integrals.

    Intervals use Bonferroni-adjusted repeat-level t intervals. Four source
    repeats are intentionally not treated as hundreds of thousands of replicates.
    """
    rows = []
    fitted_repeats = {}
    for repeat in np.unique(samples[:, 3]):
        training = samples[samples[:, 3] != repeat]
        bound=.999/max(float(np.sin(samples[:,0]).max()),1e-10)
        fitted_repeats[repeat] = (float(np.clip(fit_scalar_b(training)['b'] or 0.,-bound,bound)),
                                  fit_sine_b(training, float(samples[:, 0].max())), fit_table(training))
    mu, mw = np.polynomial.legendre.leggauss(80)
    mu, mw = (mu+1)/2, mw/2
    az = (np.arange(384)+.5)*2*np.pi/384-np.pi
    mg, ag = np.meshgrid(mu, az, indexing='ij')
    weights = np.broadcast_to(mw[:, None]*(2*np.pi/384), mg.shape)
    for tilt in np.unique(samples[:, 4]):
        subset = samples[samples[:, 4] == tilt]
        direction = _global_directions(subset[:, 1], subset[:, 2], tilt)
        grid = _global_directions(mg, ag, tilt)
        poles = np.degrees(np.arccos(np.clip(direction[:, 2], -1, 1)))
        gp = np.degrees(np.arccos(np.clip(grid[..., 2], -1, 1)))
        aa, ga = np.arctan2(direction[:, 1], direction[:, 0]), np.arctan2(grid[..., 1], grid[..., 0])
        for geometry in geometries:
            inner, outer = geometry['polar_deg']
            for rotation in (0., 45., 90., 135.):
                for name, center in geometry['quadrants_deg'].items():
                    center = np.radians(center+rotation)
                    selected = (poles >= inner) & (poles <= outer) & (np.abs((aa-center+np.pi)%(2*np.pi)-np.pi) <= np.pi/4)
                    aperture = (gp >= inner) & (gp <= outer) & (np.abs((ga-center+np.pi)%(2*np.pi)-np.pi) <= np.pi/4)
                    errors = {k: [] for k in ('constant', 'sine', 'angular_table')}
                    for repeat in np.unique(subset[:, 3]):
                        held = subset[:, 3] == repeat
                        cf, sf, tf = fitted_repeats[repeat]
                        baseline = mg/np.pi
                        factor = np.sin(np.radians(tilt))*np.sqrt(1-mg*mg)*np.cos(ag)
                        densities = {'constant': baseline*(1+cf*factor),
                                     'sine': baseline*(1+(sf['b0']+sf['b1']*np.sin(np.radians(tilt)))*factor),
                                     'angular_table': tf.density(mg, ag, tilt)}
                        observed = float(selected[held].mean())
                        for key, density in densities.items():
                            errors[key].append(float(np.sum(density*weights*aperture))-observed)
                    rows.append({'tilt': float(tilt), 'geometry': geometry['id'], 'rotation_deg': rotation,
                                 'quadrant': name, 'errors_by_repeat': errors})
    family = max(1, len(rows)*3)
    for row in rows:
        row['models'] = {}
        for key, errors in row['errors_by_repeat'].items():
            interval = mean_interval(errors, .05/family)
            ci = interval['ci95']
            interval['within_tolerance'] = ci is not None and ci[0] >= -.02 and ci[1] <= .02
            row['models'][key] = interval
    return {'interval_method': 'Bonferroni simultaneous repeat-level t intervals',
            'tolerance_absolute_probability': .02, 'orientation_grid_deg': [0,45,90,135],
            'integration_grid': [80,384], 'rows': rows,
            'passed': {key: bool(rows) and all(r['models'][key]['within_tolerance'] for r in rows)
                       for key in ('constant', 'sine', 'angular_table')}}

def run_qualification(source: Path, pilot: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if (output/'qualification_report.json').exists():
        raise FileExistsError('Use a new qualification directory; reports are immutable.')
    domain = json.loads((pilot/'summary.json').read_text())
    for case in domain['cases']:
        path=pilot/f"{case['id']}.json"
        if _sha256_path(path)!=case['record_sha256']:
            raise ValueError('Pilot record hash mismatch.')
        for name,digest in json.loads(path.read_text())['artifacts'].items():
            if _sha256_path(pilot/name)!=digest:
                raise ValueError('Pilot image artifact hash mismatch.')
    sources = json.loads((source/'source_manifest.json').read_text())
    # Verify boundaries and regenerate selected angles from them, avoiding raw collision files.
    selected = []
    source_evidence = []
    headers = []
    seen_sources=set()
    for row in sources:
        cache = Path(row['compact'])
        if row['source_sha256'] in seen_sources:
            raise ValueError('Duplicate CASINO source cannot count as an independent repeat.')
        seen_sources.add(row['source_sha256'])
        with np.load(cache, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['sha256'] != row['source_sha256'] or len(data['first']) != row['histories']:
                raise ValueError(f'Boundary cache provenance mismatch: {cache}')
            headers.append(meta['header'])
        tilt, repeat = row['run_id'].split('deg-r')
        part, _ = select_surface_exits(cache, float(tilt), int(repeat), 1.)
        selected.append(part)
        source_evidence.append({'path': str(cache.resolve()), 'sha256': _sha256_path(cache),
                                'source_sha256': row['source_sha256']})
    samples = np.concatenate(selected)
    with np.load(source/'selected_surface_exits.npz', allow_pickle=False) as saved:
        if not np.array_equal(samples, saved['samples']):
            raise ValueError('Selected angles do not reproduce from compact boundary caches.')
    # Include the next measured knot to bracket the maximum required renderer incidence.
    required_max = domain['incidence_domain_deg'][1]
    available = np.unique(samples[:, 4])
    upper = next((float(k) for k in available if k >= required_max), float(available[-1]))
    samples = samples[samples[:, 4] <= upper]
    report = analyze_samples(samples)
    table = fit_table(samples)
    _write_json(output/'angular_table_candidate.json', {'angular_table': table.to_dict()})
    _write_json(output/'candidate_models.json', {
        'status':'diagnostic_candidates_not_qualified',
        'model_comparison': {'constant_b': {'fitted_b':report['pooled_constant']['b']},
                             'b_alpha_sin': report['pooled_sine_candidate']},
        'angular_table':table.to_dict()})
    model_folds = report['model_holdouts']
    comparisons = {}
    for name, key in [('constant', 'constant_nll'), ('sine', 'sine_nll')]:
        comparisons[name] = {unit: mean_interval([r['lambertian_nll']-r[key] for r in model_folds if r['heldout_unit'] == unit]) for unit in ('repeat','tilt')}
    comparisons['sine_over_constant'] = {unit: mean_interval([r['constant_nll']-r['sine_nll'] for r in model_folds if r['heldout_unit'] == unit]) for unit in ('repeat','tilt')}
    hist_folds = []
    for unit, col in [('repeat', 3), ('tilt', 4)]:
        for held in np.unique(samples[:, col]):
            train, test = samples[samples[:, col] != held], samples[samples[:, col] == held]
            fitted = fit_table(train)
            if test[:, 4].min() < fitted.tilts_deg[0] or test[:, 4].max() > fitted.tilts_deg[-1]:
                hist_folds.append({'unit': unit, 'held': float(held), 'status': 'endpoint_no_extrapolation'})
                continue
            hist_folds.append({'unit': unit, 'held': float(held), 'status': 'scored',
                               'lambertian_nll': _rr_nll(test, 0), 'table_nll': table_nll(test, fitted)})
    comparisons['angular_table'] = {unit: mean_interval([r['lambertian_nll']-r['table_nll'] for r in hist_folds if r['unit'] == unit and r['status'] == 'scored']) for unit in ('repeat','tilt')}
    # All sampled aperture geometries are assessed, including rotated local-normal directions.
    geometries = []
    for row in domain['cases']:
        record = json.loads((pilot/f"{row['id']}.json").read_text())
        geometries.append({'id': row['id'], 'polar_deg': record['metadata']['operating_domain']['detector_polar_deg'],
                           'quadrants_deg': record['metadata']['detector']['quadrants_deg']})
    # Fit-per-repeat models are cached by the detector helper's wrapper below.
    acceptance = detector_acceptance(samples, report['pooled_constant'], report['pooled_sine_candidate'], table, geometries)
    _write_json(output/'detector_acceptance.json', acceptance)
    acceptance_diagnosis={key:{
        'comparisons':len(acceptance['rows']),
        'passed':sum(r['models'][key]['within_tolerance'] for r in acceptance['rows']),
        'mean_error_outside_tolerance':sum(abs(r['models'][key]['mean'])>.02 for r in acceptance['rows']),
        'interval_fully_outside_tolerance':sum(r['models'][key]['ci95'] is not None and
                                             (r['models'][key]['ci95'][0]>.02 or r['models'][key]['ci95'][1]<-.02) for r in acceptance['rows'])
    } for key in acceptance['passed']}
    failed = [r['nominal_tilt_deg'] for r in report['per_tilt'] if r['nominal_tilt_deg'] > 0 and r['screening_decision'] != 'PASS_SCREENING']
    material_matches = all(np.isclose(z, domain['material']['z']) for z in domain['material_z_domain'])
    reasons = []
    if failed:
        reasons.append(f'Scalar-B screening is not passing at required/bracketing tilts {failed}.')
    if required_max > upper:
        reasons.append('Renderer incidence extends beyond available CASINO angles.')
    if not material_matches:
        reasons.append('Local effective material modifiers extend beyond the simulated 304 alloy; no measured transfer is established.')
    if not any(acceptance['passed'].values()):
        reasons.append('No candidate passes simultaneous detector-probability error intervals over the sampled apertures and orientations.')
    reasons.append('Fresh independent CASINO validation after model selection is not present.')
    # Existing repeats can inform model selection, but cannot become the requested fresh validation batch.
    final = {'schema': 'emission-qualification/v1', 'status': 'blocked', 'qualified_model': None,
             'scope': '304SS at 60 keV; conditional same-entrance-surface angular distribution',
             'required_domain': domain, 'fitted_tilt_domain_deg': [0., upper],
             'selected_count': len(samples), 'source_evidence': source_evidence,
             'source_headers': headers, 'candidate_analysis': report,
             'paired_likelihood_intervals': comparisons, 'histogram_holdouts': hist_folds,
             'detector_acceptance_passed': acceptance['passed'],
             'detector_acceptance_diagnosis':acceptance_diagnosis,
             'scalar_screening_nonpassing_tilts': failed, 'material_coverage_passed': material_matches,
             'fresh_validation_passed': False, 'reasons': reasons,
             'uncertainty_scope': 'Fold-level t intervals are descriptive approximations because cross-validation training sets overlap. Fresh independent model-frozen validation is required for release.',
             'next_action': 'Resolve angular/material transfer evidence before production rendering; see targeted_runs.json.'}
    _write_json(output/'qualification_report.json', final)
    requested = sorted(set([float(k) for k in available if k <= upper] + ([math.ceil(required_max)] if required_max > upper else [])))
    _write_json(output/'targeted_runs.json', {
        'status': 'requires_external_CASINO_exports', 'energy_kev': 60., 'material': '304_SS',
        'tilts_deg': requested, 'initial_histories_per_chunk': 1000, 'max_export_bytes': 2*1024**3,
        'unseen_interpolation_validation_tilts_deg':[(a+b)/2 for a,b in zip(requested,requested[1:]) if (a+b)/2 <= required_max],
        'independent_repeats': 4, 'selection_seeds': [914710001,914710002,914710003,914710004],
        'reserved_validation_seeds': [914810001,914810002,914810003,914810004],
        'storage_policy': 'Retain verified compact entrance/final histories, input settings and hashes; never delete pre-existing raw files.',
        'geometry_requirement': 'Use a sufficiently wide flat entrance face, establish width/thickness convergence, and export every terminal class.',
        'material_requirement': 'Preserve full alloy fractions from source_headers; effective local Z changes are not automatically new alloy compositions.',
        'allocation_policy': 'Start with required/bracketing tilts failing adequacy or interpolation. Increase independent chunks only for unresolved uncertainty.'})
    return final
