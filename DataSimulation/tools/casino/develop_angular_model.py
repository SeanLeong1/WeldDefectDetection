"""Run the recorded, bounded angular-model development cycle without launching CASINO."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

from prepare_qualification_sims import PROJECT, sha
from run_qualification_batch import check_previous, input_paths, write_json
from evaluate_interpolation_campaign import verify_frozen_inputs, boundary_fingerprints

sys.path.insert(0, str(PROJECT/'DataSimulation'))
from bse_renderer.renderer.angular_table import AngularTable
from bse_renderer.renderer.emission_b import load_emission_b_fit, model_identity
from bse_renderer.validation.casino_compact import select_surface_exits, fit_scalar_b, _rr_nll
from bse_renderer.validation.emission_qualification import mean_interval, _global_directions
from bse_renderer.validation.regularized_angular import fit_regularized, table_nll


def split_runs(runs):
    groups = {key: [] for key in ('fit', 'selection', 'check')}
    for x in runs:
        repeats = np.unique(x[:, 3])
        if len(repeats) != 1 or repeats[0] not in (1, 2, 3, 4):
            raise ValueError('Development requires one repeat ID from 1..4 per file.')
        groups['fit' if repeats[0] in (1, 2) else 'selection' if repeats[0] == 3 else 'check'].append(x)
    if any(not group for group in groups.values()):
        raise ValueError('Every development split must be present.')
    return groups


def select_recipe(fitting, selection):
    """The check data cannot enter this API or influence recipe selection."""
    candidates = []
    fits = {}
    for size in (4, 8):
        for ridge in (.0001, .001, .01):
            fit = fit_regularized(fitting, feature_count=size, ridge=ridge)
            score = float(np.mean([table_nll(x, fit.table()) for x in selection]))
            candidates.append({'feature_count': size, 'ridge': ridge, 'selection_nll': score})
            fits[(size, ridge)] = fit
    best = min(r['selection_nll'] for r in candidates)
    chosen = min((r for r in candidates if r['selection_nll'] <= best+.0001),
                 key=lambda r: (r['feature_count'], -r['ridge']))
    return chosen, candidates, fits[(chosen['feature_count'], chosen['ridge'])]


def score_runs(runs, table, constant, frozen_constant, old_table):
    rows = []
    edges = np.linspace(0, 1, 13)
    moment = (edges[1:]**3-edges[:-1]**3)/(3*np.diff(edges))
    for x in runs:
        tilt = float(x[0, 4])
        # Integrate the piecewise-constant density exactly, rather than squaring bin centres.
        mu, az = np.meshgrid((edges[1:]+edges[:-1])/2,
                            (np.arange(24)+.5)*2*np.pi/24-np.pi, indexing='ij')
        mass = table.density(mu, az, tilt)*(2*np.pi/288)
        rows.append({'tilt_deg': tilt, 'repeat': int(x[0, 3]), 'exits': len(x),
                     'observed_mu_squared': float(np.mean(x[:, 1]**2)),
                     'predicted_mu_squared': float(np.sum(mass*moment[:, None])),
                     'nll': {'regularized': table_nll(x, table),
                             'matched_constant': _rr_nll(x, constant),
                             'frozen_constant': _rr_nll(x, frozen_constant),
                             'lambertian': _rr_nll(x, 0), 'old_table': table_nll(x, old_table)}})
    comparisons = {key: mean_interval([r['nll'][key]-r['nll']['regularized'] for r in rows])
                   for key in ('matched_constant', 'frozen_constant', 'lambertian', 'old_table')}
    return {'run_scores': rows, 'regularized_gain_over': comparisons}


def detector_check(runs, table, pilot):
    """Selected-aperture development diagnostic, not a qualification release test.

    Bin-aligned midpoint integration preserves table normalization exactly.
    Wilson bounds assume independent histories; RNG independence is unproven.
    """
    domain = json.loads((pilot/'summary.json').read_text())
    geometries = []
    for row in domain['cases']:
        path = pilot/f"{row['id']}.json"
        if sha(path) != row['record_sha256']:
            raise ValueError('Pilot geometry evidence changed.')
        record = json.loads(path.read_text())['metadata']
        geometries.append({'id': row['id'], 'polar_deg': record['operating_domain']['detector_polar_deg'],
                           'quadrants_deg': record['detector']['quadrants_deg']})
    # Deterministic selection from geometry alone, never from angular residuals.
    chosen = [geometries[0], min(geometries, key=lambda g: g['polar_deg'][0]),
              max(geometries, key=lambda g: g['polar_deg'][1])]
    chosen = list({g['id']: g for g in chosen}.values())
    grids = {}
    for n in (8, 16):
        mu, az = np.meshgrid((np.arange(12*n)+.5)/(12*n),
                            (np.arange(24*n)+.5)*2*np.pi/(24*n)-np.pi, indexing='ij')
        grids[n] = (mu, az, 2*np.pi/mu.size)
    rows = []
    family = len(runs)*len(chosen)*4*4
    z = float(norm.ppf(1-.05/(2*family)))
    for x in runs:
        tilt = float(x[0, 4])
        d = _global_directions(x[:, 1], x[:, 2], tilt)
        poles, angles = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1))), np.arctan2(d[:, 1], d[:, 0])
        evaluated = {}
        for n, (mu, az, area) in grids.items():
            gd = _global_directions(mu, az, tilt)
            evaluated[n] = (np.degrees(np.arccos(np.clip(gd[..., 2], -1, 1))),
                            np.arctan2(gd[..., 1], gd[..., 0]), table.density(mu, az, tilt)*area)
        for geometry in chosen:
            inner, outer = geometry['polar_deg']
            for rotation in (0, 45, 90, 135):
                for name, center in geometry['quadrants_deg'].items():
                    center = np.radians(center+rotation)
                    observed = float(np.mean((poles >= inner)&(poles <= outer)&
                        (np.abs((angles-center+np.pi)%(2*np.pi)-np.pi) <= np.pi/4)))
                    predictions = {}
                    for n, (gp, ga, mass) in evaluated.items():
                        aperture = (gp >= inner)&(gp <= outer)&(np.abs((ga-center+np.pi)%(2*np.pi)-np.pi) <= np.pi/4)
                        predictions[n] = float(mass[aperture].sum())
                    count = len(x)
                    midpoint = (observed+z*z/(2*count))/(1+z*z/count)
                    half = z*np.sqrt(observed*(1-observed)/count+z*z/(4*count*count))/(1+z*z/count)
                    ci = [float(predictions[16]-midpoint-half), float(predictions[16]-midpoint+half)]
                    rows.append({'tilt_deg': tilt, 'geometry': geometry['id'], 'quadrant': name,
                        'rotation_deg': rotation, 'exits': count, 'observed': observed,
                        'predicted': predictions[16], 'error': predictions[16]-observed,
                        'conditional_simultaneous_ci95': ci,
                        'grid_refinement_difference': abs(predictions[16]-predictions[8]),
                        'within_tolerance': ci[0] >= -.02 and ci[1] <= .02})
    return {'scope': 'Selected pilot apertures on development repeat 4; no release decision',
            'interval_assumption': 'Bonferroni Wilson intervals conditional on independent histories and a fixed fitted model; RNG independence unproven. Grid differences are diagnostics, not rigorous error bounds.',
            'geometries': chosen, 'comparisons': len(rows),
            'within_tolerance_count': sum(r['within_tolerance'] for r in rows),
            'max_abs_probability_error': max(abs(r['error']) for r in rows),
            'max_grid_refinement_difference': max(r['grid_refinement_difference'] for r in rows),
            'rows': rows}


def develop(root, freeze_path, pilot, protocol, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output folder; development results are immutable.')
    if (root/'active_wave.lock').exists() or (root/'confirmation_identity.json').exists():
        raise ValueError('Active simulation or confirmation already frozen; inspect before development.')
    freeze, model_path, models = verify_frozen_inputs(freeze_path.resolve())
    manifest = json.loads((root/'manifest.json').read_text())
    jobs = [j for j in manifest['jobs'] if j['stage'] in ('selection', 'interpolation')]
    for job in manifest['jobs']:
        if job['stage'] == 'confirmation':
            paths = input_paths(root, job)
            if check_previous(root, job, paths, False):
                raise ValueError('Confirmation has already run; use a new development campaign.')
    output.mkdir(parents=True)
    shutil.copy2(protocol, output/'protocol.md')
    write_json(output/'started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(protocol), 'script_sha256': sha(Path(__file__))})
    runs, sources, seen = [], [], set()
    for job in jobs:
        paths = input_paths(root, job)
        if not check_previous(root, job, paths, False):
            raise ValueError(f'Incomplete development run: {job["name"]}')
        folder = paths['sim_path'].parent
        cache = folder/'compact/boundaries.npz'
        receipt = json.loads((folder/'compact/chunk_receipt.json').read_text())
        if sha(folder/(job['name']+'_summary.dat')) != receipt['source_sha256']:
            raise ValueError('DAT source hash mismatch.')
        fingerprints = boundary_fingerprints(cache)
        if len(set(fingerprints)) != len(fingerprints) or seen.intersection(fingerprints):
            raise ValueError('Duplicate boundary histories.')
        seen.update(fingerprints)
        samples, blocks = select_surface_exits(cache, job['tilt_deg'], job['repeat'], 1.)
        if not len(samples) or any(b['status'] != 'eligible' for b in blocks):
            raise ValueError('Surface-selection eligibility failed.')
        with np.load(folder/'compact/selected_exits.npz', allow_pickle=False) as saved:
            if not np.array_equal(samples, saved['samples']):
                raise ValueError('Current surface selection differs from the verified cache.')
        runs.append(samples)
        sources.append({'run_id': job['name'], 'stage': job['stage'], 'histories': receipt['histories'],
            'exits': len(samples), 'compact': str(cache), 'compact_sha256': sha(cache),
            'source_sha256': receipt['source_sha256'], 'run_receipt_sha256': sha(folder/'run_receipt.json'),
            'split': 'fit' if job['repeat'] in (1, 2) else 'selection' if job['repeat'] == 3 else 'check'})
    groups = split_runs(runs)
    write_json(output/'source_manifest.json', sources)
    print(f'Verified {len(runs)} development runs and {sum(map(len,runs))} surface exits.', flush=True)
    chosen, candidates, fit = select_recipe(groups['fit'], groups['selection'])
    write_json(output/'recipe_selection.json', {'chosen': chosen, 'candidates': candidates,
                                               'check_data_used_for_selection': False})
    print(f'Selected recipe on repeat 3: {chosen}', flush=True)
    constant = fit_scalar_b(np.concatenate(groups['fit']))['b']
    old_constant = models['model_comparison']['constant_b']['fitted_b']
    old = models['angular_table']
    old_table = AngularTable(old['tilts_deg'], old['probability_mass'])
    selection = score_runs(groups['selection'], fit.table(), constant, old_constant, old_table)
    check = score_runs(groups['check'], fit.table(), constant, old_constant, old_table)
    write_json(output/'development_check.json', check)
    supported = all(check['regularized_gain_over'][k]['passed'] for k in ('matched_constant', 'lambertian'))
    recipe = {'feature_count': chosen['feature_count'], 'ridge': chosen['ridge']}
    original = [x for x, j in zip(runs, jobs) if j['stage'] == 'selection']
    intermediate = [x for x, j in zip(runs, jobs) if j['stage'] == 'interpolation']
    tilt_fit = fit_regularized(original, **recipe)
    tilt_check = score_runs(intermediate, tilt_fit.table(), fit_scalar_b(np.concatenate(original))['b'], old_constant, old_table)
    write_json(output/'tilt_holdout_diagnostic.json', tilt_check)
    print('Evaluating selected detector apertures on repeat 4.', flush=True)
    detector = detector_check(groups['check'], fit.table(), pilot)
    write_json(output/'detector_development_check.json', detector)
    code = output/'source_snapshot'
    for source in [Path(__file__), PROJECT/'DataSimulation/bse_renderer/validation/regularized_angular.py',
                   PROJECT/'DataSimulation/bse_renderer/validation/casino_compact.py',
                   PROJECT/'DataSimulation/bse_renderer/renderer/angular_table.py',
                   PROJECT/'DataSimulation/bse_renderer/renderer/emission_b.py',
                   PROJECT/'DataSimulation/bse_renderer/validation/emission_qualification.py']:
        destination = code/source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    report = {'status': 'improved_development_candidate_unqualified' if supported else 'development_improvement_not_supported',
        'selected_recipe': chosen, 'fitting_model': fit.to_dict(), 'matched_constant_b': constant,
        'constant_fit_weighting': 'pooled exits, existing scalar fit; angular model uses equal file weights',
        'baseline_overlap_warning': 'The old frozen constant and half-count table used all four original selection repeats. Their scores at original knots are partly in-sample and must not be called held-out comparisons. Only matched_constant and lambertian enter the development advancement decision.',
        'development_runs': len(runs), 'incident_histories': sum(s['histories'] for s in sources),
        'surface_exits': sum(map(len, runs)), 'selection': selection, 'check': check,
        'tilt_holdout_diagnostic': tilt_check['regularized_gain_over'],
        'uncertainty_scope': 'Descriptive t intervals over file score differences; prior exploration makes all these data development. Repeat IDs do not prove independent RNG seeds.',
        'production_qualified': False, 'simulations_started': 0,
        'remaining_gates': ['Fresh frozen-model confirmation, including all required detector apertures and adequacy diagnostics.',
            'Single-alloy evidence does not cover local effective-material modifiers.',
            'Geometry convergence and independent RNG provenance remain unqualified.',
            'Current 500-history confirmation chunks may be insufficient for simultaneous +/-0.02 acceptance intervals.']}
    write_json(output/'report.json', report)
    # An experimental freeze enables new evidence collection. It is distinct
    # from the conditional all-development refit and from production release.
    experimental = {'status': 'experimental_candidate_requires_more_evidence',
        'angular_table': fit.table().to_dict(), 'angular_model_training': fit.to_dict(),
        'model_comparison': {'constant_b': {'fitted_b': constant}},
        'fitted_domain': {'incidence_deg': [0., 60.], 'energy_kev': 60., 'material': '304_SS'},
        'evidence': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
                     [output/'report.json', output/'source_manifest.json', output/'protocol.md',
                      output/'recipe_selection.json', output/'development_check.json',
                      output/'tilt_holdout_diagnostic.json', output/'detector_development_check.json']],
        'code_evidence': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in code.rglob('*.py')],
        'training_scope': 'Repeat IDs 1 and 2 only; no all-development refit. Recipe selected on repeat 3.'}
    experiment_path = output/'experimental_candidate_model.json'
    write_json(experiment_path, experimental)
    loaded = load_emission_b_fit(experiment_path, 'angular_table')
    if loaded.angular_table.to_dict() != fit.table().to_dict():
        raise ValueError('Renderer load changed the experimental model.')
    write_json(output/'experimental_freeze.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_path': str(experiment_path), 'candidate_sha256': sha(experiment_path),
        'model_identity': model_identity(loaded), 'production_qualified': False,
        'development_improvement_supported': supported, 'confirmation_runs_observed': 0})
    if supported:
        final = fit_regularized(runs, **recipe)
        artifact = {'status': 'frozen_development_candidate_not_qualified', 'angular_table': final.table().to_dict(),
            'angular_model_training': final.to_dict(),
            'fitted_domain': {'incidence_deg': [0., 60.], 'energy_kev': 60., 'material': '304_SS',
                              'scope': 'same-entrance-surface outgoing electrons'},
            'evidence': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
                         [output/'report.json', output/'source_manifest.json', output/'protocol.md',
                          output/'recipe_selection.json', output/'development_check.json',
                          output/'tilt_holdout_diagnostic.json', output/'detector_development_check.json']],
            'code_evidence': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in code.rglob('*.py')]}
        candidate = output/'candidate_model.json'
        write_json(candidate, artifact)
        loaded = load_emission_b_fit(candidate, 'angular_table')
        if not np.array_equal(loaded.angular_table.probability_mass, final.table().probability_mass):
            raise ValueError('Renderer load did not preserve probability mass.')
        write_json(output/'candidate_freeze.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
            'candidate_path': str(candidate), 'candidate_sha256': sha(candidate),
            'model_identity': model_identity(loaded), 'production_qualified': False,
            'confirmation_runs_observed': 0, 'protocol_sha256': sha(output/'protocol.md')})
        write_json(output/'refit_training_diagnostics.json',
            score_runs(runs, final.table(), constant, old_constant, old_table))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = develop(args.root, args.freeze, args.pilot, args.protocol, args.output)
    print(json.dumps({'status': result['status'], 'check': result['check']['regularized_gain_over']}, indent=2))
