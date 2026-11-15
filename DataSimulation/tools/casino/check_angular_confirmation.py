"""Verify and score a completed frozen CASINO confirmation batch without refitting."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from prepare_qualification_sims import PROJECT, sha
from run_qualification_batch import check_previous, input_paths, write_json
from evaluate_interpolation_campaign import boundary_fingerprints

sys.path.insert(0, str(PROJECT/'DataSimulation'))
from bse_renderer.renderer.emission_b import load_emission_b_fit, model_identity
from bse_renderer.validation.casino_compact import select_surface_exits, _rr_nll
from bse_renderer.validation.regularized_angular import table_nll
from bse_renderer.validation.emission_qualification import mean_interval


def verify_candidate(freeze_path):
    freeze = json.loads(freeze_path.read_text())
    candidate = Path(freeze['candidate_path'])
    if sha(candidate) != freeze['candidate_sha256']:
        raise ValueError('Frozen candidate hash changed.')
    raw = json.loads(candidate.read_text())
    for evidence in raw['evidence']+raw['code_evidence']:
        if sha(Path(evidence['path'])) != evidence['sha256']:
            raise ValueError(f'Candidate evidence hash changed: {evidence["path"]}')
    fit = load_emission_b_fit(candidate, 'angular_table')
    if model_identity(fit) != freeze['model_identity']:
        raise ValueError('Renderer model identity changed.')
    return freeze, raw, fit


def evaluate(root, freeze_path, output):
    root, output = root.resolve(), output.resolve()
    freeze, raw, fit = verify_candidate(freeze_path.resolve())
    if output.exists():
        raise FileExistsError('Use a new confirmation report directory.')
    if (root/'active_wave.lock').exists():
        raise ValueError('Wait for the active CASINO wave.')
    jobs = [j for j in json.loads((root/'manifest.json').read_text())['jobs'] if j['stage'] == 'confirmation']
    missing = []
    for job in jobs:
        paths = input_paths(root, job)
        if not check_previous(root, job, paths, False):
            missing.append(job['name'])
    if missing:
        raise ValueError(f'Confirmation incomplete: {len(missing)}/{len(jobs)} runs not started. No model scores read or report written.')
    identity = json.loads((root/'confirmation_identity.json').read_text())
    if identity['frozen_model_sha256'] != freeze['candidate_sha256']:
        raise ValueError('Confirmation was launched against a different candidate.')
    sources_path = next(Path(e['path']) for e in raw['evidence'] if Path(e['path']).name == 'source_manifest.json')
    seen = set()
    for source in json.loads(sources_path.read_text()):
        cache = Path(source['compact'])
        if sha(cache) != source['compact_sha256']:
            raise ValueError('Development boundary cache changed.')
        seen.update(boundary_fingerprints(cache))
    rows, sources = [], []
    constant = raw['model_comparison']['constant_b']['fitted_b']
    for job in jobs:
        folder = (root/job['sim_path']).parent
        receipt = json.loads((folder/'run_receipt.json').read_text())
        if receipt['frozen_model_sha256'] != freeze['candidate_sha256'] or receipt['started_utc'] <= freeze['frozen_utc']:
            raise ValueError('Run is not bound to the model freeze before validation.')
        cache = folder/'compact/boundaries.npz'
        compact = json.loads((folder/'compact/chunk_receipt.json').read_text())
        if sha(folder/(job['name']+'_summary.dat')) != compact['source_sha256']:
            raise ValueError('Confirmation DAT changed.')
        fingerprints = boundary_fingerprints(cache)
        if len(set(fingerprints)) != len(fingerprints) or seen.intersection(fingerprints):
            raise ValueError('Duplicate histories in development/confirmation.')
        seen.update(fingerprints)
        samples, blocks = select_surface_exits(cache, job['tilt_deg'], job['repeat'], 1.)
        if not len(samples) or any(b['status'] != 'eligible' for b in blocks):
            raise ValueError('Confirmation surface selection failed.')
        with np.load(folder/'compact/selected_exits.npz', allow_pickle=False) as saved:
            if not np.array_equal(samples, saved['samples']):
                raise ValueError('Confirmation selection changed.')
        nll = table_nll(samples, fit.angular_table)
        rows.append({'run_id': job['name'], 'tilt_deg': job['tilt_deg'], 'repeat': job['repeat'],
            'exits': len(samples), 'regularized_nll': nll,
            'gain_over_constant': _rr_nll(samples, constant)-nll,
            'gain_over_lambertian': _rr_nll(samples, 0)-nll,
            'polar_mu_squared': float(np.mean(samples[:, 1]**2))})
        sources.append({'run_id': job['name'], 'compact': str(cache), 'compact_sha256': sha(cache),
            'source_sha256': compact['source_sha256'], 'run_receipt_sha256': sha(folder/'run_receipt.json')})
    report = {'status': 'frozen_confirmation_likelihood_checked_qualification_incomplete',
        'model_refitted': False, 'candidate_sha256': freeze['candidate_sha256'],
        'run_count': len(rows), 'run_scores': rows,
        'comparisons': {key: mean_interval([r[key] for r in rows]) for key in ('gain_over_constant', 'gain_over_lambertian')},
        'per_tilt': [{'tilt_deg': tilt, 'comparisons': {key: mean_interval([r[key] for r in rows if r['tilt_deg'] == tilt])
                      for key in ('gain_over_constant', 'gain_over_lambertian')}} for tilt in sorted({r['tilt_deg'] for r in rows})],
        'production_qualified': False,
        'remaining_gates': ['Full detector-acceptance and angular-adequacy qualification.',
                            'Material coverage, geometry convergence and RNG independence evidence.'],
        'uncertainty_scope': 'Descriptive file-level t intervals. Numerical repeats do not establish independent RNG streams.'}
    output.mkdir(parents=True)
    write_json(output/'report.json', report)
    write_json(output/'source_manifest.json', sources)
    write_json(output/'provenance.json', {'analysis_code_sha256': sha(Path(__file__)),
        'freeze_sha256': sha(freeze_path), 'source_manifest_sha256': sha(output/'source_manifest.json')})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.verify_only:
            freeze, _, _ = verify_candidate(args.freeze)
            print(f'Frozen experimental candidate and evidence verified: {freeze["candidate_sha256"]}')
            raise SystemExit(0)
        if args.root is None or args.output is None:
            parser.error('--root and --output are required for confirmation analysis.')
        report = evaluate(args.root, args.freeze, args.output)
        print(json.dumps({'status': report['status'], 'comparisons': report['comparisons']}, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(2, str(error)+'\n')
