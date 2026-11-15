"""Verify a prepared selection sweep and analyze its existing boundary caches.

No CASINO simulation is launched. Production qualification remains separate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from prepare_qualification_sims import PROJECT, sha
from run_qualification_batch import check_previous, input_paths, write_json

sys.path.insert(0, str(PROJECT/'DataSimulation'))
from bse_renderer.validation.casino_compact import analyze_samples, select_surface_exits
from bse_renderer.validation.emission_qualification import run_qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; analysis reports are preserved.')
    if (root/'active_wave.lock').exists():
        raise ValueError('Finish the active CASINO wave before reviewing the campaign.')
    manifest = json.loads((root/'manifest.json').read_text())
    jobs = [j for j in manifest['jobs'] if j['stage']=='selection']
    sources, arrays, blocks, fingerprints = [], [], [], set()
    # Existing loader verifies current SIM, CAS and compact records. It never
    # launches a process; ingest=True compacts any newly exported final wave.
    for job in jobs:
        paths = input_paths(root, job)
        if not check_previous(root, job, paths, True):
            raise ValueError(f'Incomplete selection run: {job["name"]}')
        folder=paths['sim_path'].parent
        cache=folder/'compact/boundaries.npz'
        with np.load(cache, allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if not np.isfinite(data['first']).all() or not np.isfinite(data['final']).all():
                raise ValueError(f'Nonfinite records in {cache}')
            if not np.allclose(data['first'][:,6],60,rtol=0,atol=.001):
                raise ValueError(f'Wrong beam energy in {cache}')
            # Ignore header differences when detecting duplicated histories.
            for row in np.column_stack([data['first'],data['final'],data['kind'],data['row_count']]):
                key=tuple(row)
                if key in fingerprints:
                    raise ValueError('Identical complete boundary history across the selection data.')
                fingerprints.add(key)
        if meta['sha256'] in [s['source_sha256'] for s in sources]:
            raise ValueError('Duplicate source file.')
        part, records=select_surface_exits(cache,job['tilt_deg'],job['repeat'],1.)
        if not len(part) or any(b['status']!='eligible' for b in records):
            raise ValueError(f'Ineligible surface selection in {job["name"]}: {records}')
        arrays.append(part)
        blocks.extend(records)
        sources.append({'run_id':job['name'],'source':meta['source'],
                        'source_bytes':meta['source_bytes'],'source_sha256':meta['sha256'],
                        'compact':str(cache),'compact_bytes':cache.stat().st_size,
                        'compact_sha256':sha(cache),'histories':meta['trajectory_count'],
                        'collision_rows':meta['collision_rows_read'],
                        'run_receipt_path':str(folder/'run_receipt.json'),
                        'run_receipt_sha256':sha(folder/'run_receipt.json')})
    output.mkdir(parents=True)
    samples=np.concatenate(arrays)
    np.savez_compressed(output/'selected_surface_exits.npz',samples=samples)
    write_json(output/'source_manifest.json',sources)
    write_json(output/'surface_selection.json',blocks)
    write_json(output/'analysis_provenance.json',{
        'review_script':str(Path(__file__).resolve()),'review_script_sha256':sha(Path(__file__)),
        'campaign_manifest_sha256':sha(root/'manifest.json'),
        'input_policy':'verified_selection_only_existing_boundary_caches',
        'geometry_and_confirmation_excluded':True,'simulations_started':0,
        'independent_rng_seed_control_established':False})
    print(f'Verified {len(jobs)} runs, {sum(s["histories"] for s in sources)} histories, {len(samples)} eligible exits.',flush=True)
    result=run_qualification(output,args.pilot.resolve(),output/'qualification')
    write_json(output/'surface_b_report.json',result['candidate_analysis'])
    print(json.dumps({'status':result['status'],'selected_count':result['selected_count'],
                      'nonpassing_tilts':result['scalar_screening_nonpassing_tilts'],
                      'likelihood':result['paired_likelihood_intervals'],
                      'detector':result['detector_acceptance_diagnosis']},indent=2))


if __name__=='__main__':
    main()
