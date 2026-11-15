"""Evaluate the frozen candidate set on new tilts without refitting any model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from prepare_qualification_sims import PROJECT, sha
from run_qualification_batch import check_previous, input_paths, write_json

sys.path.insert(0, str(PROJECT/'DataSimulation'))
from bse_renderer.renderer.angular_table import AngularTable
from bse_renderer.validation.casino_compact import select_surface_exits, _rr_nll
from bse_renderer.validation.emission_qualification import mean_interval


def verify_frozen_inputs(freeze_path):
    freeze=json.loads(freeze_path.read_text())
    model_path=Path(freeze['candidate_models_path'])
    if sha(model_path)!=freeze['candidate_models_sha256']:
        raise ValueError('Frozen candidate hash changed; cannot call this held-out evaluation.')
    folder=freeze_path.parent
    for name,key in [('source_manifest.json','selection_source_manifest_sha256'),
                     ('selected_surface_exits.npz','selected_exits_sha256')]:
        if sha(folder/name)!=freeze[key]:
            raise ValueError('Frozen selection evidence changed.')
    return freeze,model_path,json.loads(model_path.read_text())


def boundary_fingerprints(cache):
    with np.load(cache,allow_pickle=False) as data:
        rows=np.column_stack([data['first'],data['final'],data['kind'],data['row_count']])
        if not np.isfinite(rows).all():
            raise ValueError('Nonfinite boundary data.')
        return [hashlib.sha256(row.tobytes()).hexdigest() for row in rows]


def evaluate(root, freeze_path, output):
    root,freeze_path,output=root.resolve(),freeze_path.resolve(),output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory.')
    if (root/'active_wave.lock').exists():
        raise ValueError('Wait for the active CASINO wave to finish.')
    freeze,model_path,models=verify_frozen_inputs(freeze_path)
    constant=models['model_comparison']['constant_b']['fitted_b']
    sine=models['model_comparison']['b_alpha_sin']
    table_data=models['angular_table']
    table=AngularTable(np.array(table_data['tilts_deg']),np.array(table_data['probability_mass']))
    seen=set()
    selection_sources=json.loads((freeze_path.parent/'source_manifest.json').read_text())
    for source in selection_sources:
        cache=Path(source['compact'])
        if sha(cache)!=source['compact_sha256']:
            raise ValueError('Selection boundary cache changed.')
        seen.update(boundary_fingerprints(cache))
    manifest=json.loads((root/'manifest.json').read_text())
    jobs=[j for j in manifest['jobs'] if j['stage']=='interpolation']
    if sorted(set(j['tilt_deg'] for j in jobs))!=freeze['interpolation_tilts_deg']:
        raise ValueError('Interpolation angles differ from frozen evaluation plan.')
    rows,samples_all,sources,blocks=[],[],[],[]
    for job in jobs:
        paths=input_paths(root,job)
        if not check_previous(root,job,paths,True):
            raise ValueError(f'Incomplete interpolation run: {job["name"]}')
        folder=paths['sim_path'].parent
        cache=folder/'compact/boundaries.npz'
        receipt=json.loads((folder/'compact/chunk_receipt.json').read_text())
        if sha(folder/(job['name']+'_summary.dat'))!=receipt['source_sha256']:
            raise ValueError(f'Source export changed: {job["name"]}')
        fingerprints=boundary_fingerprints(cache)
        if len(set(fingerprints))!=len(fingerprints) or seen.intersection(fingerprints):
            raise ValueError('Duplicate boundary histories across fitting/evaluation runs.')
        seen.update(fingerprints)
        samples,records=select_surface_exits(cache,job['tilt_deg'],job['repeat'],1.)
        if not len(samples) or any(r['status']!='eligible' for r in records):
            raise ValueError(f'Failed surface selection: {job["name"]}')
        # Reuse the frozen coefficients and table. Nominal tilt defines table
        # interpolation; tiny measured angle deviations remain in diagnostics.
        law=sine['b0']+sine['b1']*np.sin(samples[:,0])
        density=table.density(samples[:,1],samples[:,2],samples[:,4])
        if not np.all(density>0):
            raise ValueError('Frozen table has nonpositive held-out density.')
        scores={'lambertian':_rr_nll(samples,0), 'constant':_rr_nll(samples,constant),
                'sine':_rr_nll(samples,law),'table':float(-np.log(density).mean())}
        rows.append({'run_id':job['name'],'tilt_deg':job['tilt_deg'],'repeat':job['repeat'],
                     'histories':receipt['histories'],'exits':len(samples),
                     'polar_mean_mu_squared':float(np.mean(samples[:,1]**2)), 'nll':scores})
        samples_all.append(samples)
        blocks.extend(records)
        sources.append({'run_id':job['name'],'compact':str(cache),'compact_sha256':sha(cache),
                        'source_sha256':receipt['source_sha256'],
                        'run_receipt_sha256':sha(folder/'run_receipt.json')})
    comparisons={}
    for name,reference in [('constant','lambertian'),('sine','lambertian'),
                           ('table','lambertian'),('sine','constant'),('table','constant')]:
        improvements=[r['nll'][reference]-r['nll'][name] for r in rows]
        comparisons[f'{name}_over_{reference}']=mean_interval(improvements)
    per_tilt=[]
    for tilt in freeze['interpolation_tilts_deg']:
        subset=[r for r in rows if r['tilt_deg']==tilt]
        polar=mean_interval([r['polar_mean_mu_squared'] for r in subset])
        polar.pop('passed') # mean_interval's positivity flag is not an adequacy test.
        per_tilt.append({'tilt_deg':tilt,'histories':sum(r['histories'] for r in subset),
                         'exits':sum(r['exits'] for r in subset),'polar_mu_squared':polar,
                         'nll':{k:float(np.mean([r['nll'][k] for r in subset])) for k in rows[0]['nll']}})
    report={'status':'heldout_interpolation_evaluated_no_model_qualified','model_refitted':False,
            'run_count':len(rows),'incident_histories':sum(r['histories'] for r in rows),
            'selected_exits':sum(r['exits'] for r in rows),
            'frozen_candidate_sha256':freeze['candidate_models_sha256'],
            'comparisons':comparisons,'per_tilt':per_tilt,'run_scores':rows,
            'uncertainty_scope':'Descriptive t intervals over independent simulation-file score means, equal run weight. Numerical repeats do not establish RNG independence. Per-tilt polar intervals are not simultaneous.',
            'qualification_limits':['Fixed-model likelihood and polar diagnostics only; no full detector-acceptance release test here.',
                                    'Geometry equivalence and RNG provenance remain unqualified.',
                                    'Single alloy does not establish renderer effective-material coverage.',
                                    'Reserved final confirmation has not run.'],
            'simulations_started':0}
    output.mkdir(parents=True)
    write_json(output/'report.json',report)
    write_json(output/'source_manifest.json',sources)
    write_json(output/'surface_selection.json',blocks)
    np.savez_compressed(output/'selected_surface_exits.npz',samples=np.concatenate(samples_all))
    shutil.copy2(model_path,output/'frozen_candidates.json')
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    write_json(output/'analysis_provenance.json',{'evaluation_script_sha256':sha(Path(__file__)),
        'freeze_manifest_sha256':sha(freeze_path),'candidate_models_sha256':sha(model_path),
        'source_manifest_sha256':sha(output/'source_manifest.json'),
        'angular_table_code_sha256':sha(PROJECT/'DataSimulation/bse_renderer/renderer/angular_table.py'),
        'nll_code_sha256':sha(PROJECT/'DataSimulation/bse_renderer/validation/casino_compact.py')})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=evaluate(args.root,args.freeze,args.output)
    print(json.dumps(report,indent=2))
