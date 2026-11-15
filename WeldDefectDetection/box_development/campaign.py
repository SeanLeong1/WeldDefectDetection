from pathlib import Path
import os
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
from box_comparison.campaign import REPO, DEV, OLD, ARMS, read, inventory, verify_files
from box_comparison.core import write, sha, subset, summarize, selection_key, boxes_from_mask, match_boxes
from box_refinement.campaign import DEFAULT as PREVIOUS, PREVIOUS as FIRST, source_files as prior_sources, require, receipt
from src.data.sequence_audit import audit_sequence_dataset
from .data import load, training_sets, shuffle_context
from .methods import fit_scale, simple_scores, select_scores, evaluate_spec
from .training import fit

DEFAULT = REPO/'DataSimulation/data/output/box_context_development_20260921_v3'
SEEDS = [921710001, 921710002, 921710003]


def sources():
    result = prior_sources()
    files = list((REPO/'WeldDefectDetection/box_development').glob('*.py'))
    files += [REPO/'WeldDefectDetection/develop_boxes.py', REPO/'WeldDefectDetection/infer_developed_boxes.py',
              REPO/'WeldDefectDetection/tests/test_box_development.py']
    result.update({str(p.resolve()):sha(p) for p in files})
    return result


def prior_datasets():
    return [DEV, OLD/'datasets/sequence_confirmation', FIRST/'confirmation', PREVIOUS/'confirmation']


def check(root):
    require(root,'prepare'); p = read(root/'protocol.json')
    verify_files(p['source_hashes']); verify_files(p['preserved_files'])
    return p


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists(): raise ValueError('Partial preparation exists; inspect before retrying.')
    for old in (FIRST, PREVIOUS): verify_files(read(old/'FINAL_MANIFEST.json')['files'])
    preserved = {str(p):sha(p) for p in (FIRST/'FINAL_MANIFEST.json', PREVIOUS/'FINAL_MANIFEST.json',
        FIRST/'frozen_models.json', PREVIOUS/'frozen_models.json', DEV/'topo_normalization.json', OLD/'model/assumed_constant.json')}
    protocol = dict(schema='topo-box-development/v3',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        objective='Execute error/visibility audit, local-variation methods, nested data-size experiment, box-supervision experiment, then independent confirmation.',
        domain=dict(energy_kev=60,pixel_size_mm=.05,frame_size=64,frames=9,radius=2,stride=64,layout='straight_seam',B='assumed_constant'),
        seeds=dict(training=SEEDS,expansion=[300100000,400100000,500100000],confirmation=700100000),
        expansion=dict(batches=3,sequences_per_class_per_batch=18,parallel_processes=2,
            assignment='Every sequence in the three new datasets is designated training before generation. Their generator-local split labels are retained as provenance only. Original val/test remain held out.',
            populations='Nested 126/252/504 training sequences; 630/1260/2520 targets. Shared original validation: 42 sequences, 210 targets.'),
        training=dict(epochs=20,seeds=SEEDS,validation_every=5,threads=4,batch_size=32,
            loss='Weighted BCE plus Dice; optional rectangle-output loss weight 0.25.',
            optimizer='AdamW lr .001, weight_decay .0001, cosine to .00005',
            sampling='Expected 50% positive targets, replacement; epochs mean passes at each dataset size, so larger sets also receive more optimizer steps.',
            augmentation='Shared horizontal flip and common 0.9..1.1 gain / -0.02..0.02 offset. Fixed noise-relative fourth channel is dimensionless.',
            experiments='Three seeds for CNN and original 3-channel hybrid at 1x/2x/4x. Three seeds for 4-channel scaled-context hybrid at 4x.',
            rectangle_gate='If any 4x mask-only model in an arm has at least 3 validation truth regions with best box IoU in [0.1,0.5), train a two-output mask/rectangle network at 4x for its three seeds. CNN uses target, hybrid uses original 3-channel context; architecture otherwise unchanged.'),
        simple=dict(background_sigma=[4.,8.,12.],median_alignment='Smooth row-profile correction of each neighbour, median excludes target',
            scale='Spatial residual MAD fitted only to clean 4x training windows with clean context, smoothed sigma2 and floored at .003',
            thresholds=[2.,3.,4.,6.,8.]),
        selection='Original validation micro box F1 at IoU>=0.5; tie actual segmentation-mask IoU then fewer false positives. Historical configurations from both prior campaigns remain candidates.',
        truth='Every 8-connected ground-truth mask region, unchanged; rectangles for auxiliary training are derived from those same regions. No truth filtering.',
        postprocessing=dict(neural_thresholds=[.35,.5,.65,.8],min_area=[3,15],closing=[1,3],low_ratio=1.,padding=0),
        confirmation=dict(sequences_per_class=10,targets=350,freeze_before_generation=True,
            comparisons='Selected four methods, both prior versions, shuffled context, all-background and fixed central band rows20:44.'),
        statistics='2000 paired bootstrap resamples by sequence; exploratory multiple comparisons without multiplicity adjustment.',
        limitations=['Synthetic straight-weld domain only; constant B unqualified.','Repeated development validation is not independent evidence.',
            'Training size comparison holds epochs fixed, not compute. Three seed results are reported separately.',
            'No tuning after confirmation. No guarantee that further development improves performance.'],
        source_hashes=sources(),preserved_files=preserved)
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source_hashes']: archive.write(name,str(Path(name).relative_to(REPO)))
    receipt(root,'prepare',[root/'protocol.json',root/'source_snapshot.zip'])


def historical():
    return {label:{a:dict(arm=a,implementation='historical',spec={**s,'implementation':s.get('implementation','previous')},metrics=s['metrics'],origin=label)
                   for a,s in read(folder/'frozen_models.json')['models'].items()}
            for label,folder in (('previous',PREVIOUS),('original',FIRST))}


def audit(root):
    check(root)
    from .audit import diagnose, replay_visibility
    from box_comparison.audit import visibility_audit
    report = audit_sequence_dataset(DEV)
    if report['status']!='ok': raise ValueError('Development contract failed.')
    write(root/'development_audit.json',report)
    data=load(DEV,'original'); val=subset(data,'val')
    diagnose(val,historical()['previous'],root/'error_audit.json')
    visibility_audit(data,root/'contrast_audit.json')
    replay_visibility(DEV,root/'visibility',REPO/'DataSimulation/config',OLD/'model/assumed_constant.json')
    write(root/'development_inventory.json',inventory(DEV))
    receipt(root,'audit',[root/'development_audit.json',root/'error_audit.json',root/'contrast_audit.json',
        root/'development_inventory.json',*(root/'visibility').glob('*')],[root/'development_inventory.json'])


def generate_dataset(root,name,seed,count,comparisons):
    from bse_renderer.domain_randomisation.dataset_forms import generate_topo_sequence_dataset
    output = root/'datasets'/name
    planned={seed+10000*k+j+1000000*attempt for k in range(7) for j in range(count) for attempt in range(25)}
    for other in comparisons:
        if planned & {r['seed'] for r in read(other/'sequence_manifest.json')}:
            raise ValueError('Generation seed schedule overlaps a previous dataset.')
    summary=generate_topo_sequence_dataset(config_dir=REPO/'DataSimulation/config',output_dir=output,
        sequences_per_class=count,frames_per_sequence=9,frame_size=64,stride_px=64,context_radius=2,pixel_size_mm=.05,
        seed=seed,render_model_mode='formula_only',emission_b_mode='assumed_constant',
        emission_b_path=str(OLD/'model/assumed_constant.json'),normalization_path=DEV/'topo_normalization.json',
        weld_layout_mode='straight_seam',noise_profile='development',fixed_energy_kev=60)
    write(root/f'{name}_generation.json',summary)
    report=audit_sequence_dataset(output,compare_roots=comparisons)
    write(root/f'{name}_audit.json',report)
    if report['status']!='ok': raise ValueError(f'{name} dataset audit failed: {report["errors"]}')
    write(root/f'{name}_inventory.json',inventory(output))
    receipt(root,name,[root/f'{name}_generation.json',root/f'{name}_audit.json',root/f'{name}_inventory.json'],[root/f'{name}_inventory.json'])


def batch(root,index):
    p=check(root); require(root,'audit')
    generate_dataset(root,f'train_{index}',p['seeds']['expansion'][index-1],18,prior_datasets())


def expand(root):
    check(root);require(root,'audit')
    pending=[i for i in range(1,4) if not (root/'receipts'/f'train_{i}.json').exists()]
    active=[];started=time.monotonic();last=started
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while pending or active:
        while pending and len(active)<2:
            i=pending.pop(0); log=(root/f'train_{i}_worker.log').open('w',encoding='utf-8')
            process=subprocess.Popen([sys.executable,str(REPO/'WeldDefectDetection/develop_boxes.py'),
                '--root',str(root),'--stage',f'train_{i}'],stdout=log,stderr=subprocess.STDOUT,env=env)
            active.append((i,process,log));print(f'Started training-data batch {i}',flush=True)
        for item in list(active):
            i,process,log=item
            if process.poll() is not None:
                log.close();active.remove(item)
                if process.returncode:
                    # Leave other workers to finish their immutable independent datasets.
                    for _,other,handle in active: other.wait();handle.close()
                    raise RuntimeError(f'Batch {i} failed; inspect train_{i}_worker.log')
                print(f'Completed training-data batch {i}',flush=True)
        if time.monotonic()-last>45:
            print(f'Training-data generation running: {[i for i,_,_ in active]}, elapsed {time.monotonic()-started:.0f}s',flush=True);last=time.monotonic()
        if active: time.sleep(1)
    for i in range(1,4):require(root,f'train_{i}')
    comparisons=prior_datasets(); reports=[]
    for i in range(1,4):
        folder=root/'datasets'/f'train_{i}'
        report=audit_sequence_dataset(folder,compare_roots=comparisons)
        if report['status']!='ok':raise ValueError('Expansion cross-batch audit failed.')
        reports.append(report);comparisons.append(folder)
    write(root/'expansion_audit.json',reports)
    sets=training_sets(root,DEV)
    write(root/'training_membership.json',{str(k):v['meta'] for k,v in sets.items()})
    write(root/'training_populations.json',{str(k):dict(targets=len(v['meta']),sequences=len({m['sequence_id'] for m in v['meta']}),
        positive_targets=int(v['mask'].reshape(len(v['mask']),-1).any(1).sum())) for k,v in sets.items()})
    receipt(root,'expand',[root/'expansion_audit.json',root/'training_membership.json',root/'training_populations.json',
        *[root/f'train_{i}_inventory.json' for i in range(1,4)]],[root/f'train_{i}_inventory.json' for i in range(1,4)])


def train(root):
    p=check(root);require(root,'audit');require(root,'expand')
    sets=training_sets(root,DEV);val=subset(load(DEV,'original'),'val'); folder=root/'training';folder.mkdir(exist_ok=True)
    old=historical(); candidates={a:[old['previous'][a],old['original'][a]] for a in ARMS}
    # Recompute historical metrics on this exact validation population.
    for arm in ARMS:
        for spec in candidates[arm]:spec['metrics']=summarize(evaluate_spec(val,spec))
    for arm in ('thresholding','median'):
        for sigma in ((4.,8.,12.) if arm=='thresholding' else (8.,)):
            scale,diagnostic=fit_scale(sets[4],arm,sigma)
            spec=dict(arm=arm,implementation='local_variation',background_sigma=sigma,scale=scale.tolist(),scale_fit=diagnostic)
            from box_refinement.methods import select
            chosen,trials=select(val,simple_scores(val,spec),(2.,3.,4.,6.,8.),min_areas=(3,15),closings=(1,3),low_ratios=(1.,),paddings=(0,))
            spec.update(chosen);candidates[arm].append(spec)
            write(folder/f'{arm}_{sigma}_selection.json',dict(selected=spec,trials=trials))
    scale,_=fit_scale(sets[4],'median');scale=scale.tolist()
    curves=[]
    for multiplier in (1,2,4):
        for arm in ('cnn','cnn_median'):
            for seed in SEEDS:
                result=fit(sets[multiplier],val,folder,arm,seed,multiplier,arm,epochs=20)
                candidates[arm].append(result);curves.append(result)
    for seed in SEEDS:
        candidates['cnn_median'].append(fit(sets[4],val,folder,'cnn_median',seed,4,'context_scaled',epochs=20,scale=scale))
    from .audit import diagnose
    gate={}
    for arm in ('cnn','cnn_median'):
        eligible=[s for s in candidates[arm] if s.get('multiplier')==4 and s.get('variant')==arm]
        counts=[]
        for spec in eligible:
            d=diagnose(val,{arm:spec},folder/f'extent_{arm}_{spec["seed"]}.json')
            counts.append(d[arm]['counts']['inaccurate_extent'])
        gate[arm]=dict(inaccurate_extent_counts=counts,run_auxiliary=max(counts)>=3)
        if gate[arm]['run_auxiliary']:
            for seed in SEEDS:
                candidates[arm].append(fit(sets[4],val,folder,arm,seed,4,arm,epochs=20,rectangle=True))
    write(root/'rectangle_gate.json',gate)
    write(root/'learning_curve.json',[{k:r[k] for k in ('arm','seed','multiplier','metrics','epoch','training_targets','positive_training_targets','training_sequences','elapsed_seconds')} for r in curves])
    selected={a:max(candidates[a],key=lambda c:selection_key(c['metrics'])) for a in ARMS}
    write(root/'model_candidates.json',candidates);write(root/'selected_models.json',selected)
    receipt(root,'train',[root/'rectangle_gate.json',root/'learning_curve.json',root/'model_candidates.json',root/'selected_models.json',*folder.glob('*.json'),*folder.glob('*.pt')])


def freeze(root):
    check(root);require(root,'train')
    models=read(root/'selected_models.json')
    write(root/'frozen_models.json',dict(models=models,primary=max(ARMS,key=lambda a:selection_key(models[a]['metrics'])),
        historical=historical(),normalization=dict(path=str(DEV/'topo_normalization.json'),sha256=sha(DEV/'topo_normalization.json')),
        protocol_sha256=sha(root/'protocol.json'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    receipt(root,'freeze',[root/'frozen_models.json'])


def confirm(root):
    p=check(root);require(root,'freeze');require(root,'expand')
    generate_dataset(root,'confirmation',p['seeds']['confirmation'],10,prior_datasets()+[root/'datasets'/f'train_{i}' for i in range(1,4)])
    receipt(root,'confirm',[root/'confirmation_audit.json',root/'confirmation_inventory.json'],[root/'confirmation_inventory.json'])


def baselines(data):
    from box_refinement.methods import evaluate
    scores=np.zeros_like(data['target']); p=dict(threshold=.5,min_area=1,closing=1,low_ratio=1.,padding=0)
    empty=evaluate(data,scores,p);scores[:,20:44,:]=1
    return {'all_background':empty,'fixed_weld_band':evaluate(data,scores,p)}


def bootstrap(rows,primary):
    ids=sorted({r['sequence_id'] for r in rows['cnn']})
    totals={}
    for arm,rr in rows.items():
        totals[arm]=np.array([[sum(r[k] for r in rr if r['sequence_id']==s) for k in ('tp','fp','fn','pixel_tp','pixel_fp','pixel_fn')]+
            [sum(r['target_positive'] and r['predicted_positive'] for r in rr if r['sequence_id']==s),
             sum(r['target_positive'] for r in rr if r['sequence_id']==s),
             sum(not r['target_positive'] and not r['predicted_positive'] for r in rr if r['sequence_id']==s),
             sum(not r['target_positive'] for r in rr if r['sequence_id']==s)] for s in ids])
    rng=np.random.default_rng(921770101);draws={a:[] for a in rows}
    for _ in range(2000):
        indices=rng.integers(len(ids),size=len(ids))
        for a,t in totals.items():
            tp,fp,fn,pt,pf,pn,it,ip,inn,ine=t[indices].sum(0)
            draws[a].append([2*tp/max(2*tp+fp+fn,1),pt/max(pt+pf+pn,1),.5*(it/max(ip,1)+inn/max(ine,1))])
    draws={a:np.array(v) for a,v in draws.items()};keys=('box_f1','pixel_iou','balanced_accuracy')
    intervals={a:{k:np.quantile(v[:,i],[.025,.975]).tolist() for i,k in enumerate(keys)} for a,v in draws.items()}
    pairs=[(a,label+'_'+a) for label in ('previous','original') for a in ARMS]
    pairs += [('cnn_median','cnn'),('median','thresholding'),('cnn_median','cnn_median_shuffled'),('median','median_shuffled'),(primary,'all_background'),(primary,'fixed_weld_band')]
    differences={a+'_minus_'+b:{k:np.quantile(draws[a][:,i]-draws[b][:,i],[.025,.975]).tolist() for i,k in enumerate(keys)} for a,b in pairs}
    return dict(draws=2000,cluster_unit='sequence',intervals=intervals,paired_difference_intervals=differences)


def evaluate(root):
    check(root);require(root,'freeze');require(root,'confirm');require(root,'train')
    frozen=read(root/'frozen_models.json');data=load(root/'datasets/confirmation','confirmation')
    rows={a:evaluate_spec(data,s) for a,s in frozen['models'].items()}
    for label,models in frozen['historical'].items():
        for a,s in models.items():rows[label+'_'+a]=evaluate_spec(data,s)
    shuffled=shuffle_context(data)
    for a in ('median','cnn_median'):rows[a+'_shuffled']=evaluate_spec(shuffled,frozen['models'][a])
    rows.update(baselines(data))
    for name,rr in rows.items():write(root/'predictions'/f'{name}.json',rr)
    uncertainty=bootstrap(rows,frozen['primary']);primary=frozen['primary']
    pixel_skill=all(uncertainty['paired_difference_intervals'][primary+'_minus_'+b]['pixel_iou'][0]>0 for b in ('all_background','fixed_weld_band'))
    write(root/'results.json',dict(primary=primary,metrics={a:summarize(rr) for a,rr in rows.items()},uncertainty=uncertainty,
        per_class={a:{c:summarize([r for r in rr if r['source_class']==c]) for c in sorted({r['source_class'] for r in rr})} for a,rr in rows.items()},
        synthetic_pixel_image_skill_gate=bool(pixel_skill and uncertainty['intervals'][primary]['balanced_accuracy'][0]>.5),
        skill_gate_note='Image/pixel gate does not establish satisfactory box localisation or physical SEM transfer.'))
    receipt(root,'evaluate',[root/'results.json',*(root/'predictions').glob('*.json')])


def verify(root):
    check(root)
    for stage in ('audit','expand','train','freeze','confirm','evaluate'):require(root,stage)
    for folder in (FIRST,PREVIOUS):verify_files(read(folder/'FINAL_MANIFEST.json')['files'])
    from .inference import smoke
    smoke(root)
    write(root/'verification.json',dict(status='ok',historical_artifacts_unchanged=True,label_free_inference_matches=True,
        source_and_dataset_hashes_verified=True,constant_B_physically_qualified=False))
    receipt(root,'verify',[root/'verification.json',*(root/'inference_smoke').glob('*')])


def report(root):
    from .report import build
    build(root)


STAGES=dict(prepare=prepare,audit=audit,expand=expand,train=train,freeze=freeze,confirm=confirm,evaluate=evaluate,verify=verify,report=report)


def run(root,stage):
    root=Path(root).resolve();torch.set_num_threads(1 if stage.startswith('train_') else 4)
    if stage in ('train_1','train_2','train_3'):
        if (root/'receipts'/f'{stage}.json').exists():check(root);require(root,stage);return
        return batch(root,int(stage[-1]))
    for name in (list(STAGES) if stage=='all' else [stage]):
        if (root/'receipts'/f'{name}.json').exists():
            check(root);require(root,name);print(f'Hash-verified completed stage: {name}',flush=True);continue
        started=time.monotonic();print(f'Starting {name}',flush=True);STAGES[name](root)
        print(f'Completed {name} in {time.monotonic()-started:.1f}s',flush=True)
