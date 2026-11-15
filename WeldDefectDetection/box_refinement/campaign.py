from __future__ import annotations
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from box_comparison.campaign import (REPO,DEV,OLD,ARMS,read,inventory,verify_files,
    check_protocol as check_previous,require as require_previous,shuffled_references)
from box_comparison.core import (sha,write,load_windows,subset,summarize,selection_key,
    evaluate_scores as previous_evaluate,box_iou)
from box_comparison.training import scores_for as previous_scores
from src.data.sequence_audit import audit_sequence_dataset
from .methods import evaluate,select
from .training import fit_simple,fit_network,scores_for

PREVIOUS=REPO/'DataSimulation/data/output/box_context_campaign_20260921_v1'
DEFAULT=REPO/'DataSimulation/data/output/box_context_refinement_20260921_v2_run2'
SEEDS=(921610001,921610002,921610003)


def source_files():
    paths=[REPO/'WeldDefectDetection/refine_boxes.py',REPO/'WeldDefectDetection/infer_refined_boxes.py']
    for folder in ('WeldDefectDetection/box_refinement','WeldDefectDetection/box_comparison',
                   'WeldDefectDetection/src','DataSimulation/bse_renderer','DataSimulation/config'):
        paths += [p for p in (REPO/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')]
    return {str(p.resolve()):sha(p) for p in sorted(paths)}


def require(root,stage):
    p=root/'receipts'/f'{stage}.json'
    if not p.is_file():raise ValueError(f'Missing prerequisite: {stage}')
    verify_files(read(p)['files'])
    for name in read(p).get('inventories',[]):verify_files(read(name))


def receipt(root,stage,paths,inventories=()):
    write(root/'receipts'/f'{stage}.json',dict(files={str(Path(p).resolve()):sha(p) for p in paths},
        inventories=[str(Path(p).resolve()) for p in inventories],completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


def check(root):
    require(root,'prepare');p=read(root/'protocol.json');verify_files(p['source_hashes']);verify_files(p['preserved_files'])
    return p


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise ValueError('Existing partial preparation must be inspected.')
    check_previous(PREVIOUS);require_previous(PREVIOUS,'freeze')
    verify_files(read(PREVIOUS/'FINAL_MANIFEST.json')['files'])
    previous=read(PREVIOUS/'frozen_models.json')
    preserved={str(PREVIOUS/'frozen_models.json'):sha(PREVIOUS/'frozen_models.json'),
        str(PREVIOUS/'FINAL_MANIFEST.json'):sha(PREVIOUS/'FINAL_MANIFEST.json'),
        str(DEV/'topo_normalization.json'):sha(DEV/'topo_normalization.json')}
    for s in previous['models'].values():
        if 'checkpoint' in s:preserved[s['checkpoint']]=s['checkpoint_sha256']
    protocol=dict(schema='topo-box-refinement/v2',previous_root=str(PREVIOUS),development_root=str(DEV),
        authorization='User requested a further tuning/training campaign after viewing the first confirmation results.',
        hypotheses=['Continuous weld-band response obscures local anomalies: subtract a robust row profile.',
            'Sparse positive training crops and ten epochs undertrain localisation: balanced sampling and forty epochs.',
            'Detected cores and disconnected fragments undershoot labelled regions: validate hysteresis, closing and bounded box padding.'],
        selection='Development validation micro box F1 at unchanged IoU >= 0.5; ties pixel IoU then fewer false positives.',
        ground_truth='Unchanged 8-connected regions in the complete renderer masks; no filtering or new easier test definition.',
        training=dict(seeds=list(SEEDS),epochs=40,validation_every_epochs=5,threads=4,batch_size=32,
            optimizer='AdamW',initial_learning_rate=.001,final_learning_rate=.00005,weight_decay=.0001,
            scheduler='cosine',positive_sample_probability=.5,
            augmentation='Common horizontal flip, gain 0.9..1.1 and offset -0.02..0.02 applied to target/reference; mask only flipped.',
            architecture='Existing TinyDefectSegNet, one channel CNN or three channel CNN+median, same size as v1.'),
        simple_features=dict(row_median_subtraction=True,sigma=[1.,2.],standardized=[False,True],scale_floor=.003,
            raw_thresholds=[.005,.01,.02,.04,.08],standardized_thresholds=[1.5,2.5,4.,6.]),
        postprocessing=dict(cnn_thresholds=[.2,.4,.6,.8],min_areas=[1,5,15],closings=[1,3],low_ratios=[.5,1.],padding_px=[0,2,4],
            note='Padding changes output boxes only, never the scored pixel masks or truth.'),
        candidates='Previous frozen settings retained for every arm; previous CNN weights with new box extraction; three newly trained seeds per CNN.',
        budget='One development cycle. Freeze after listed candidates. No tuning after fresh confirmation.',
        confirmation=dict(seed=1921210001,sequences_per_class=10,frames=9,size=64,stride=64,radius=2,
            energy_kev=60,pixel_size_mm=.05,emission='assumed_constant',noise='development'),
        evaluation='Evaluate new and previous frozen methods on the same new confirmation cases; paired sequence bootstrap, 2000 draws. Correct-context and shuffled-reference controls.',
        limitations=['Synthetic straight-weld TOPO only','B remains assumed','Prior confirmation results motivated this campaign but its images are excluded from tuning',
            'Validation is reused for development; the new confirmation is the independent check','Exploratory multiple comparisons, no multiplicity adjustment'],
        source_hashes=source_files(),preserved_files=preserved)
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in protocol['source_hashes']:archive.write(path,str(Path(path).relative_to(REPO)))
    receipt(root,'prepare',[root/'protocol.json',root/'source_snapshot.zip'])


def audit(root):
    check(root)
    verify_files(read(PREVIOUS/'development_inventory.json'))
    report=audit_sequence_dataset(DEV);write(root/'development_audit.json',report)
    if report['status']!='ok':raise ValueError('Development audit failed.')
    data=load_windows(DEV);val=subset(data,'val');train=subset(data,'train');previous=read(PREVIOUS/'frozen_models.json')['models']
    diagnostic=dict(training_targets=len(train['meta']),positive_training_targets=int(train['mask'].reshape(len(train['mask']),-1).any(1).sum()),
        validation_targets=len(val['meta']),prior_models={})
    for arm in ARMS:
        rows=previous_evaluate(val,previous_scores(val,previous[arm]),previous[arm]['parameters'])
        best_ious=[max((box_iou(t,p) for p in r['boxes_xyxy']),default=0) for r in rows for t in r['truth_boxes']]
        diagnostic['prior_models'][arm]=dict(metrics=summarize(rows),labelled_regions=len(best_ious),
            regions_with_any_overlap=sum(x>0 for x in best_ious),regions_with_iou_ge_025=sum(x>=.25 for x in best_ious),
            regions_with_iou_ge_050=sum(x>=.5 for x in best_ious))
    write(root/'development_diagnosis.json',diagnostic)
    write(root/'development_inventory.json',read(PREVIOUS/'development_inventory.json'))
    receipt(root,'audit',[root/'development_audit.json',root/'development_diagnosis.json',root/'development_inventory.json'],[root/'development_inventory.json'])


def train(root):
    p=check(root);require(root,'audit');data=load_windows(DEV);training=subset(data,'train');val=subset(data,'val')
    previous=read(PREVIOUS/'frozen_models.json')['models'];folder=root/'training';folder.mkdir(exist_ok=True)
    specifications=fit_simple(val,folder,previous)
    for arm in ('cnn','cnn_median'):
        candidates=[{**previous[arm],'implementation':'previous','training_origin':'previous_frozen'}]
        retuned,trials=select(val,previous_scores(val,previous[arm]),(.2,.4,.6,.8))
        candidates.append({**previous[arm],**retuned,'implementation':'refined','training_origin':'previous_weights_new_boxes'})
        write(folder/f'{arm}_previous_weights_retuned.json',dict(selected=retuned,trials=trials))
        for seed in p['training']['seeds']:
            r=fit_network(training,val,folder,arm,seed,p['training']['epochs'])
            candidates.append(dict(arm=arm,implementation='refined',training_origin='balanced_augmented_40epochs',
                seed=seed,epoch=r['selected']['epoch'],parameters=r['selected']['parameters'],metrics=r['selected']['metrics'],
                checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256']))
        specifications[arm]=max(candidates,key=lambda c:selection_key(c['metrics']))
        write(folder/f'{arm}_candidate_comparison.json',dict(selected=specifications[arm],candidates=candidates))
    write(root/'selected_models.json',specifications)
    receipt(root,'train',[root/'selected_models.json',*folder.glob('*.pt'),*folder.glob('*.json')])


def freeze(root):
    check(root);require(root,'train');models=read(root/'selected_models.json')
    frozen=dict(models=models,primary=max(ARMS,key=lambda a:selection_key(models[a]['metrics'])),
        previous_models=read(PREVIOUS/'frozen_models.json')['models'],
        normalization=dict(path=str(DEV/'topo_normalization.json'),sha256=sha(DEV/'topo_normalization.json')),
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=sha(root/'protocol.json'))
    write(root/'frozen_models.json',frozen);receipt(root,'freeze',[root/'frozen_models.json'])


def generate(root):
    p=check(root);require(root,'freeze')
    from bse_renderer.domain_randomisation.dataset_forms import generate_topo_sequence_dataset
    c=p['confirmation'];comparison=[DEV,OLD/'datasets/sequence_confirmation',PREVIOUS/'confirmation']
    planned={c['seed']+10000*k+j for k in range(7) for j in range(c['sequences_per_class'])}
    for folder in comparison:
        actual={r['seed'] for r in read(folder/'sequence_manifest.json')}
        if planned & actual:raise ValueError('Planned confirmation seeds overlap a previous campaign.')
    summary=generate_topo_sequence_dataset(config_dir=REPO/'DataSimulation/config',output_dir=root/'confirmation',
        sequences_per_class=c['sequences_per_class'],frames_per_sequence=9,frame_size=64,stride_px=64,context_radius=2,
        pixel_size_mm=.05,seed=c['seed'],render_model_mode='formula_only',emission_b_mode='assumed_constant',
        emission_b_path=str(OLD/'model/assumed_constant.json'),normalization_path=DEV/'topo_normalization.json',
        weld_layout_mode='straight_seam',noise_profile='development',fixed_energy_kev=60)
    write(root/'confirmation_generation.json',summary)
    report=audit_sequence_dataset(root/'confirmation',compare_roots=comparison);write(root/'confirmation_audit.json',report)
    if report['status']!='ok':raise ValueError('Fresh confirmation audit failed.')
    write(root/'confirmation_inventory.json',inventory(root/'confirmation'))
    receipt(root,'generate',[root/'confirmation_generation.json',root/'confirmation_audit.json',root/'confirmation_inventory.json'],[root/'confirmation_inventory.json'])


def evaluate_spec(data,spec):
    scores=scores_for(data,spec)
    return previous_evaluate(data,scores,spec['parameters']) if spec['implementation']=='previous' else evaluate(data,scores,spec['parameters'])


def bootstrap(rows):
    ids=sorted({r['sequence_id'] for r in rows['cnn']});columns=('tp','fp','fn','pixel_tp','pixel_fp','pixel_fn')
    totals={a:np.array([[sum(r[k] for r in rr if r['sequence_id']==s) for k in columns] for s in ids]) for a,rr in rows.items()}
    rng=np.random.default_rng(921991101);draws={a:[] for a in rows}
    for _ in range(2000):
        chosen=rng.integers(len(ids),size=len(ids))
        for a,t in totals.items():
            tp,fp,fn,pt,pf,pn=t[chosen].sum(0)
            draws[a].append([2*tp/max(2*tp+fp+fn,1),pt/max(pt+pf+pn,1)])
    draws={a:np.array(v) for a,v in draws.items()}
    intervals={a:{metric:np.quantile(d[:,i],[.025,.975]).tolist() for i,metric in enumerate(('box_f1','pixel_iou'))} for a,d in draws.items()}
    pairs=[(a,'previous_'+a) for a in ARMS]+[('cnn_median','cnn'),('median','thresholding'),('cnn_median','cnn_median_shuffled'),('median','median_shuffled')]
    differences={f'{a}_minus_{b}':{metric:np.quantile(draws[a][:,i]-draws[b][:,i],[.025,.975]).tolist()
        for i,metric in enumerate(('box_f1','pixel_iou'))} for a,b in pairs}
    return dict(cluster_unit='sequence',draws=2000,intervals=intervals,paired_difference_intervals=differences)


def evaluate_campaign(root):
    check(root);require(root,'freeze');require(root,'generate');require(root,'audit')
    frozen=read(root/'frozen_models.json');data=load_windows(root/'confirmation')
    rows={a:evaluate_spec(data,frozen['models'][a]) for a in ARMS}
    for arm in ARMS:
        spec=frozen['previous_models'][arm]
        rows['previous_'+arm]=previous_evaluate(data,previous_scores(data,spec),spec['parameters'])
    shuffled=shuffled_references(data)
    for arm in ('median','cnn_median'):rows[arm+'_shuffled']=evaluate_spec(shuffled,frozen['models'][arm])
    for arm,rr in rows.items():write(root/'predictions'/f'{arm}.json',rr)
    write(root/'results.json',dict(primary=frozen['primary'],metrics={a:summarize(rr) for a,rr in rows.items()},
        per_class={a:{c:summarize([r for r in rr if r['source_class']==c]) for c in sorted({r['source_class'] for r in rr})} for a,rr in rows.items()},
        uncertainty=bootstrap(rows)))
    receipt(root,'evaluate',[root/'results.json',*(root/'predictions').glob('*.json')])


def verify(root):
    check(root)
    for stage in ('audit','train','freeze','generate','evaluate'):require(root,stage)
    f=read(root/'frozen_models.json');verify_files({f['normalization']['path']:f['normalization']['sha256']})
    for s in f['models'].values():
        if 'checkpoint' in s:verify_files({s['checkpoint']:s['checkpoint_sha256']})
    from .inference import smoke
    smoke(root)
    write(root/'verification.json',dict(status='ok',physical_qualification=False,frozen_artifacts_verified=True,
        independent_confirmation=True,ground_truth_and_iou_criterion_unchanged=True,label_free_inference_matches=True))


def report(root):
    from .report import build
    build(root)


STAGES=dict(prepare=prepare,audit=audit,train=train,freeze=freeze,generate=generate,evaluate=evaluate_campaign,verify=verify,report=report)


def run(root,stage):
    root=Path(root).resolve();torch.set_num_threads(4)
    for name in (list(STAGES) if stage=='all' else [stage]):
        if (root/'receipts'/f'{name}.json').exists():
            check(root);require(root,name);print(f'Hash-verified completed stage: {name}',flush=True);continue
        print(f'Starting {name}',flush=True);start=time.monotonic();STAGES[name](root)
        print(f'Completed {name} in {time.monotonic()-start:.1f}s',flush=True)
