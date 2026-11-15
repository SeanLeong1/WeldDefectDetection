"""Prospective pore-only training and independently seeded confirmation."""
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from box_comparison.campaign import REPO,OLD,DEV,ARMS,read,inventory,verify_files
from box_comparison.core import write,sha,subset,summarize,selection_key
from box_refinement.campaign import receipt,require,source_files
from box_refinement.methods import residual_score,select
from box_development.data import load,shuffle_context
from box_development.methods import fit_scale,simple_scores,evaluate_spec,make_network,predict
from box_development.training import fit
from box_development.campaign import baselines
from src.data.sequence_audit import audit_sequence_dataset
from .generation import generate_pore_sequences,DATASET_CLASSES

DEFAULT=REPO/'DataSimulation/data/output/pore_campaign_20260921_v1'
PREVIOUS=REPO/'DataSimulation/data/output/box_context_development_20260921_v3'
MODEL=OLD/'model/assumed_constant.json'
NORMALIZATION=DEV/'topo_normalization.json'
SEEDS=(921810001,921810002,921810003)


def check(root):
    require(root,'prepare')
    p=read(root/'protocol.json')
    verify_files(p['source_hashes']);verify_files(p['preserved_files'])
    return p


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise ValueError('Existing unreceipted preparation must be inspected.')
    verify_files(read(PREVIOUS/'FINAL_MANIFEST.json')['files'])
    paths=source_files()
    for directory in ('box_development','pore_campaign'):
        paths.update({str(p.resolve()):sha(p) for p in (REPO/'WeldDefectDetection'/directory).glob('*.py')})
    for name in ('run_pore_campaign.py','infer_pore_boxes.py','tests/test_pore_campaign.py'):
        p=REPO/'WeldDefectDetection'/name;paths[str(p.resolve())]=sha(p)
    preserved={str(p):sha(p) for p in (PREVIOUS/'FINAL_MANIFEST.json',PREVIOUS/'frozen_models.json',NORMALIZATION,MODEL)}
    p=dict(schema='surface-pore-box-campaign/v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        scope='Synthetic surface-expressed pores on straight welds; no claim for internal pores or physical SEM validity.',
        classes=list(DATASET_CLASSES),secondary_defects=False,
        domain=dict(energy_kev=60,pixel_size_mm=.05,frame_size=64,frames=9,radius=2,stride=64,material='304SS',layout='straight_seam',B='assumed_constant'),
        pore_ranges=dict(keyhole_pore=dict(diameter_mm=[.5,1.2],depth_parameter_mm=[.25,.6]),
                         intrinsic_pore=dict(diameter_mm=[.5,1.],depth_parameter_mm=[.25,.65])),
        renderer_policy='Existing coefficients, occupancy labels, noise, blur and material response unchanged. Depth parameter is not measured surface depression.',
        pilot=dict(seed=800100000,sequences_per_class=9,grid='Three endpoint/midpoint diameters by three endpoint/midpoint depths per pore type.',
                   gate='All masks reproduce; finite signals; each pore grid sequence changes surface height and noiseless TOPO. No minimum contrast or score filter.'),
        development=dict(seed=1100100000,sequences_per_class=100,split='60/20/20 grouped by sequence; untouched test reserved for later work'),
        confirmation=dict(seed=1000100000,sequences_per_class=30,freeze_before_generation=True,all_windows_evaluated=True,paired_noiseless=True),
        preprocessing=dict(path=str(NORMALIZATION),sha256=sha(NORMALIZATION),policy='Preserved original training-only mapping, no refit; equal preprocessing for old/new models.'),
        training=dict(seeds=list(SEEDS),epochs=20,validation_every=5,threads=4,batch_size=32,
                      recipe='Existing balanced sampling, shared flip/gain/offset, weighted BCE+Dice, AdamW and cosine schedule. Six new fits; no confirmation-driven retraining.'),
        threshold_selection='Validation micro box F1 at IoU>=0.5, ties pixel IoU then fewer FP. Existing residual and local-variation classical features; neural thresholds .35/.5/.65/.8, area3/15, closing1/3.',
        overfit_diagnostic='First eight positive and eight clean development-training targets, target-only CNN, 300 full-batch optimizer steps, no augmentation. Diagnostic only; no model candidate.',
        truth='All 8-connected geometric-mask regions, no filtering or altered labels; half-open xyxy boxes.',
        statistics='2000 paired sequence bootstrap draws; descriptive exploratory comparisons without multiplicity adjustment.',
        completion='Operational integrity separate from synthetic skill and practical box F1. Report F1 0.7 target without forcing it or tuning on confirmation.',
        source_hashes=paths,preserved_files=preserved)
    write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in paths:z.write(f,str(Path(f).relative_to(REPO)))
    receipt(root,'prepare',[root/'protocol.json',root/'source_snapshot.zip'])


def comparison_roots(root):
    candidates=[DEV,OLD/'datasets/sequence_confirmation',
        REPO/'DataSimulation/data/output/box_context_campaign_20260921_v1/confirmation',
        REPO/'DataSimulation/data/output/box_context_refinement_20260921_v2_run2/confirmation']
    candidates+=list((PREVIOUS/'datasets').iterdir())
    candidates+=list((root/'datasets').iterdir()) if (root/'datasets').exists() else []
    return [p for p in candidates if (p/'sequence_manifest.json').is_file()]


def generate(root,name,settings,pilot=False):
    target=root/'datasets'/name
    generated_stage='generated_'+name
    if (root/'receipts'/(generated_stage+'.json')).exists():
        require(root,generated_stage)
        print(f'Hash-verified generated dataset: {name}',flush=True)
        return
    others=[p for p in comparison_roots(root) if p!=target]
    planned={settings['seed']+10000*k+j+1000000*a for k in range(3) for j in range(settings['sequences_per_class']) for a in range(25)}
    for other in others:
        if planned & {r['seed'] for r in read(other/'sequence_manifest.json')}:
            raise ValueError('New seed schedule overlaps prior data.')
    generate_pore_sequences(config_dir=REPO/'DataSimulation/config',output_dir=target,
        sequences_per_class=settings['sequences_per_class'],seed=settings['seed'],frames_per_sequence=9,
        frame_size=64,stride_px=64,context_radius=2,pixel_size_mm=.05,render_model_mode='formula_only',
        emission_b_mode='assumed_constant',emission_b_path=str(MODEL),normalization_path=NORMALIZATION,
        weld_layout_mode='straight_seam',noise_profile='development',fixed_energy_kev=60,pilot_grid=pilot)
    audit=audit_sequence_dataset(target,compare_roots=others)
    write(root/(name+'_audit.json'),audit)
    if audit['status']!='ok':raise ValueError(f'Dataset audit failed: {audit["errors"]}')
    manifest=read(target/'sequence_manifest.json')
    for m in manifest:
        expected=[] if m['planned_class']=='clean' else [m['planned_class']]
        if m['planned_defect_types']!=expected:raise ValueError('Unexpected secondary or unsupported defect.')
    write(root/(name+'_inventory.json'),inventory(target))
    receipt(root,generated_stage,[root/(name+'_audit.json'),root/(name+'_inventory.json')],[root/(name+'_inventory.json')])


def pilot(root):
    p=check(root)
    generate(root,'pilot',p['pilot'],True)
    from .audit import replay,pilot_figures
    report=replay(root/'datasets/pilot',root/'pilot_pairs',REPO/'DataSimulation/config',MODEL,counterfactual=True)
    positive=report['rows']
    sequences={r['sequence_id'] for r in positive}
    gate=(len(sequences)==18 and all(r['max_height_change_mm']>0 and r['median_signal_change']>0 for r in positive))
    write(root/'pilot_gate.json',dict(passed=gate,scope='Geometric and noiseless response checks; no minimum noisy contrast gate.',
        all_cases_retained=True,rows=positive))
    pilot_figures(root/'datasets/pilot',root/'pilot_pairs',root)
    if not gate:raise ValueError('Pore surface response pilot failed; inspect before training.')
    receipt(root,'pilot',[root/'pilot_gate.json',root/'pilot_audit.json',root/'pilot_inventory.json',
        *root.glob('*_pilot.png'),*(root/'pilot_pairs').glob('*')],[root/'pilot_inventory.json'])


def development(root):
    p=check(root);require(root,'pilot')
    generate(root,'development',p['development'])
    data=load(root/'datasets/development','development')
    counts={s:dict(targets=len(subset(data,s)['meta']),positive_targets=int(subset(data,s)['mask'].reshape(len(subset(data,s)['meta']),-1).any(1).sum())) for s in ('train','val','test')}
    write(root/'development_population.json',counts)
    receipt(root,'development',[root/'development_population.json',root/'development_audit.json',root/'development_inventory.json'],[root/'development_inventory.json'])


def overfit(root,train):
    from src.training.train_cnn import WeightedBCEDiceLoss
    positive=train['mask'].reshape(len(train['mask']),-1).any(1)
    ids=np.r_[np.flatnonzero(positive)[:8],np.flatnonzero(~positive)[:8]]
    torch.manual_seed(921820001)
    x=torch.from_numpy(train['target'][ids,None]);y=torch.from_numpy(train['mask'][ids,None].astype(np.float32))
    model=make_network(1);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    loss_fn=WeightedBCEDiceLoss(10.,dice_weight=.5);history=[]
    for i in range(300):
        model.train();optimizer.zero_grad();loss=loss_fn(model(x),y)
        if not torch.isfinite(loss):raise ValueError('Nonfinite overfit diagnostic.')
        loss.backward();optimizer.step()
        if i in (0,49,99,199,299):history.append(dict(step=i+1,loss=float(loss.detach())))
    pred=predict(model,x.numpy())[:,0]>=.5;truth=y.numpy()[:,0]>0
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
    write(root/'overfit_diagnostic.json',dict(window_ids=[train['meta'][i]['window_id'] for i in ids],history=history,
        pixel_f1=2*tp/max(2*tp+fp+fn,1),pixel_tp=tp,pixel_fp=fp,pixel_fn=fn,
        note='Training-subset memorization diagnostic, not validation or confirmation performance.'))


def train(root):
    check(root);require(root,'development')
    data=load(root/'datasets/development','development');training=subset(data,'train');val=subset(data,'val')
    folder=root/'training';folder.mkdir(exist_ok=True);overfit(root,training)
    candidates={a:[] for a in ARMS}
    for arm in ('thresholding','median'):
        for sigma in (1.,2.):
            scores=np.stack([residual_score(t,r,arm,sigma=sigma,standardized=True) for t,r in zip(val['target'],val['reference'])])
            chosen,trials=select(val,scores,(1.5,2.5,4.,6.,8.),min_areas=(1,5,15),closings=(1,3),low_ratios=(.5,1.),paddings=(0,))
            spec=dict(arm=arm,implementation='historical',spec=dict(arm=arm,implementation='refined',feature=dict(sigma=sigma,standardized=True),**chosen),metrics=chosen['metrics'])
            candidates[arm].append(spec);write(folder/f'{arm}_row_{sigma}.json',dict(selected=spec,trials=trials))
        for sigma in ((4.,8.,12.) if arm=='thresholding' else (8.,)):
            scale,diagnostic=fit_scale(training,arm,sigma)
            spec=dict(arm=arm,implementation='local_variation',background_sigma=sigma,scale=scale.tolist(),scale_fit=diagnostic)
            chosen,trials=select(val,simple_scores(val,spec),(2.,3.,4.,6.,8.),min_areas=(3,15),closings=(1,3),low_ratios=(1.,),paddings=(0,))
            spec.update(chosen);candidates[arm].append(spec);write(folder/f'{arm}_local_{sigma}.json',dict(selected=spec,trials=trials))
    for arm in ('cnn','cnn_median'):
        for seed in SEEDS:candidates[arm].append(fit(training,val,folder,arm,seed,1,arm,epochs=20))
    selected={a:max(candidates[a],key=lambda s:selection_key(s['metrics'])) for a in ARMS}
    write(root/'model_candidates.json',candidates);write(root/'selected_models.json',selected)
    receipt(root,'train',[root/'overfit_diagnostic.json',root/'model_candidates.json',root/'selected_models.json',*folder.glob('*.json'),*folder.glob('*.pt')])


def freeze(root):
    check(root);require(root,'train')
    models=read(root/'selected_models.json')
    write(root/'frozen_models.json',dict(models=models,primary=max(ARMS,key=lambda a:selection_key(models[a]['metrics'])),
        normalization=dict(path=str(NORMALIZATION),sha256=sha(NORMALIZATION)),protocol_sha256=sha(root/'protocol.json'),
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    receipt(root,'freeze',[root/'frozen_models.json'])


def confirmation(root):
    p=check(root);require(root,'freeze')
    generate(root,'confirmation',p['confirmation'])
    from .audit import replay
    replay(root/'datasets/confirmation',root/'confirmation_pairs',REPO/'DataSimulation/config',MODEL)
    receipt(root,'confirmation',[root/'confirmation_audit.json',root/'confirmation_inventory.json',*(root/'confirmation_pairs').glob('*')],[root/'confirmation_inventory.json'])


def bootstrap(rows,primary,draw_count=2000):
    ids=sorted({r['sequence_id'] for r in rows[primary]})
    identity={(r['sequence_id'],r['window_id']) for r in rows[primary]}
    if not identity or any({(r['sequence_id'],r['window_id']) for r in rr}!=identity or len(rr)!=len(identity) for rr in rows.values()):
        raise ValueError('Paired comparisons require identical unique targets.')
    totals={a:np.array([[sum(r[k] for r in rr if r['sequence_id']==sid) for k in ('tp','fp','fn','pixel_tp','pixel_fp','pixel_fn')]+
        [sum(r['target_positive'] and r['predicted_positive'] for r in rr if r['sequence_id']==sid),sum(r['target_positive'] for r in rr if r['sequence_id']==sid),
         sum(not r['target_positive'] and not r['predicted_positive'] for r in rr if r['sequence_id']==sid),sum(not r['target_positive'] for r in rr if r['sequence_id']==sid)] for sid in ids]) for a,rr in rows.items()}
    rng=np.random.default_rng(921830001);draws={a:[] for a in rows}
    for _ in range(draw_count):
        ix=rng.integers(len(ids),size=len(ids))
        for a,t in totals.items():
            tp,fp,fn,pt,pf,pn,it,ip,inn,ine=t[ix].sum(0)
            draws[a].append([2*tp/max(2*tp+fp+fn,1),pt/max(pt+pf+pn,1),.5*(it/max(ip,1)+inn/max(ine,1))])
    draws={a:np.array(v) for a,v in draws.items()};keys=('box_f1','pixel_iou','balanced_accuracy')
    pairs=[('cnn_median','cnn'),('median','thresholding'),('median','median_shuffled'),('cnn_median','cnn_median_shuffled')]
    pairs += [(a,'previous_'+a) for a in ARMS]+[('quiet_'+a,a) for a in ARMS]+[(primary,'all_background'),(primary,'fixed_weld_band')]
    return dict(draws=draw_count,cluster_unit='sequence',intervals={a:{k:np.quantile(v[:,i],[.025,.975]).tolist() for i,k in enumerate(keys)} for a,v in draws.items()},
        paired_difference_intervals={a+'_minus_'+b:{k:np.quantile(draws[a][:,i]-draws[b][:,i],[.025,.975]).tolist() for i,k in enumerate(keys)} for a,b in pairs})


def evaluate(root):
    check(root);require(root,'confirmation');require(root,'freeze')
    from .audit import quiet_data
    frozen=read(root/'frozen_models.json');data=load(root/'datasets/confirmation','confirmation')
    quiet=quiet_data(data,root/'confirmation_pairs',read(NORMALIZATION))
    rows={a:evaluate_spec(data,s) for a,s in frozen['models'].items()}
    rows.update({'quiet_'+a:evaluate_spec(quiet,s) for a,s in frozen['models'].items()})
    rows.update({'previous_'+a:evaluate_spec(data,s) for a,s in read(PREVIOUS/'frozen_models.json')['models'].items()})
    shuffled=shuffle_context(data)
    for a in ('median','cnn_median'):rows[a+'_shuffled']=evaluate_spec(shuffled,frozen['models'][a])
    rows.update(baselines(data))
    for a,r in rows.items():write(root/'predictions'/(a+'.json'),r)
    primary=frozen['primary'];uncertainty=bootstrap(rows,primary)
    skill=uncertainty['intervals'][primary]['balanced_accuracy'][0]>.5 and all(uncertainty['paired_difference_intervals'][primary+'_minus_'+b]['pixel_iou'][0]>0 for b in ('all_background','fixed_weld_band'))
    write(root/'results.json',dict(primary=primary,metrics={a:summarize(rr) for a,rr in rows.items()},uncertainty=uncertainty,
        per_class={a:{c:summarize([r for r in rr if r['source_class']==c]) for c in DATASET_CLASSES} for a,rr in rows.items()},
        synthetic_pixel_image_skill_gate=bool(skill),physical_validation=False,
        note='Independent synthetic confirmation; one pore type per sequence. No confirmation-driven selection.'))
    receipt(root,'evaluate',[root/'results.json',*(root/'predictions').glob('*.json')])


def verify(root):
    check(root)
    for stage in ('pilot','development','train','freeze','confirmation','evaluate'):require(root,stage)
    verify_files(read(PREVIOUS/'FINAL_MANIFEST.json')['files'])
    from .inference import smoke
    smoke(root)
    write(root/'verification.json',dict(status='ok',previous_campaign_unchanged=True,all_four_label_free_inference_match=True,
        source_and_dataset_hashes_verified=True,B_physically_qualified=False,physical_validation=False))
    receipt(root,'verify',[root/'verification.json',*(root/'inference_smoke').glob('*')])


def report(root):
    check(root);require(root,'verify')
    from .report import build
    build(root)
    receipt(root,'report',[root/'REPORT.md',root/'comparison.png',root/'examples.png'])
    files=inventory(root);files.pop(str((root/'FINAL_MANIFEST.json').resolve()),None)
    write(root/'FINAL_MANIFEST.json',dict(files=files,scope='All campaign artifacts except this manifest. Operational success is not physical validation.'))


STAGES=dict(prepare=prepare,pilot=pilot,development=development,train=train,freeze=freeze,confirmation=confirmation,evaluate=evaluate,verify=verify,report=report)


def run(root,stage):
    root=Path(root).resolve();torch.set_num_threads(4)
    for name in (list(STAGES) if stage=='all' else [stage]):
        if (root/'receipts'/(name+'.json')).exists():
            check(root);require(root,name)
            if name=='report':verify_files(read(root/'FINAL_MANIFEST.json')['files'])
            print(f'Hash-verified completed stage: {name}',flush=True);continue
        print(f'Starting {name}',flush=True);start=time.monotonic();STAGES[name](root)
        print(f'Completed {name} in {time.monotonic()-start:.1f}s',flush=True)
