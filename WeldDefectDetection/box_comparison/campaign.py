"""Versioned, hash-checked development and independent box comparison."""
from __future__ import annotations
import json
from pathlib import Path
import platform
import time
import zipfile
import numpy as np
import torch
from .core import sha, write, load_windows, subset, evaluate_scores, summarize, selection_key
from .training import fit_simple, fit_network, scores_for
from .audit import visibility_audit, counterfactual_audit
from src.data.sequence_audit import audit_sequence_dataset

REPO = Path(__file__).resolve().parents[2]
OLD = REPO/'DataSimulation/data/output/assumed_constant_campaign_20260915_v2'
DEV = OLD/'datasets/sequence_dev'
DEFAULT = REPO/'DataSimulation/data/output/box_context_campaign_20260921_v1'
ARMS = ('thresholding', 'cnn', 'median', 'cnn_median')
SEEDS = (921510001, 921510002, 921510003)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inventory(root):
    return {str(p.resolve()): sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def sources():
    paths = [REPO/'WeldDefectDetection/compare_boxes.py',REPO/'WeldDefectDetection/infer_boxes.py']
    for folder in ('WeldDefectDetection/box_comparison', 'WeldDefectDetection/src', 'DataSimulation/bse_renderer', 'DataSimulation/config'):
        paths.extend(p for p in (REPO/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json'))
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def verify_files(files):
    for name, expected in files.items():
        if not Path(name).is_file() or sha(name) != expected:
            raise ValueError(f'Integrity mismatch: {name}')


def receipt(root, stage, files):
    write(root/'receipts'/f'{stage}.json', dict(stage=stage, files={str(Path(p).resolve()): sha(p) for p in files}))


def require(root, stage):
    path = root/'receipts'/f'{stage}.json'
    if not path.exists():
        raise ValueError(f'Complete prerequisite stage {stage} first.')
    verify_files(read(path)['files'])


def check_protocol(root):
    require(root, 'prepare')
    protocol = read(root/'protocol.json')
    verify_files(protocol['source_hashes'])
    verify_files(protocol['preserved_historical_files'])
    return protocol


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        require(root,'prepare'); return
    protocol = dict(schema='topo-box-comparison/v1', development_root=str(DEV),
        approaches=list(ARMS), seeds=list(SEEDS), epochs=10, threads=4,
        confirmation_seed=2026092101, confirmation_sequences_per_class=10,
        selection='Validation box F1 at IoU >= 0.5, then pixel IoU, then fewer false positive boxes.',
        truth='Every 8-connected region of the full renderer mask; no truth-area filter. Region detection, not physical instance identification.',
        box_coordinates='Half-open xyxy, local crop pixels; global coordinates add saved crop offsets.',
        domain=dict(frame_size=64,pixel_size_mm=.05,energy_kev=60,layout='straight_seam',context_radius=2,stride_px=64),
        context='Four native-coordinate, non-overlapping neighbours; target excluded. No label-based context cleaning.',
        training='Training windows only; existing training-only normalization. Three seeds per CNN, Adam lr .001, weighted BCE/Dice, ten epochs, no augmentation.',
        thresholds=dict(simple=[.02,.04,.06,.08,.12,.18],cnn=[.2,.35,.5,.65,.8],min_areas=[3,8,15],simple_openings=[1,3]),
        inference='Boxes without probability labels. CNN sigmoid scores remain internal threshold inputs.',
        uncertainty='2000 paired sequence-cluster bootstrap draws; aggregate micro box F1 and pixel IoU. Exploratory intervals, not multiplicity adjusted.',
        controls='Frozen median and hybrid with cross-sequence shuffled references, plus all-background and fixed central-band predictions.',
        limitations=['Synthetic procedural TOPO only','Constant B is assumed, not physically qualified',
                     'Straight native-aligned crops only','No circular-weld or real SEM validation',
                     'A median is robust to isolated outliers; two or more contaminated neighbours can corrupt it'],
        source_hashes=sources(), preserved_historical_files={str(p.resolve()):sha(p) for p in
            (OLD/'frozen_models.json',OLD/'FINAL_CAMPAIGN_MANIFEST.json',DEV/'topo_normalization.json',OLD/'model/assumed_constant.json')},
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in protocol['source_hashes']:
            archive.write(path,str(Path(path).relative_to(REPO)))
    receipt(root,'prepare',[root/'protocol.json',root/'source_snapshot.zip'])


def audit_development(root):
    check_protocol(root)
    report=audit_sequence_dataset(DEV)
    write(root/'development_audit.json',report)
    if report['status']!='ok':
        raise ValueError('Development audit failed; inspect development_audit.json.')
    data=load_windows(DEV)
    visibility_audit(data,root/'visibility_audit.json')
    counterfactual_audit(DEV,root/'counterfactual',REPO/'DataSimulation/config',OLD/'model/assumed_constant.json')
    write(root/'development_inventory.json',inventory(DEV))
    write(root/'population.json',{s:dict(targets=len(subset(data,s)['meta']),
        sequences=len({m['sequence_id'] for m in subset(data,s)['meta']})) for s in ('train','val','test')})
    receipt(root,'audit',[*[root/p for p in ('development_audit.json','visibility_audit.json','development_inventory.json','population.json')],*(root/'counterfactual').glob('*')])


def train(root):
    protocol=check_protocol(root); require(root,'audit')
    verify_files(read(root/'development_inventory.json'))
    data=load_windows(DEV); train_data=subset(data,'train'); val=subset(data,'val')
    folder=root/'training'; folder.mkdir(exist_ok=True)
    simple=fit_simple(val,folder)
    specs={arm:dict(arm=arm,**selected) for arm,selected in simple.items()}
    for arm in ('cnn','cnn_median'):
        runs=[fit_network(train_data,val,folder,arm,seed,protocol['epochs']) for seed in protocol['seeds']]
        chosen=max(runs,key=lambda r:selection_key(r['selected']['metrics']))
        specs[arm]=dict(arm=arm,seed=chosen['seed'],epoch=chosen['selected']['epoch'],
            parameters=chosen['selected']['parameters'],metrics=chosen['selected']['metrics'],
            checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'])
    write(root/'selected_models.json',specs)
    receipt(root,'train',[root/'selected_models.json',*folder.glob('*.json'),*folder.glob('*.pt')])


def freeze(root):
    check_protocol(root); require(root,'train')
    selected=read(root/'selected_models.json')
    frozen=dict(models=selected,primary=max(ARMS,key=lambda a:selection_key(selected[a]['metrics'])),
        normalization=dict(path=str(DEV/'topo_normalization.json'),sha256=sha(DEV/'topo_normalization.json')),
        protocol_sha256=sha(root/'protocol.json'),source_hashes=sources(),
        frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(root/'frozen_models.json',frozen)
    receipt(root,'freeze',[root/'frozen_models.json'])


def generate(root):
    protocol=check_protocol(root); require(root,'freeze')
    from bse_renderer.domain_randomisation.dataset_forms import generate_topo_sequence_dataset
    destination=root/'confirmation'
    summary=generate_topo_sequence_dataset(config_dir=REPO/'DataSimulation/config',output_dir=destination,
        sequences_per_class=protocol['confirmation_sequences_per_class'],frames_per_sequence=9,
        frame_size=64,stride_px=64,context_radius=2,pixel_size_mm=.05,seed=protocol['confirmation_seed'],
        render_model_mode='formula_only',emission_b_mode='assumed_constant',
        emission_b_path=str(OLD/'model/assumed_constant.json'),normalization_path=DEV/'topo_normalization.json',
        weld_layout_mode='straight_seam',noise_profile='development',fixed_energy_kev=60)
    write(root/'confirmation_generation.json',summary)
    report=audit_sequence_dataset(destination,compare_roots=[DEV,OLD/'datasets/sequence_confirmation'])
    write(root/'confirmation_audit.json',report)
    if report['status']!='ok':
        raise ValueError('Confirmation audit failed.')
    write(root/'confirmation_inventory.json',inventory(destination))
    receipt(root,'generate',[root/'confirmation_generation.json',root/'confirmation_audit.json',root/'confirmation_inventory.json'])


def cluster_intervals(rows_by_arm, draws=2000):
    arms=list(rows_by_arm); ids=sorted({r['sequence_id'] for r in rows_by_arm[arms[0]]})
    columns=('tp','fp','fn','pixel_tp','pixel_fp','pixel_fn')
    totals={a:np.array([[sum(r[c] for r in rows_by_arm[a] if r['sequence_id']==s) for c in columns] for s in ids]) for a in arms}
    rng=np.random.default_rng(921990001)
    samples={a:[] for a in arms}
    for _ in range(draws):
        chosen=rng.integers(0,len(ids),len(ids))
        for arm in arms:
            tp,fp,fn,pt,pf,pn=totals[arm][chosen].sum(axis=0)
            samples[arm].append([2*tp/max(2*tp+fp+fn,1),pt/max(pt+pf+pn,1)])
    samples={a:np.array(v) for a,v in samples.items()}
    ci={a:{metric:np.quantile(samples[a][:,i],[.025,.975]).tolist() for i,metric in enumerate(('box_f1','pixel_iou'))} for a in arms}
    comparisons={}
    for a,b in [('cnn_median','cnn'),('cnn_median','median'),('median','thresholding'),('cnn','thresholding'),
                ('median','median_shuffled'),('cnn_median','cnn_median_shuffled')]:
        comparisons[f'{a}_minus_{b}']={metric:np.quantile(samples[a][:,i]-samples[b][:,i],[.025,.975]).tolist()
                                     for i,metric in enumerate(('box_f1','pixel_iou'))}
    return dict(draws=draws,cluster_unit='sequence',intervals=ci,paired_difference_intervals=comparisons)


def shuffled_references(data):
    """Deterministic cross-sequence assignment, without consulting defect labels."""
    groups={}
    for i,m in enumerate(data['meta']): groups.setdefault(m['sequence_id'],[]).append(i)
    ids=list(groups); rng=np.random.default_rng(921880001); rng.shuffle(ids)
    reference=data['reference'].copy()
    for i,sid in enumerate(ids):
        src=groups[ids[(i+1)%len(ids)]]
        dest=groups[sid]
        if len(src)!=len(dest): raise ValueError('Shuffled control requires equal windows per sequence.')
        reference[dest]=data['reference'][src]
    return {**data,'reference':reference}


def evaluate(root):
    check_protocol(root); require(root,'freeze'); require(root,'generate')
    verify_files(read(root/'development_inventory.json')); verify_files(read(root/'confirmation_inventory.json'))
    frozen=read(root/'frozen_models.json')
    populations={'development_test':subset(load_windows(DEV),'test'),'confirmation':load_windows(root/'confirmation')}
    results={}
    for population,data in populations.items():
        rows={a:evaluate_scores(data,scores_for(data,frozen['models'][a]),frozen['models'][a]['parameters']) for a in ARMS}
        shuffled=shuffled_references(data)
        for a in ('median','cnn_median'):
            rows[a+'_shuffled']=evaluate_scores(data,scores_for(shuffled,frozen['models'][a]),frozen['models'][a]['parameters'])
        zero=np.zeros_like(data['target']); band=zero.copy(); band[:,24:40,:]=1
        for name,score in [('all_background',zero),('fixed_central_band',band)]:
            rows[name]=evaluate_scores(data,score,dict(threshold=.5,min_area=1))
        results[population]=dict(metrics={a:summarize(r) for a,r in rows.items()},
            per_class={a:{c:summarize([r for r in rr if r['source_class']==c]) for c in sorted({r['source_class'] for r in rr})} for a,rr in rows.items()},
            uncertainty=cluster_intervals(rows))
        for arm,rr in rows.items(): write(root/'predictions'/f'{population}_{arm}.json',rr)
        if population=='confirmation':
            make_examples(data,rows,root/'confirmation_examples.png')
    write(root/'results.json',dict(primary=frozen['primary'],**results))
    receipt(root,'evaluate',[root/'results.json',root/'confirmation_examples.png',*(root/'predictions').glob('*.json')])


def make_examples(data,rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    classes=sorted({m['source_class'] for m in data['meta']})
    fig,axes=plt.subplots(len(classes),5,figsize=(13,2.3*len(classes)))
    for r,label in enumerate(classes):
        ids=[i for i,m in enumerate(data['meta']) if m['source_class']==label]
        idx=max(ids,key=lambda i:int(data['mask'][i].sum()))
        for col,arm in enumerate(('truth',)+ARMS):
            ax=axes[r,col]; ax.imshow(data['target'][idx],cmap='gray',vmin=0,vmax=1)
            boxes=data['meta'][idx]['truth_boxes'] if arm=='truth' else rows[arm][idx]['boxes_xyxy']
            for x0,y0,x1,y1 in boxes: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime' if arm=='truth' else 'orange',lw=1))
            ax.set_title(f'{label}\n{arm}',fontsize=8);ax.axis('off')
    fig.suptitle('Largest labelled target per source class, chosen from masks, not model success. Green truth; orange prediction.')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def verify(root):
    check_protocol(root)
    for stage in ('audit','train','freeze','generate','evaluate'): require(root,stage)
    verify_files(read(root/'development_inventory.json'));verify_files(read(root/'confirmation_inventory.json'))
    frozen=read(root/'frozen_models.json');verify_files(frozen['source_hashes'])
    for model in frozen['models'].values():
        if 'checkpoint' in model: verify_files({model['checkpoint']:model['checkpoint_sha256']})
    smoke_inference(root)
    write(root/'verification.json',dict(status='ok',operational=True,physical_emission_qualified=False,
        historical_files_preserved=True,frozen_settings_verified=True,confirmation_opened_after_freeze=True,
        note='Model performance is reported separately in results.json. Integrity does not establish physical validity.'))


def smoke_inference(root):
    from .inference import infer
    from src.data.temporal_window_dataset import TemporalWindowDataset
    ds=TemporalWindowDataset(root/'confirmation','all')
    window=next(w for w in ds.windows if w['target_is_defective'])
    record=ds.record_map[window['target_sample_id']]
    neighbours=window['previous_sample_ids']+window['future_sample_ids']
    destination=root/'inference_smoke';destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/'input.npz',
        target_raw=np.load(root/'confirmation'/record['raw_channels']['topo']),
        context_raw=np.stack([np.load(root/'confirmation'/ds.record_map[s]['raw_channels']['topo']) for s in neighbours]),
        context_offsets_xy=np.array([[-128,0],[-64,0],[64,0],[128,0]]),
        target_origin_xy=np.array([record['crop_x0_px'],record['crop_y0_px']]),
        energy_kev=60.,pixel_size_mm=.05,layout='straight_seam',rotation_deg=0.)
    for arm in ARMS:
        result=infer(root,destination/'input.npz',destination/f'{arm}.json',arm)
        expected=next(r for r in read(root/'predictions'/f'confirmation_{arm}.json') if r['window_id']==window['window_id'])
        if result['boxes_xyxy']!=expected['boxes_xyxy']: raise ValueError('Raw-input inference differs from evaluation.')
    write(destination/'verification.json',dict(status='ok',input_contains_labels=False,all_four_match_evaluation=True,window_id=window['window_id']))


STAGES=dict(prepare=prepare,audit=audit_development,train=train,freeze=freeze,generate=generate,evaluate=evaluate,verify=verify)


def run(root,stage):
    root=Path(root).resolve();torch.set_num_threads(4)
    for name in (list(STAGES) if stage=='all' else [stage]):
        if (root/'receipts'/f'{name}.json').exists():
            require(root,name)
            if name!='prepare': check_protocol(root)
            print(f'Hash-verified completed stage: {name}',flush=True)
            continue
        print(f'Starting {name}',flush=True);start=time.monotonic()
        STAGES[name](root)
        print(f'Completed {name} in {time.monotonic()-start:.1f}s',flush=True)
