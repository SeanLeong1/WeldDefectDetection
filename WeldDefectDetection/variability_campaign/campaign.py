"""Versioned, paired clean/pore variability experiment with prospective selection."""
from pathlib import Path
from dataclasses import asdict, replace
import json
import time
import zipfile
import numpy as np
import torch
from bse_renderer.config_loader import load_config_bundle, default_slice_runtime
from bse_renderer.domain_randomisation.parameter_sampler import sample_parameters
from bse_renderer.domain_randomisation.dataset_forms import _set_regime, _apply_noise_profile, _regime_for_sample
from bse_renderer.models import WeldLayoutRuntime
from box_comparison.core import sha, write, subset, truth_box, summarize, selection_key
from box_development.methods import fit_scale, simple_scores, evaluate_spec, scores_for
from box_development.training import fit
from box_refinement.methods import select
from pore_campaign.generation import pore_parameters
from .renderer import VariableSurfaceEngine, gradients

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO/'DataSimulation/data/output/assumed_constant_campaign_20260915_v2/model/assumed_constant.json'
KINDS = ('clean', 'keyhole_pore', 'intrinsic_pore')
ARMS = ('thresholding', 'median', 'cnn')
FRAME, FRAMES, SPACING = 64, 9, .05
SEEDS = (926810001, 926810002, 926810003)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def source_paths():
    paths = list((REPO/'DataSimulation/bse_renderer').rglob('*.py'))
    paths += list((REPO/'DataSimulation/config').glob('*.yaml'))
    for directory in ('variability_campaign','box_comparison','box_refinement','box_development','pore_campaign','src'):
        paths += list((REPO/'WeldDefectDetection'/directory).rglob('*.py'))
    paths += [REPO/'WeldDefectDetection/run_variability_campaign.py', MODEL]
    return sorted(set(paths))


def verify_sources(root):
    p = read(root/'protocol.json')
    changed = [f for f,h in p['source_hashes'].items() if sha(f) != h]
    if changed:
        raise ValueError(f'Prospectively frozen sources changed: {changed}')
    return p


def seal(root, stage, files):
    write(root/'receipts'/f'{stage}.json', dict(files={str(Path(p).resolve()):sha(p) for p in files}))


def require(root, stage):
    for path,digest in read(root/'receipts'/f'{stage}.json')['files'].items():
        if sha(path) != digest:
            raise ValueError(f'Changed {stage} artifact: {path}')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        verify_sources(root)
        return
    p = dict(schema='paired-weld-variability/v1', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        objective='Detect surface-expressed pores while tolerating declared background surface variation.',
        domain=dict(material='304SS', energy_kev=60., pixel_size_mm=SPACING, frame_size=FRAME, frames=FRAMES,
                    B=read(MODEL)['fitted_b'], B_status='engineering_assumption', gradient_method='finite_difference'),
        development=dict(backgrounds=100, seed=1260100000, train=60, val=20, test=20, amplitudes=[0.,.05,.10]),
        confirmation=dict(backgrounds=30, seed=1360100000, amplitudes=[0.,.05,.10,.20], generated_after_model_freeze=True),
        pilot=dict(backgrounds=3, seed=1460100000),
        variation=dict(width_and_depth_fraction_bound='amplitude', axial_control_spacing_mm=6.4,
                       spatial_function='natural cubic through random controls, bounded after interpolation',
                       background_warp='bilinear clean relief width warp then multiplicative depth scaling',
                       defect_policy='Add unchanged empirical pore relief/occupancy in original coordinates',
                       acceptance='Synthetic experiment tolerances only; not a manufacturing standard'),
        pairing='Each background has clean, keyhole and intrinsic scenes with identical acquisition parameters; all related scenes/levels stay in one split.',
        labels='Geometric surface-slab masks; no contrast filtering or invisible-case removal.',
        preprocessing='Fit 1/99-percentile range with 8 percent padding on noisy development TRAIN pixels only; sqrt tone curve; raw float32 for all paths.',
        training=dict(seeds=list(SEEDS), epochs=20, batch_size=32, threads=4,
                      architecture='TinyDefectSegNet target-only', recipe='Existing balanced sampler, BCE+Dice, AdamW, cosine schedule; validation at 5/10/15/20 epochs.'),
        selection='Validation micro box F1, tie pixel IoU then fewer FP; no confirmation retuning; CNN primary.',
        methods=list(ARMS),
        metrics='IoU>=0.5 one-to-one box F1/recall; pixel IoU; false alarms on all negative crops and clean-source crops; 2000 paired background-cluster bootstrap draws.',
        spline='Separate analytic and paired-render diagnostics, not used to select primary training generator.',
        noise='Historical development noise; noiseless channels captured before contamination/Poisson/thermal with identical geometry.',
        excluded='Independent beam-direction variation, internal-pore detection, physical SEM validation, report-outline changes.',
        source_hashes={str(p.resolve()):sha(p) for p in source_paths()})
    write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in p['source_hashes']:
            archive.write(name,str(Path(name).relative_to(REPO)))
    seal(root,'prepare',[root/'protocol.json',root/'source_snapshot.zip'])


def engine(method='finite_difference'):
    return VariableSurfaceEngine(image_size=FRAME*FRAMES, pixel_size_mm=SPACING,
        render_model_mode='formula_only', emission_b_mode='assumed_constant', emission_b_path=str(MODEL),
        render_view_mode='top_surface', weld_layout=WeldLayoutRuntime(mode='straight_seam'),
        frame_size=FRAME, halo=32, gradient_method=method)


def sampled(cfg, seed, index):
    sample=sample_parameters(cfg, seed=seed, material_name='304SS', max_defects=0, randomize_defect_location=True)
    sample=_set_regime(_apply_noise_profile(sample,'development'), _regime_for_sample(index,0), 60.)
    target=2+index%5
    return replace(sample, slice=replace(default_slice_runtime(cfg.slice),slice_offset_mm=(target-4)*FRAME*SPACING)),target


def generate(root, phase):
    p=verify_sources(root);settings=p[phase]
    if phase=='confirmation':require(root,'train')
    cfg=load_config_bundle(REPO/'DataSimulation/config')
    folder=root/'datasets'/phase;folder.mkdir(parents=True,exist_ok=True)
    rows=[];files=[];started=time.monotonic()
    for i in range(settings['backgrounds']):
        seed=settings['seed']+i
        sample,target=sampled(cfg,seed,i)
        split='confirmation' if phase=='confirmation' else ('train' if i<60 else 'val' if i<80 else 'test')
        # Beam regime cycles at i % 3. Use a different cycle so every regime
        # occurs at every variation level, rather than confounding the two.
        levels=settings['amplitudes'] if phase=='confirmation' else [settings['amplitudes'][(i//3)%3]]
        e=engine()
        for amplitude in levels:
            e.set_variation(amplitude)
            clean_quiet=None
            for kind in KINDS:
                sid=f'{phase}-{i:03d}-a{round(amplitude*100):02d}-{kind}'
                npz=folder/(sid+'.npz');meta=folder/(sid+'.json')
                # Resumption checks each scene against its saved digest; the clean
                # render is still needed to establish the exact baseline in RAM.
                if meta.exists() and npz.exists() and kind!='clean':
                    row=read(meta)
                    if row['npz_sha256']!=sha(npz):raise ValueError(f'Changed scene {sid}')
                    rows.append(row);files.extend([npz,meta]);continue
                defects=pore_parameters(kind,np.random.default_rng(seed+310003))
                result=e.run(replace(sample,defects=defects))
                sl=slice(e.halo,e.halo+FRAME)
                raw=result.channels.topo[sl].copy();quiet=e.quiet_topo[sl].copy()
                mask=result.defect_mask[sl].copy()
                if kind=='clean':clean_quiet=quiet.copy()
                if not np.isfinite(raw).all() or not np.isfinite(quiet).all():raise ValueError('Nonfinite signal')
                if kind!='clean' and not mask[:,target*FRAME:(target+1)*FRAME].any():
                    raise ValueError(f'Geometric target intersection absent, retained for investigation: {sid}')
                y0=(e.image_size-FRAME)//2
                relief=e.full_relief[y0:y0+FRAME].copy()
                delta=e.pore_delta[y0:y0+FRAME]
                n=e.last_normals[sl]
                row=dict(scene_id=sid,background_id=f'{phase}-{i:03d}',seed=seed,split=split,source_class=kind,
                         amplitude=amplitude,target_frame=target,parameters=asdict(replace(sample,defects=defects)),
                         renderer_metadata=result.metadata,npz_path=str(npz.resolve()),
                         realized_relief_change_um=float(np.max(np.abs(delta))*1000),
                         signal_change_rms=float(np.sqrt(np.mean((quiet-clean_quiet)**2))),
                         signal_change_in_mask=float(np.mean(np.abs(quiet-clean_quiet)[mask>0])) if mask.any() else 0.,
                         noise_rms=float(np.sqrt(np.mean((raw-quiet)**2))),
                         incidence_quantiles_deg=np.percentile(np.degrees(np.arccos(np.clip(n[...,2],-1,1))),[0,50,95,99,100]).tolist())
                if meta.exists():
                    old=read(meta)
                    with np.load(npz) as saved:
                        if not np.array_equal(saved['raw'],raw) or not np.array_equal(saved['mask'],mask):
                            raise ValueError('Resumed clean scene did not replay exactly.')
                    row=old
                else:
                    np.savez_compressed(npz,raw=raw,quiet=quiet,mask=mask,relief=relief,
                                        width_fraction=1+amplitude*e.width_field,depth_fraction=1+amplitude*e.depth_field)
                    row['npz_sha256']=sha(npz);write(meta,row)
                rows.append(row);files.extend([npz,meta])
        print(f'{phase} background {i+1}/{settings["backgrounds"]} elapsed={time.monotonic()-started:.1f}s',flush=True)
    write(folder/'index.json',rows);files.append(folder/'index.json')
    # A background is the independent grouping unit, including all pore variants.
    splits={}
    for row in rows:splits.setdefault(row['background_id'],set()).add(row['split'])
    if any(len(s)>1 for s in splits.values()):raise ValueError('Background split leakage')
    if phase=='confirmation':
        dev=read(root/'datasets/development/index.json')
        if {r['seed'] for r in dev}&{r['seed'] for r in rows}:raise ValueError('Seed overlap')
    write(root/f'{phase}_audit.json',dict(status='ok',scenes=len(rows),backgrounds=len(splits),
          windows=len(rows)*5,split_by_background=True,paired_labels_preserved=True,
          elapsed_seconds=time.monotonic()-started))
    files.append(root/f'{phase}_audit.json');seal(root,phase,files)


def fit_normalization(root):
    records=read(root/'datasets/development/index.json')
    values=[];ids=[]
    for r in records:
        if r['split']=='train':
            with np.load(r['npz_path']) as a:values.append(a['raw'].reshape(-1))
            ids.append(r['scene_id'])
    lo,hi=np.quantile(np.concatenate(values),[.01,.99]);pad=.08*(hi-lo)
    norm=dict(raw_low=max(0.,float(lo-pad)),raw_high=float(hi+pad),tone_curve='sqrt',source_split='train',scene_ids=ids)
    if norm['raw_high']<=norm['raw_low']:raise ValueError('Degenerate training normalization')
    write(root/'normalization.json',norm)


def load(root, phase, quiet=False):
    norm=read(root/'normalization.json');records=read(root/'datasets'/phase/'index.json')
    targets=[];contexts=[];masks=[];meta=[]
    for row in records:
        with np.load(row['npz_path']) as a:
            raw=a['quiet' if quiet else 'raw'];mask=a['mask']
            normalized=np.sqrt(np.clip((raw-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1)).astype(np.float32)
        frames=np.stack([normalized[:,k*FRAME:(k+1)*FRAME] for k in range(FRAMES)])
        mm=np.stack([mask[:,k*FRAME:(k+1)*FRAME] for k in range(FRAMES)])
        for t in range(2,7):
            ix=[t-2,t-1,t+1,t+2]
            targets.append(frames[t]);contexts.append(frames[ix]);masks.append(mm[t])
            meta.append(dict(window_id=f'{row["scene_id"]}-w{t}',sequence_id=row['background_id'],
                       scene_id=row['scene_id'],source_class=row['source_class'],split=row['split'],
                       amplitude=row['amplitude'],target_frame=t,context_defective_count=int(mm[ix].reshape(4,-1).any(1).sum()),
                       truth_boxes=truth_box(mm[t]),diameter_mm=(row['parameters']['defects'][0]['params']['diameter_mm'] if row['parameters']['defects'] else None),
                       realized_relief_change_um=row['realized_relief_change_um']))
    context=np.stack(contexts)
    return dict(target=np.stack(targets),context=context,reference=np.median(context,axis=1),mask=np.stack(masks),meta=meta)


def train(root):
    verify_sources(root);require(root,'development');require(root,'pilot')
    fit_normalization(root)
    data=load(root,'development');training=subset(data,'train');val=subset(data,'val')
    folder=root/'training';folder.mkdir(exist_ok=True)
    candidates={a:[] for a in ARMS}
    for arm in ('thresholding','median'):
        for sigma in ((4.,8.,12.) if arm=='thresholding' else (8.,)):
            scale,diagnostic=fit_scale(training,arm,sigma)
            spec=dict(arm=arm,implementation='local_variation',background_sigma=sigma,scale=scale.tolist(),scale_fit=diagnostic)
            chosen,trials=select(val,simple_scores(val,spec),(1.,1.5,2.,3.,4.,6.,8.),min_areas=(3,15),closings=(1,3),low_ratios=(1.,),paddings=(0,))
            spec.update(chosen);candidates[arm].append(spec)
            write(folder/f'{arm}_{sigma}.json',dict(spec=spec,trials=trials))
    for seed in SEEDS:
        candidates['cnn'].append(fit(training,val,folder,'cnn',seed,1,'cnn',epochs=20))
    selected={a:max(candidates[a],key=lambda s:selection_key(s['metrics'])) for a in ARMS}
    write(root/'model_candidates.json',candidates)
    write(root/'frozen_models.json',dict(models=selected,primary='cnn',normalization_sha256=sha(root/'normalization.json'),
          protocol_sha256=sha(root/'protocol.json'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    write(root/'development_population.json',{s:dict(windows=len(subset(data,s)['meta']),backgrounds=len({m['sequence_id'] for m in subset(data,s)['meta']}),
          positive_windows=int(subset(data,s)['mask'].reshape(len(subset(data,s)['meta']),-1).any(1).sum())) for s in ('train','val','test')})
    seal(root,'train',[root/'model_candidates.json',root/'frozen_models.json',root/'normalization.json',root/'development_population.json',*folder.glob('*.pt'),*folder.glob('*.json')])


def extended_summary(rows):
    s=summarize(rows)
    s['clean_crop_false_alarm_rate']=s['clean_false_alarms']/max(s['clean_targets'],1)
    clean=[r for r in rows if r['source_class']=='clean']
    s['clean_source_false_alarm_rate']=sum(r['predicted_positive'] for r in clean)/max(len(clean),1)
    s['clean_source_targets']=len(clean)
    return s


def clustered_intervals(rows_by_arm, draws=2000):
    ids=sorted({r['sequence_id'] for r in rows_by_arm['cnn']})
    totals={}
    for arm,rows in rows_by_arm.items():
        totals[arm]=np.array([[sum(r[k] for r in rows if r['sequence_id']==sid) for k in ('tp','fp','fn')]+
            [sum(r['predicted_positive'] and not r['target_positive'] for r in rows if r['sequence_id']==sid),
             sum(not r['target_positive'] for r in rows if r['sequence_id']==sid)] for sid in ids])
    rng=np.random.default_rng(926990003);ix=rng.integers(len(ids),size=(draws,len(ids)))
    values={}
    for arm,t in totals.items():
        z=t[ix].sum(1);tp,fp,fn,fa,neg=z.T
        values[arm]=np.stack([2*tp/np.maximum(2*tp+fp+fn,1),tp/np.maximum(tp+fn,1),fa/np.maximum(neg,1)],axis=1)
    keys=('box_f1','box_recall','clean_crop_false_alarm_rate')
    return dict(cluster_unit='latent_background_all_related_variants',backgrounds=len(ids),draws=draws,
        intervals={a:{k:np.quantile(v[:,i],[.025,.975]).tolist() for i,k in enumerate(keys)} for a,v in values.items()},
        cnn_minus_median={k:np.quantile((values['cnn']-values['median'])[:,i],[.025,.975]).tolist() for i,k in enumerate(keys)})


def evaluate(root):
    verify_sources(root);require(root,'confirmation');require(root,'train')
    frozen=read(root/'frozen_models.json');results={};all_rows={}
    for quiet in (False,True):
        label='quiet' if quiet else 'noisy';data=load(root,'confirmation',quiet)
        rows={a:evaluate_spec(data,s) for a,s in frozen['models'].items()}
        all_rows[label]=rows
        for a,rr in rows.items():write(root/'predictions'/f'{label}_{a}.json',rr)
        results[label]={}
        for level in (0.,.05,.10,.20):
            grouped={a:[r for r in rr if r['amplitude']==level] for a,rr in rows.items()}
            results[label][str(level)]=dict(metrics={a:extended_summary(rr) for a,rr in grouped.items()},uncertainty=clustered_intervals(grouped),
                per_pore={a:{kind:extended_summary([r for r in rr if r['source_class']==kind]) for kind in KINDS[1:]} for a,rr in grouped.items()},
                diameter_bins={a:{name:extended_summary([r for r in rr if r['diameter_mm'] is not None and low<=r['diameter_mm']<high])
                    for name,low,high in [('0.50_to_0.75',.5,.75),('0.75_to_1.21',.75,1.21)]} for a,rr in grouped.items()})
    test=subset(load(root,'development'),'test')
    results['reserved_development_test']={a:extended_summary(evaluate_spec(test,s)) for a,s in frozen['models'].items()}
    results['claim']='Held-out synthetic performance. No measured SEM or weld acceptance validation.'
    write(root/'results.json',results)
    seal(root,'evaluate',[root/'results.json',*(root/'predictions').glob('*.json')])


def run(root, stage):
    root=Path(root);torch.set_num_threads(4)
    stages=['prepare','pilot','development','train','confirmation','evaluate','report']
    for name in stages if stage=='all' else [stage]:
        receipt=root/'receipts'/f'{name}.json'
        if receipt.exists():
            verify_sources(root);require(root,name);print(f'Verified completed {name}',flush=True);continue
        if name=='prepare':prepare(root)
        elif name=='pilot':
            from .diagnostics import pilot
            pilot(root)
        elif name in ('development','confirmation'):generate(root,name)
        elif name=='train':train(root)
        elif name=='evaluate':evaluate(root)
        elif name=='report':
            from .report import build
            build(root)
