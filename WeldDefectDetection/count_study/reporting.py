"""Audit-ready summaries, paired comparisons and matched example figures."""
from dataclasses import replace,asdict
from pathlib import Path
import gc
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import campaign as c, learning as learn
from raised_crown import rendering as r, learning as base
from circular_study import metrics
from bse_renderer.noise.raster_acquisition import AcquisitionProfile, acquire
from bse_renderer.renderer.image_formation import form_channels

FIG=c.ROOT/'docs/figures/count_diversity_20261027'


def diagnostic():
    """Independent teaching/ceiling case; never enters any selection."""
    folder=c.OUT/'diagnostic';folder.mkdir(exist_ok=True)
    if (folder/'index.json').exists():return
    seed=c.BASES['diagnostic'];cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config')
    sample,_=r.oldg.baseline.c.sampled(cfg,seed,0)
    sample=replace(sample,beam=replace(sample.beam,dwell_time_us=8.))
    engine=r.Engine(seed,0.);engine.acquisition_profile=AcquisitionProfile()
    rows=[]
    for kind in ('clean','open_intrinsic'):
        plan=[] if kind=='clean' else [dict(type='intrinsic_pore',params=dict(kind='intrinsic_pore',effect='open',
            diameter_mm=.2,depth_mm=.2,angle_rad=1.,radial_offset_mm=1.05,coupling_fraction=0.,roof_mm=.15))]
        changed=replace(sample,defects=plan);arrays,meta=r.render(engine,changed)
        geometry=folder/f'{kind}_geometry.npz'
        np.savez_compressed(geometry,mask=arrays['mask'],height=arrays['height'],quiet=arrays['quiet'])
        q=engine.acquisition_quiet_quadrants;gain=float(np.interp(sample.beam.focusing_current_norm,[0,1],[1.10,.92]))
        for mode in ('noiseless','shot_only','full'):
            noise=replace(changed.noise,poisson_enabled=mode!='noiseless',thermal_enabled=mode=='full',contamination_enabled=mode=='full')
            for m in ((1,) if mode=='noiseless' else c.MULTIPLIERS):
                for repeat in range(1 if mode=='noiseless' else 2):
                    profile=replace(engine.acquisition_profile,effective_count_rate=engine.acquisition_profile.effective_count_rate*m)
                    x,_=acquire(q,noise,seed=seed+100000000*repeat,dwell_us=8.,profile=profile)
                    raw=r.extract(form_channels(x).topo*gain).astype('f4')
                    target=folder/f'{kind}_{mode}_c{m}_r{repeat}.npy';np.save(target,raw)
                    rows.append(dict(scene_id=target.stem,background_id='diagnostic-fixed',split='diagnostic',
                        raw_path=str(target),geometry_path=str(geometry),raw_sha256=c.sha(target),geometry_sha256=c.sha(geometry),
                        count_multiplier=m,repeat=repeat,source_class=kind,mode=mode,diameter_mm=0 if kind=='clean' else .2,
                        dwell_bin=3,amplitude=0.,profile=asdict(profile)))
                    del x,raw
        del arrays,q;gc.collect()
    c.write(folder/'index.json',rows)


def paired(a,b,repeats=2000):
    assert {x['scene_id'] for x in a}=={x['scene_id'] for x in b}
    keys=sorted({x['background_id'] for x in a});rng=np.random.default_rng(102710)
    def matrix(rows):
        return np.array([[sum(x[f] for x in rows if x['background_id']==k) for f in ('tp','fp','fn')] for k in keys])
    aa,bb=matrix(a),matrix(b)
    ids=rng.integers(len(keys),size=(repeats,len(keys)))
    av=aa[ids].sum(1);bv=bb[ids].sum(1)
    def f1(v):return 2*v[...,0]/np.maximum(2*v[...,0]+v[...,1]+v[...,2],1)
    delta=f1(av)-f1(bv)
    return dict(difference=float(f1(aa.sum(0))-f1(bb.sum(0))),interval_95=np.quantile(delta,[.025,.975]).tolist(),
                independent_backgrounds=len(keys),repeats=repeats)


def selected_results(results):
    chosen={}
    for name,result in results.items():
        if result['arm']=='retained':chosen[name]=result
        elif result['kind']=='median' or result.get('selected'):
            chosen[f'{result["arm"]}_{result["kind"]}']=result
    return chosen


def noise_summary():
    rows=c.read(c.OUT/'datasets/test/index.json');lookup={x['scene_id']:x for x in rows}
    records=[]
    for row in rows:
        if row['repeat']!=0:continue
        other=lookup[row['scene_id'][:-1]+'1']
        a=np.load(row['raw_path'],mmap_mode='r')[16:144]
        b=np.load(other['raw_path'],mmap_mode='r')[16:144]
        records.append(dict(background_id=row['background_id'],source_class=row['source_class'],
            count_multiplier=row['count_multiplier'],dwell_bin=row['dwell_bin'],
            repeat_difference_rms_over_sqrt2=float(np.sqrt(np.mean((a-b)**2)/2))))
    c.write(c.OUT/'measured_noise.json',records)
    return records


def diagnostic_evaluation(frozen):
    diagnostic();rows=c.read(c.OUT/'diagnostic/index.json');data=learn.Data(rows,c.read(c.OUT/'normalization.json'))
    output={}
    for arm in ('single1_96','single4_96','single64_96','mixed_96'):
        item=frozen['arms'][arm];model=base.model_for(item['cnn'])
        for kind in ('cnn','median'):
            spec=item[kind];scores=[]
            for i in range(len(rows)):
                scores.append(base.predict_image(data.image(i),model) if kind=='cnn' else
                              learn.median_score(data.image(i,spec['spec']['sigma']),spec['spec']))
            predictions=learn.evaluate_scores(data,np.stack(scores),spec['selection']['best'])
            for pred,meta in zip(predictions,rows):pred['mode']=meta['mode']
            output[f'{arm}_{kind}']=predictions
        del model;gc.collect()
    c.write(c.OUT/'diagnostic_results.json',output)


def figures(chosen,noise):
    FIG.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    arms=list(c.ARMS)
    for ax,kind in zip(axes,('median','cnn')):
        values=np.array([[chosen[f'{a}_{kind}']['by_count'][str(m)]['f1'] for m in c.MULTIPLIERS] for a in arms])
        im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set(xticks=range(3),xticklabels=['1x','4x','64x'],yticks=range(len(arms)),yticklabels=arms,
               xlabel='Confirmation effective counts',title=kind.upper()+' full-ring box F1')
        for (y,x),value in np.ndenumerate(values):ax.text(x,y,f'{value:.3f}',ha='center',va='center',color='white' if value<.55 else 'black')
    fig.colorbar(im,ax=axes,shrink=.7);fig.suptitle('Fresh synthetic confirmation: 48 backgrounds; count variants and repeats stay paired')
    fig.savefig(FIG/'train_test_matrix.png',dpi=180);fig.savefig(FIG/'train_test_matrix.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for bin_id in range(4):
        values=[np.median([x['repeat_difference_rms_over_sqrt2'] for x in noise if x['count_multiplier']==m and x['dwell_bin']==bin_id and x['source_class']=='clean']) for m in c.MULTIPLIERS]
        ax.plot(c.MULTIPLIERS,values,'o-',label=f'Dwell bin {bin_id+1}')
    ax.set(xscale='log',yscale='log',xlabel='Effective-count multiplier',ylabel='TOPO repeat-difference RMS / sqrt(2)',title='Measured total acquisition variability on clean rings')
    ax.legend();ax.grid(alpha=.3);fig.savefig(FIG/'measured_noise.png',dpi=180);plt.close(fig)
    # Predetermined example, never selected by detector performance.
    rows=[x for x in c.read(c.OUT/'datasets/test/index.json') if x['index']==0 and x['repeat']==0 and x['source_class'] in ('clean','open_intrinsic')]
    allrows=c.read(c.OUT/'datasets/test/index.json');index={x['scene_id']:i for i,x in enumerate(allrows)}
    frozen=c.read(c.OUT/'frozen_models.json');selection=frozen['arms']['mixed_96']
    cnn=np.load(c.OUT/'confirmation'/f'mixed_96_cnn_{selection["cnn"]["seed"]}.npy',mmap_mode='r')
    median=np.load(c.OUT/'confirmation/mixed_96_median.npy',mmap_mode='r')
    norm=c.read(c.OUT/'normalization.json')
    pore=next(x for x in rows if x['source_class']=='open_intrinsic')
    with np.load(pore['geometry_path']) as a:truth=a['mask'][16:144]
    ys,xs=np.where(truth);center=int(xs[len(xs)//2]);cols=(np.arange(96)+center-48)%5024
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for y,kind in enumerate(('clean','open_intrinsic')):
        for x,m in enumerate(c.MULTIPLIERS):
            row=next(z for z in rows if z['source_class']==kind and z['count_multiplier']==m)
            i=index[row['scene_id']];image=base.normalize(np.load(row['raw_path']),norm)[16:144]
            ax=axes[y,x];ax.imshow(image[:,cols],origin='lower',cmap='gray',vmin=0,vmax=1)
            with np.load(row['geometry_path']) as a:t=a['mask'][16:144][:,cols]
            if t.any():ax.contour(t,levels=[.5],colors=['lime'],linestyles='dashed',linewidths=1.5)
            for scores,method,color in ((cnn,'cnn','cyan'),(median,'median','orange')):
                op=selection[method]['selection']['best'];pred,_=metrics.binary(scores[i],op['threshold'],op['min_area'],op['closing'])
                if pred[:,cols].any():ax.contour(pred[:,cols],levels=[.5],colors=[color],linewidths=1.2)
            ax.set(title=f'{kind}, {m}x counts',xlabel='Local angular pixels',ylabel='Scored radial pixels')
    fig.suptitle('Fixed example test-000; green truth, orange median, cyan CNN\nMixed-noise models; display crop chosen from truth after complete-ring inference')
    fig.savefig(FIG/'matched_clean_pore.png',dpi=200);plt.close(fig)


def report():
    acquisition_sets={s:{r['acquisition_seed'] for r in c.read(c.OUT/f'datasets/{s}/index.json')} for s in c.COUNTS}
    assert not acquisition_sets['train']&acquisition_sets['val']
    assert not acquisition_sets['train']&acquisition_sets['test']
    assert not acquisition_sets['val']&acquisition_sets['test']
    assert c.read(c.OUT/'noise_checks.json')['status']=='passed'
    assert c.read(c.OUT/'method_checks.json')['status']=='passed'
    frozen=learn.verify_freeze();results=c.read(c.OUT/'results.json');chosen=selected_results(results)
    noise=noise_summary();figures(chosen,noise);diagnostic_evaluation(frozen)
    comparisons={}
    for m in c.MULTIPLIERS:
        take=lambda name:[x for x in chosen[name]['rows'] if x['count_multiplier']==m]
        pairs=[(f'single{m}_96_cnn',f'single{m}_96_median'),('mixed_96_cnn','mixed_96_median'),
               ('mixed_96_cnn','single1_96_cnn'),('mixed_96_cnn','mixed_24_cnn'),('single1_96_cnn','single1_24_cnn'),
               ('mixed_96_median','single1_96_median'),('mixed_96_median','mixed_24_median')]
        comparisons[str(m)]={f'{a} minus {b}':paired(take(a),take(b)) for a,b in pairs}
    c.write(c.OUT/'paired_comparisons.json',comparisons)
    lines=['# Effective-count and training-diversity study, 27 October 2026','',
        f'Campaign: `{c.OUT.name}`. All values below come from newly generated synthetic confirmation. No physical validation.',
        '', '## Completed design','',
        '96 training, 32 validation and 48 confirmation backgrounds; each has clean, intrinsic-pore and keyhole-pore conditions at 1x/4x/64x effective counts, with two acquisition repeats. Total: 3,168 scenes across 176 independent backgrounds. All paired variants remain in the same split.',
        'Count multiplication changes effective count conversion at fixed dwell, geometry and other acquisition parameters. It does not specify calibrated primary electrons. The original broad dwell/profile distribution remains, so these are relative count levels rather than three absolute noise amplitudes.',
        'All six arms use the same TinyDefectSegNet architecture, 20 epochs and 1,200 patch draws per epoch. Three seeds per arm, epochs 10/20 and detection operating points enter validation-only selection. Specialists select on their own count level; mixed arms maximize mean validation F1 across levels. The mixed/single and 24/96 comparisons preserve total training updates.',
        'The shared normalization uses the nested 24 training backgrounds at all count levels. Consequently, single-noise arms restrict supervised weight training, not exposure of the global normalization to the other count levels.',
        'Each median searches six Gaussian scales, six thresholds, three area cutoffs and two closing settings, with clean-training calibration. All final choices freeze before confirmation generation.',
        '', '## Fresh confirmation results','',
        '| Training arm / method | F1 at 1x | F1 at 4x | F1 at 64x |', '|---|---:|---:|---:|']
    for name,item in chosen.items():lines.append('| '+name+' | '+' | '.join(f'{item["by_count"][str(m)]["f1"]:.4f}' for m in c.MULTIPLIERS)+' |')
    lines+=['','## Training-seed variability','',
            '| Arm | Count | Minimum seed F1 | Maximum seed F1 |','|---|---:|---:|---:|']
    for arm in c.ARMS:
        candidates=[x for x in results.values() if x['arm']==arm and x['kind']=='cnn']
        for m in c.MULTIPLIERS:
            values=[x['by_count'][str(m)]['f1'] for x in candidates]
            lines.append(f'| {arm} | {m} | {min(values):.4f} | {max(values):.4f} |')
    lines+=['','![Train/test matrix](figures/count_diversity_20261027/train_test_matrix.png)','',
            '## Selected operating points','', '| Arm | CNN seed / epoch | CNN threshold / area / closing | Median sigma / threshold / area / closing |', '|---|---|---|---|']
    for arm,item in frozen['arms'].items():
        cnn=item['cnn'];co=cnn['selection']['best'];med=item['median'];mo=med['selection']['best']
        lines.append(f'| {arm} | {cnn["seed"]} / {cnn["epoch"]} | {co["threshold"]} / {co["min_area"]} / {co["closing"]} | {med["spec"]["sigma"]} / {mo["threshold"]} / {mo["min_area"]} / {mo["closing"]} |')
    lines+=['','## Precision, recall and clean alarms','', '| Method | Counts | Precision | Recall | Clean rings flagged | False regions on clean | Pixel IoU |','|---|---:|---:|---:|---:|---:|---:|']
    for name,item in chosen.items():
        for m in c.MULTIPLIERS:
            v=item['by_count'][str(m)]
            lines.append(f'| {name} | {m} | {v["precision"]:.3f} | {v["recall"]:.3f} | {v["clean_rings_flagged"]}/{v["clean_rings"]} | {v["clean_false_regions"]} | {v["pixel_iou"]:.3f} |')
    lines+=['','## Paired F1 differences','', '| Counts | Comparison | Difference | 95% background interval |','|---:|---|---:|---|']
    for m,pairs in comparisons.items():
        for name,v in pairs.items():lines.append(f'| {m} | {name} | {v["difference"]:+.3f} | [{v["interval_95"][0]:+.3f}, {v["interval_95"][1]:+.3f}] |')
    lines+=['','Intervals resample independent backgrounds, retaining all count, defect and noise-repeat variants together. They condition on fitted models; the saved results separately retain all three training seeds. These exploratory multiple comparisons are not multiplicity-adjusted confirmatory claims.',
        '', '## Noise and examples','', '![Measured noise](figures/count_diversity_20261027/measured_noise.png)',
        'Repeat-difference RMS includes acquisition variability. It is not calibrated physical electron noise. Nonlinear TOPO magnitude and unchanged electronic noise can prevent the full image from following the Poisson-only 1/sqrt(count) law.',
        '', '![Matched example](figures/count_diversity_20261027/matched_clean_pore.png)',
        '', 'The fixed .2 mm pore diagnostic includes noiseless, shot-only and full acquisition images. It never enters training or model selection. Its saved predictions diagnose a local ceiling; one geometry cannot establish population performance.',
        '', '## Limits and next steps','',
        'The comparison addresses relative count statistics and training diversity under declared ring coordinates. It does not resolve ring localisation, real-image transfer, detector calibration, buried-pore visibility, or measured beam-current/dwell equivalence.',
        'We retained all historical campaigns and did not implement the proposed two-layer CNN. A size study remains optional. The next scientific priorities are real-image calibration/transfer and localisation robustness. The next delivery task is independent compatible-image import and export.',
        'Small-pore sampling extends the earlier .5-1.0 mm training distribution, so old-model performance here is transfer to a joint size/background/count challenge. It is not an isolated causal comparison of training noise.',
        '', '## Evidence and reproduction','',
        f'Artifacts: `DataSimulation/data/output/{c.OUT.name}/`: protocol, noise checks, method checks, train/validation/test indexes, training histories and checkpoints, frozen manifest, all-seed predictions, paired comparisons, diagnostic results and audit.',
        'Run `.venv/Scripts/python.exe WeldDefectDetection/run_count_study_20261027.py all`. Completed stages verify and reuse their saved outputs. Historical code and data remain read-only.',
        'Prediction timers include image reading, normalization, network/median application and score-file writing. They exclude metric selection and bootstrap calculations. They are workflow timings, not pure model latency.']
    report_path=c.ROOT/'docs/COUNT_DIVERSITY_RESULTS_20261027.md';report_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(c.OUT/'completion.json',dict(status='complete',report=str(report_path),audit=c.read(c.OUT/'audit.json'),
        confirmation_models=len(results),physical_validation=False,two_layer_cnn=False))
