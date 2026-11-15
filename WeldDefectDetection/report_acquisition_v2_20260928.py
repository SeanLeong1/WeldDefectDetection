"""Numerical and visual audit of the v2 acquisition deliverables."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_acquisition_campaign_20260928 import OUT as CAMPAIGN, ROOT, c
from bse_renderer.noise.raster_acquisition import AcquisitionProfile, acquire
from bse_renderer.renderer.image_formation import form_channels
from bse_renderer.config_loader import load_config_bundle

DATA=ROOT/'DataSimulation/data/output/circular_acquisition_20260928_v2'
FIG=ROOT/'docs/figures/acquisition_v2_20260928'
FIG.mkdir(parents=True,exist_ok=True)


def visuals():
    with np.load(DATA/'a10_clean.npz') as a: raw=a['raw']; quiet=a['quiet']
    with np.load(ROOT/'docs/figures/circular_banding_diagnostic_20260928/staged_topo.npz') as a:
        old=a['complete_current_model']; oldquiet=a['before_acquisition_noise']
    err=float(np.max(np.abs(oldquiet-quiet))); assert err<1e-7,err
    yy,xx=np.indices(raw.shape); r=np.hypot((xx-960)*.05,(yy-960)*.05)
    bg=(r<35)|(r>44)
    def measure(a):
        masked=np.where(bg,a,np.nan)
        return dict(background_mean=float(a[bg].mean()),background_std=float(a[bg].std()),
            row_mean_std=float(np.nanmean(masked,axis=1).std()),column_mean_std=float(np.nanmean(masked,axis=0).std()))
    metrics={k:measure(a) for k,a in [('quiet',quiet),('legacy',old),('raster_v2',raw)]}
    metrics['same_geometry_quiet_max_error']=err
    # Fix the legacy display mapping for this comparison, avoiding independent contrast stretching.
    norm=c.read(ROOT/'DataSimulation/data/output/variability_campaign_20260926_v1/normalization.json')
    fig,axes=plt.subplots(1,3,figsize=(13.5,5.2))
    for ax,a,title,key in zip(axes,[quiet,old,raw],['Before acquisition','Legacy alternating-axis drift','Raster v2, assumed reference'],['quiet','legacy','raster_v2']):
        ax.imshow(np.sqrt(np.clip((a-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1)),cmap='gray',vmin=0,vmax=1,origin='lower')
        ax.set_title(title,fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"Background SD {metrics[key]['background_std']:.5f} raw units")
    fig.suptitle('Same 80 mm circular weld, same display scale',fontsize=18)
    fig.text(.5,.045,'Geometry, angular model and blur are unchanged. V2 amplitudes are unmeasured assumptions.\n'
        'Removing the arbitrary bands is a model correction; it does not establish agreement with real BSE.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.02,right=.98,top=.87,bottom=.2,wspace=.1)
    fig.savefig(FIG/'01_acquisition_comparison.png',dpi=150); plt.close(fig)
    with np.load(DATA/'clean_channel_reference.npz') as a: q=a['quiet_quadrants']; tx=a['tx']; ty=a['ty']; compo=a['compo']
    fig,axes=plt.subplots(1,4,figsize=(15.5,5.1))
    for ax,a,title in zip(axes,[compo,tx,ty,raw],['Detector sum','Signed horizontal difference Tx','Signed vertical difference Ty','Implemented magnitude TOPO']):
        if title.startswith('Signed'):
            hi=float(np.quantile(np.abs(a),.995)); lo=-hi
        else: lo,hi=map(float,np.quantile(a,[.005,.995]))
        im=ax.imshow(a,cmap='gray',vmin=lo,vmax=hi,origin='lower'); ax.set_title(title,fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im,ax=ax,fraction=.046,pad=.025).ax.tick_params(labelsize=7)
    fig.suptitle('One synthetic surface can produce very different detector images',fontsize=17)
    fig.text(.5,.05,'Each panel uses the stated raw-unit colorbar. Signed channels put zero at mid-gray.\n'
        'This illustrates channel dependence; the channel identities in the supplied screenshot remain unknown.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.02,right=.98,top=.88,bottom=.22,wspace=.23)
    fig.savefig(FIG/'02_detector_channels.png',dpi=150); plt.close(fig)
    cfg=load_config_bundle(ROOT/'DataSimulation/config'); sample,_=c.sampled(cfg,1360100000,0)
    gain=float(np.interp(sample.beam.focusing_current_norm,[0.,1.],[1.10,.92]))
    profiles={
        'lower_noise_assumption':replace(AcquisitionProfile(),effective_count_rate=2e9,readout_rms=.0002,baseline_rms=.0001,common_gain_rms=.001),
        'reference_assumption':AcquisitionProfile(),
        'higher_noise_assumption':replace(AcquisitionProfile(),effective_count_rate=5e8,readout_rms=.0008,baseline_rms=.0004,common_gain_rms=.004)}
    for name,p in profiles.items():
        noisy,meta=acquire(q,sample.noise,seed=sample.seed,dwell_us=sample.beam.dwell_time_us,profile=p)
        top=form_channels(noisy).topo*gain
        metrics[name]=dict(**measure(top),profile=meta['profile'],residual_rms=float(np.sqrt(np.mean((top-quiet)**2))))
        if name=='reference_assumption':
            metrics['v2_acquisition_exact_replay_max_error']=float(np.max(np.abs(top-raw)))
            assert metrics['v2_acquisition_exact_replay_max_error']<1e-7
        del noisy,top
    c.write(FIG/'measurements.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)


def delivery():
    c.verify_sources(CAMPAIGN)
    for stage in ('prepare','pilot','development','train','confirmation','evaluate'): c.require(CAMPAIGN,stage)
    results=c.read(CAMPAIGN/'results.json'); models=c.read(CAMPAIGN/'frozen_models.json')['models']
    rows=c.read(CAMPAIGN/'datasets/confirmation/index.json')
    dev=c.read(CAMPAIGN/'datasets/development/index.json')
    assert len(rows)==270 and len(dev)==300
    assert not {r['seed'] for r in rows}&{r['seed'] for r in dev}
    allrows=dev+rows
    assert all(r['renderer_metadata']['acquisition']['version']=='raster_v2' for r in allrows)
    assert all(max(r['incidence_quantiles_deg'])<=60 for r in allrows)
    # Keep a portable raw target plus four genuine neighbouring patches for the GUI.
    smoke=CAMPAIGN/'inference_smoke'; smoke.mkdir(exist_ok=True)
    record=next(r for r in rows if r['source_class']=='keyhole_pore' and r['amplitude']==0)
    with np.load(record['npz_path']) as a:
        frames=np.stack([a['raw'][:,k*64:(k+1)*64] for k in range(9)])
        t=record['target_frame']; target=frames[t]; context=frames[[t-2,t-1,t+1,t+2]]
    np.savez_compressed(smoke/'keyhole_pore_0.00.npz',target_raw=target,context_raw=context,pixel_size_mm=np.array(.05))
    c.write(smoke/'keyhole_pore_0.00.npz.json',dict(channel='topo_magnitude',pixel_size_mm=.05,source_kind='synthetic',
        source_scene_id=record['scene_id'],target_frame=t,context_frames=[t-2,t-1,t+1,t+2]))
    # All frozen historical model/data files remain checked against their existing manifest.
    legacy=ROOT/'DataSimulation/data/output/variability_campaign_20260926_v1'
    oldfiles=c.read(legacy/'FINAL_MANIFEST.json')['files']; checked=0
    for name,digest in oldfiles.items():
        path=Path(name)
        if not path.is_relative_to(legacy): continue
        assert c.sha(path)==digest,name; checked+=1
    reference=ROOT/'docs/references/poor_bse_reference_20260928.png'
    supplied=Path('C:/Users/sfleo/AppData/Local/Temp/codex-clipboard-f13bee99-5c29-4a3c-b0ea-47a4299e893d.png')
    if supplied.exists():
        reference.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(supplied,reference)
        c.write(reference.with_suffix('.json'),dict(source=str(supplied),sha256=c.sha(reference),
            scope='User identifies poor BSE examples with multiple test rings. Screenshot only; no calibration or expert defect labels inferred.'))
    lines=['# Raster acquisition v2: delivery and measured synthetic results','',
        'Completed a new acquisition model and a fresh CNN/median experiment. All numerical noise amplitudes remain assumptions. The user-supplied screenshot is qualitative context only.','',
        '## Five-step status','',
        '1. Common raster timing replaces arbitrary alternating detector axes. Stationary electronic drift has bounded variance.',
        '2. Geometric motion resamples both signal and labels separately from brightness drift. It defaults to zero in this experiment.',
        '3. Geometry and each acquisition mechanism use separate deterministic random streams.',
        '4. Reference and sensitivity profiles are recorded prospectively. Measured noise calibration remains pending suitable raw acquisitions.',
        '5. A new dataset, three completed CNN training runs, validation-frozen CNN/median settings, independent synthetic confirmation and new circular examples are produced.','',
        '## Population and protocol','',
        '300 development scenes / 1,500 windows: 60 training, 20 validation and 20 reserved-test backgrounds. Each background contains clean, keyhole and intrinsic variants.',
        '270 confirmation scenes / 1,350 windows: 30 new backgrounds, three variability levels and three classes. No confirmation thresholds were tuned.',
        'Variation changes background width/relief smoothly. These tolerances are synthetic study definitions, not weld acceptance limits.','',
        '## Held-out noisy confirmation','',
        '| Width/relief variation | Method | Box F1 | Recall | Clean-source false-alarm rate |',
        '|---|---|---:|---:|---:|']
    for level in ('0.0','0.05','0.1'):
        for arm in ('cnn','median'):
            m=results['noisy'][level]['metrics'][arm]
            lines.append(f"| {float(level):.0%} | {arm} | {m['box_f1']:.3f} | {m['box_recall']:.3f} | {m['clean_source_false_alarm_rate']:.1%} |")
    lines += ['', 'At 10% variability, per-pore results are:', '', '| Source class | Method | Box F1 | Recall |', '|---|---|---:|---:|']
    for kind in c.KINDS[1:]:
        for arm in ('cnn','median'):
            m=results['noisy']['0.1']['per_pore'][arm][kind]
            lines.append(f"| {kind} | {arm} | {m['box_f1']:.3f} | {m['box_recall']:.3f} |")
    lines += ['', 'Full counts, 2,000-draw background-cluster intervals, quiet-image diagnostics and reserved-test results are in `results.json` in the campaign directory.',
        'Three training runs used 20 epochs each. Validation selected the deployed checkpoint; the other seeds remain recorded.','',
        f"At 10% variation the same CNN scores {results['quiet']['0.1']['metrics']['cnn']['box_f1']:.3f} F1 on quiet images versus {results['noisy']['0.1']['metrics']['cnn']['box_f1']:.3f} on noisy images. Removing noise changes the input distribution; this is evidence of model dependence on acquisition appearance, not evidence that noise improves physical defect visibility.", '',
        '## Interpretation','',
        'These results establish performance within the declared synthetic model. Differences from the previous campaign combine a new noise model, new samples and retraining, so they are not a controlled estimate of real-world improvement.',
        'The matched circular quiet/legacy/v2 comparison isolates the acquisition change while keeping geometry and display scaling fixed. It does not establish agreement with genuine BSE.',
        'The CNN and median may miss small pores or flag acceptable variation. Report the misses and false alarms, including both pore classes separately. Do not call synthetic clean labels metallurgical acceptance decisions.',
        'The renderer models surface-expressed relief. The stated 8–10 mm process keyhole depth/width has not been calibrated into a volume-sensitive BSE model.','',
        '## Use and reproduce','',
        'The GUI inspection panel offers the new Raster v2 frozen model and keeps the legacy model selectable. Load example supplies one raw target and four genuine neighbours; ordinary photos support CNN but cannot automatically supply median context.',
        'For CLI inspection, pass `--campaign DataSimulation/data/output/acquisition_campaign_20260928_v2` to `WeldDefectDetection/inspect_images.py`. `--method all` uses the methods present in that campaign.',
        'Run `run_acquisition_campaign_20260928.py --stage all`, `circular_acquisition_v2_20260928.py`, then `report_acquisition_v2_20260928.py`. The campaign verifies frozen sources and stage receipts on resumption.',
        'The assumptions and equations are in `docs/acquisition_model_v2.md`; a corrected NIST source URL is in `docs/acquisition_model_v2_reference_erratum.md`. Dataset/model/source receipts are under `DataSimulation/data/output/acquisition_campaign_20260928_v2`.','',
        '## Checks','',f'{checked} historical campaign artifacts matched their frozen manifest. New campaign stage receipts, source hashes, separate background seeds, acquisition versions and local incidence <=60 degrees passed.',
        'The project automated suite passed 309 tests before the GUI model-selector addition; the final delivery audit records subsequent GUI/CLI checks separately.',
        'Real-image CV transfer, measured noise calibration and expert defect agreement remain pending.']
    (ROOT/'docs/ACQUISITION_V2_DELIVERY_20260928.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(CAMPAIGN/'delivery_audit.json',dict(status='passed',historical_hashes_checked=checked,development_scenes=len(dev),
        confirmation_scenes=len(rows),all_local_incidence_within_60_degrees=True,real_image_validation=False,
        cnn_epochs=[s['epochs'] for s in c.read(CAMPAIGN/'model_candidates.json')['cnn']],
        selected_seed=models['cnn']['seed'],selected_epoch=models['cnn']['epoch']))
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.6))
    levels=['0.0','0.05','0.1']; x=np.array([0,5,10])
    for arm,color in [('cnn','#087f8c'),('median','#b1641d')]:
        y=np.array([results['noisy'][s]['metrics'][arm]['box_f1'] for s in levels])
        ci=np.array([results['noisy'][s]['uncertainty']['intervals'][arm]['box_f1'] for s in levels])
        axes[0].errorbar(x,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o-',capsize=4,label=arm.upper(),color=color)
        fa=[100*results['noisy'][s]['metrics'][arm]['clean_source_false_alarm_rate'] for s in levels]
        axes[1].plot(x,fa,'o-',label=arm.upper(),color=color)
    for ax in axes:
        ax.set_xlabel('Width / relief variability (%)'); ax.set_xticks(x); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylim(0,1); axes[0].set_ylabel('Box F1'); axes[0].set_title('Localization, IoU >= 0.5')
    axes[1].set_ylim(0,8); axes[1].set_ylabel('Clean-weld crops flagged (%)'); axes[1].set_title('False alarms on clean specimens')
    fig.suptitle('Independent synthetic confirmation: 30 backgrounds',fontsize=14)
    fig.text(.5,.035,'Shared backgrounds across variability levels. F1 bars: 95% background-cluster bootstrap intervals.\n'
             'Noise amplitudes are assumptions. These are not measured BSE or manufacturing acceptance results.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.97,top=.82,bottom=.24,wspace=.3)
    fig.savefig(FIG/'03_confirmation_performance.png',dpi=170); plt.close(fig)
    print('Delivery audit and report written.',flush=True)


def verify_circle():
    from box_refinement.methods import predicted_regions
    from inspection.engine import FrozenInspector
    import torch
    torch.set_num_threads(4)
    inspector=FrozenInspector(CAMPAIGN)
    models=c.read(CAMPAIGN/'frozen_models.json')['models']
    checked=0; errors={}; score_errors=[]
    legacy=ROOT/'DataSimulation/data/output/circular_examples_20260928_v1'
    for name in ('a00_clean','a05_clean','a10_clean','a10_keyhole_pore','a10_intrinsic_pore'):
        meta=c.read(DATA/f'{name}.json'); oldmeta=c.read(legacy/f'{name}.json')
        assert c.sha(DATA/f'{name}.npz')==meta['npz_sha256']
        assert c.sha(DATA/f'{name}.scores.npz')==meta['scores_sha256']
        assert c.sha(legacy/f'{name}.npz')==oldmeta['npz_sha256']
        assert meta['renderer']['acquisition']['version']=='raster_v2'
        assert meta['renderer']['operating_domain']['incidence_deg_quantiles'][-1]<=60
        with np.load(DATA/f'{name}.npz') as new,np.load(legacy/f'{name}.npz') as old:
            errors[name]={k:float(np.max(np.abs(new[k].astype(float)-old[k]))) for k in ('relief','mask','crops_mask')}
            assert max(errors[name].values())==0
            norm=inspector.norm
            x=np.sqrt(np.clip((new['crops_raw']-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1)).astype(np.float32)
        with np.load(DATA/f'{name}.scores.npz') as scores:
            for arm in ('cnn','median'):
                for score,row in zip(scores[arm+'_scores'],meta['predictions'][arm]):
                    _,boxes=predicted_regions(score,**models[arm]['parameters'])
                    assert boxes==row['boxes_xyxy']; checked+=1
                for sector in (0,25,77):
                    item=dict(target=x[sector],context=x[[(sector+k)%78 for k in (-2,-1,1,2)]])
                    score,_,_=inspector.predict(item,arm)
                    error=float(np.max(np.abs(score-scores[arm+'_scores'][sector])))
                    assert error<1e-6,error; score_errors.append(error)
    result=dict(status='passed',old_and_new_geometry_max_errors=errors,saved_decisions_replayed=checked,
                fresh_inference_replays=len(score_errors),maximum_score_replay_error=max(score_errors),
                all_local_incidence_within_60_degrees=True)
    c.write(DATA/'independent_verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['visuals','delivery','verify_circle','all'],default='all'); args=p.parse_args()
    if args.stage in ('visuals','all'): visuals()
    if args.stage in ('delivery','all'): delivery()
    if args.stage in ('verify_circle','all'): verify_circle()
