"""Measured synthetic outcomes and explanatory figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .campaign import read,write,sha,require,verify_sources,seal,load,ARMS,KINDS,REPO


def build(root):
    verify_sources(root);require(root,'evaluate');require(root,'pilot')
    results=read(root/'results.json');pilot=read(root/'pilot_measurements.json');b=read(root/'constant_b_evidence.json')
    levels=[0.,.05,.1,.2];colors={'cnn':'#1764a2','median':'#bd5729','thresholding':'#677176'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,key,title in zip(axes,['box_f1','box_recall','clean_crop_false_alarm_rate'],['Box F1','Defect box recall','False-alarm rate on negative crops']):
        for arm in ARMS:
            yy=[results['noisy'][str(l)]['metrics'][arm][key] for l in levels]
            ax.plot(np.array(levels)*100,yy,'o-',label=arm,color=colors[arm])
            ci=np.array([results['noisy'][str(l)]['uncertainty']['intervals'][arm][key] for l in levels])
            ax.fill_between(np.array(levels)*100,ci[:,0],ci[:,1],color=colors[arm],alpha=.12)
        ax.set(xlabel='Width/depth variation bound (%)',ylabel=title,ylim=(0,1));ax.grid(alpha=.2);ax.axvline(10,color='gray',ls=':',lw=1)
    axes[0].legend();fig.suptitle('Frozen models on 30 fresh paired synthetic weld backgrounds; shaded 95% cluster intervals')
    fig.savefig(root/'robustness.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for label,ax in zip(['noisy','quiet'],axes):
        for arm in ARMS:
            ax.plot(np.array(levels)*100,[results[label][str(l)]['metrics'][arm]['box_f1'] for l in levels],'o-',label=arm,color=colors[arm])
        ax.set(title=label.capitalize(),xlabel='Width/depth variation bound (%)',ylabel='Box F1',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(root/'noise_comparison.png',dpi=180);plt.close(fig)
    # Fixed scene selection by index, not by attractive model performance.
    records=read(root/'datasets/confirmation/index.json')
    fig,axes=plt.subplots(3,4,figsize=(12,6),layout='constrained')
    for i,kind in enumerate(KINDS):
        for j,level in enumerate(levels):
            row=next(r for r in records if r['background_id']=='confirmation-000' and r['source_class']==kind and r['amplitude']==level)
            with np.load(row['npz_path']) as a:
                crop=a['raw'][:,128:192];mask=a['mask'][:,128:192]
            norm=read(root/'normalization.json')
            image=np.sqrt(np.clip((crop-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1))
            ax=axes[i,j];ax.imshow(image,cmap='gray',vmin=0,vmax=1)
            if mask.any():ax.contour(mask,levels=[.5],colors=['#ec6558'],linewidths=.8)
            ax.set_title(f'{kind}, {level:.0%}');ax.axis('off')
    fig.suptitle('Fixed example background and target position; shared training normalization; red = geometric label')
    fig.savefig(root/'examples.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4),layout='constrained')
    for method in ('finite_difference','cubic_spline'):
        rr=[r for r in pilot['analytic_gradient_rows'] if r['surface']=='smooth_wave' and r['method']==method]
        ax.loglog([r['spacing_mm'] for r in rr],[r['interior_rmse'] for r in rr],'o-',label=method)
    ax.set(xlabel='Pixel spacing (mm)',ylabel='Interior gradient RMSE',title='Known smooth surface: derivative discretization error');ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'spline_gradient_error.png',dpi=180);plt.close(fig)
    lines=['# Paired weld variability experiment','',
        'This experiment measures computer-vision detection of surface-expressed pores under declared synthetic weld variation. It does not validate real SEM images or manufacturing acceptance limits.','',
        '## Completed experiment','',
        '- Development: 100 independent latent backgrounds, each rendered as clean/keyhole/intrinsic; 300 scenes and 1,500 target windows. Split by background: 60/20/20.',
        '- Three target-only CNN fits, twenty epochs each. CNN and classical settings selected on validation only, then frozen before confirmation generation.',
        '- Confirmation: 30 fresh backgrounds, three scene types and four variation levels: 360 scenes, 3,240 frames, 1,800 target windows. Related variants are paired, not independent observations.',
        '- Baseline surface width and relief amplitude vary smoothly; pore geometry and mask remain unchanged. No image-contrast eligibility gate.',
        '- Noise-free signals are captured before contamination/Poisson/thermal on the same rendering. Fixed detector sensitivity mismatch is retained.',
        '- All image inputs are float32 with the same training-only normalization. No 8-bit evaluation path.',
        '- Variability levels 0/5/10 percent occur in development; 20 percent is an untrained stress level.','',
        '## Noisy confirmation measurements','',
        '| Variation | Method | Box F1 (95% CI) | Box recall | Pixel IoU | Negative-crop false alarms | Clean-source false alarm rate |',
        '|---:|---|---|---:|---:|---|---:|']
    for level in levels:
        group=results['noisy'][str(level)]
        for arm in ARMS:
            m=group['metrics'][arm];ci=group['uncertainty']['intervals'][arm]['box_f1']
            lines.append(f'| {level:.0%} | {arm} | {m["box_f1"]:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {m["box_recall"]:.3f} | {m["pixel_iou"]:.3f} | {m["clean_false_alarms"]}/{m["clean_targets"]} | {m["clean_source_false_alarm_rate"]:.3f} |')
    lines += ['', 'Confidence intervals resample complete latent backgrounds, retaining every paired class and crop. These are exploratory intervals, not independent-crop intervals. A zero level means no added width/depth warp; the original stochastic weld texture remains.', '',
        '## Matched noise-free measurements','', '| Variation | Threshold F1 | Median F1 | CNN F1 |','|---:|---:|---:|---:|']
    for level in levels:
        m=results['quiet'][str(level)]['metrics']
        lines.append(f'| {level:.0%} | {m["thresholding"]["box_f1"]:.3f} | {m["median"]["box_f1"]:.3f} | {m["cnn"]["box_f1"]:.3f} |')
    lines += ['', 'The same checkpoints and decision thresholds are applied to noisy and quiet data. Noise-free performance is a diagnostic, not a separately tuned detector.', '',
        '## Cubic-spline comparison','',
        'The alternative fits interpolating cubic splines in the physical x/y coordinates and differentiates at the original knots. Boundary condition: not-a-knot. No smoothing is applied. In this tensor-product construction, row/column derivatives at sample knots agree with partial derivatives of the tensor-product interpolant.', '',
        '| Pilot scene | Mean normal difference (degrees) | 99th percentile (degrees) | Quiet TOPO RMS difference | Existing noise RMS |',
        '|---|---:|---:|---:|---:|']
    for r in pilot['paired_render_rows']:
        lines.append(f'| {r["background"]}: {r["kind"]} | {r["normal_angle_mean_deg"]:.4f} | {r["normal_angle_p99_deg"]:.4f} | {r["topo_rms_difference"]:.6g} | {r["finite_difference_noise_rms"]:.6g} |')
    lines += ['', 'Spline derivatives have a known reference only on the analytic plane/quadratic/wave tests. Differences on procedural welds show numerical sensitivity, not which method better describes a measured weld. Exact interpolation can amplify sharp sampling transitions; smoothness does not certify physical accuracy. All paired gradient masks were identical.', '',
        f'Noiseless-capture maximum replay error: {pilot["quiet_capture_max_error"]:.6g}. Zero-variation cropped/full-frame maximum error: {pilot["zero_variation_strip_max_error"]:.6g}.', '',
        '## Beam-angle interpretation','',
        'Let b point along incoming electron travel, and n be the outward normal. Local incidence is alpha = acos(-b dot n). Current b=(0,0,-1), so alpha=acos(n_z). Width/depth variation changes n and therefore changes local incidence without independent beam deflection.', '',
        'Independent beam variation needs a normalized b input, tangent t1=(b-(b dot n)n)/norm(...), t2=n cross t1, consistent detector projections, incidence-dependent yield, recorded angle parameters, and flat-plane/rotation tests. A tilted specimen also changes projection and potentially visibility. Merely changing slice_tilt_deg changes geometry-sampling/calibration settings and is not a complete beam-rotation model. This campaign does not implement independent beam jitter.', '',
        '## Constant B: statistical support and limits','',f'B = {b["B"]:.12f}; no coefficient was refitted. Recomputed {b["selected_exits"]} cached exits across 40 historical intermediate-angle runs.', '',
        '| Comparison | Mean gain (nats/exit) | Descriptive run-level 95% interval | Tilt-cluster bootstrap interval |', '|---|---:|---|---|']
    for name,r in b['comparisons'].items():
        lo,hi=r['run_t_interval'];blo,bhi=r['tilt_cluster_bootstrap_interval']
        lines.append(f'| {name} | {r["mean_gain_nats_per_exit"]:.6f} | [{lo:.6f}, {hi:.6f}] | [{blo:.6f}, {bhi:.6f}] |')
    lines += ['',b['conclusion'],'','The scalar-B law keeps the polar marginal p(mu)=2mu and E[mu^2]=0.5. The saved 60-degree data contradict that marginal. A likelihood advantage over B=0 and lack of a demonstrated sine-law advantage support a pragmatic baseline; they do not prove B is a universal constant or establish full angular adequacy.', '',
        '## Files and reproduction','',
        '`protocol.json` and `source_snapshot.zip` freeze methods before primary generation/training. `datasets/*/index.json` records every scene, seed, physical metadata and file digest. Each scene NPZ contains raw/quiet images, geometric mask, relief and the realised width/depth fields. `normalization.json` records training-only preprocessing. `frozen_models.json` identifies selected settings/checkpoints. `model_candidates.json` includes all three CNN seeds. `results.json` includes pore-type/diameter strata and the reserved development test. `predictions/` stores all evaluated boxes.', '',
        '```powershell',r'.\.venv\Scripts\python.exe .\WeldDefectDetection\run_variability_campaign.py --stage all','```','',
        'Existing frozen campaigns are retained. Resume verifies stage receipts and frozen source hashes. No real-image or physical weld measurements were acquired.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    seal(root,'report',[root/'REPORT.md',*root.glob('*.png')])
    inventory={str(p.resolve()):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='FINAL_MANIFEST.json'}
    write(root/'FINAL_MANIFEST.json',dict(files=inventory,physical_validation=False,scope='All completed campaign artifacts'))
