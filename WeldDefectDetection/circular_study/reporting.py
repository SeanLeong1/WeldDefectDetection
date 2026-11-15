"""Reproducible figures, layer teaching, results and portable delivery."""
from pathlib import Path
import gc,json,zipfile,textwrap
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from box_comparison.core import sha
from . import generation as g,metrics,median,training,evaluation

FIG=g.ROOT/'docs/figures/circular_study_20261004'
REPORT=g.ROOT/'docs/CIRCULAR_STUDY_RESULTS_20261004.md'
GUIDE=g.ROOT/'docs/CIRCULAR_STUDY_METHODS_20261004.md'


def make_figures(frozen,results):
    FIG.mkdir(parents=True,exist_ok=True)
    data=g.load('test',frozen['cnn']['config']['channels'])
    selected=frozen['cnn'];model=training.load_model(selected)
    # Fixed first keyhole example, never selected for an attractive prediction.
    ring=1;mask=data['mask'][ring];columns=np.flatnonzero(mask.any(0));centre=int(np.median(columns))
    x,y=training.extract(data,[(ring,centre)],selected['config']['input_size'],selected['config']['channels'])
    features={};layers=[];handles=[]
    for name,module in model.named_modules():
        if not list(module.children()):
            def hook(mod,args,out,name=name):
                layers.append(dict(name=name,type=type(mod).__name__,input=list(args[0].shape),output=list(out.shape),parameters=sum(p.numel() for p in mod.parameters(recurse=False))))
            handles.append(module.register_forward_hook(hook))
        if name in ('enc1','enc2','bottleneck','dec2','dec1','head'):
            def hook_feature(mod,args,out,name=name):features[name]=out.detach().numpy()[0]
            handles.append(module.register_forward_hook(hook_feature))
    with torch.inference_mode():model(torch.from_numpy(x))
    for handle in handles:handle.remove()
    g.write(g.OUT/'cnn_layer_audit.json',dict(layers=layers,config=selected['config'],parameters=selected['parameters'],
        checkpoint_sha256=selected['checkpoint_sha256'],example_scene=data['meta'][ring]['scene_id']))
    fig,axes=plt.subplots(1,6,figsize=(15,3),layout='constrained')
    for ax,(name,a) in zip(axes,features.items()):
        shown=np.mean(abs(a),axis=0) if name!='head' else 1/(1+np.exp(-a[0]))
        ax.imshow(shown,cmap='viridis');ax.set_title(f'{name}\n{a.shape[0]} x {a.shape[1]} x {a.shape[2]}');ax.axis('off')
    fig.suptitle('Actual selected CNN features; each feature panel scaled separately, not physical quantities')
    fig.savefig(FIG/'cnn_layers.png',dpi=170);plt.close(fig)
    cnnscore=training.predict({k:(v[:3] if k!='truth' else v[:3]) for k,v in data.items()},model,selected['config'])
    medscore=median.scores({k:v[:3] for k,v in data.items()},frozen['median']['spec'])
    fig,axes=plt.subplots(3,4,figsize=(12,8),layout='constrained')
    median_display_max=2*frozen['median']['selection']['best']['threshold']
    for i,kind in enumerate(g.baseline.c.KINDS):
        images=[median.tile(data['x'][i,0,16:80],centre),median.tile(data['mask'][i],centre),
                median.tile(cnnscore[i],centre),median.tile(medscore[i],centre)]
        for j,(image,title) in enumerate(zip(images,['TOPO','Geometric mask','CNN score (0 to 1)',f'Median score (0 to {median_display_max:g})'])):
            axes[i,j].imshow(image,cmap='gray' if j<2 else 'magma',vmin=0,vmax=1 if j<3 else median_display_max)
            axes[i,j].set_title(kind+'\n'+title);axes[i,j].axis('off')
    fig.suptitle('Fixed paired background test-000, 3.2 mm windows; shared display scales within each column')
    fig.savefig(FIG/'paired_examples.png',dpi=170);plt.close(fig)
    image=data['x'][ring,0,16:80];target=median.tile(image,centre)
    offsets=(-256,-192,-128,-64,64,128,192,256) if frozen['median']['variant']=='neighbours8' else (-128,-64,64,128)
    context=np.stack([median.tile(image,centre+k) for k in offsets])
    reference=median.aligned_reference(target,context);res=target-reference
    fig,axes=plt.subplots(1,5,figsize=(13,3),layout='constrained')
    for ax,im,title in zip(axes,[target,np.median(context,axis=0),reference,res,median.tile(medscore[ring],centre)],
            ['Target',f'{len(offsets)} neighbours: median','Profile-adjusted reference','Target minus reference','Calibrated score']):
        ax.imshow(im,cmap='viridis');ax.set_title(title);ax.axis('off')
    fig.suptitle('Neighbouring-median data flow for the selected neighbour count; residual smoothing precedes calibration')
    fig.savefig(FIG/'median_steps.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(14,5),layout='constrained')
    for i,ax in enumerate(axes):
        ax.imshow(data['x'][i,0,16:80],cmap='gray',vmin=0,vmax=1,aspect='auto',extent=[0,360,1.6,-1.6])
        ax.set_ylabel(g.baseline.c.KINDS[i]+'\nradial mm')
    axes[-1].set_xlabel('Angle around the continuous ring (degrees)');fig.suptitle('One complete circular background, three paired scenes')
    fig.savefig(FIG/'continuous_ring.png',dpi=170);plt.close(fig)
    del data,model;gc.collect()
    base=max([s for s in frozen['cnn_candidates'] if s['variant']=='fixed_negatives'],key=lambda s:metrics.key(s['selection']['best']['metrics']))
    selected_names=[results['selected_cnn']+'_best',f'cnn_fixed_negatives_{base["seed"]}_best',
                    results['selected_median']+'_best','median_legacy_scale_best']
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    labels=['Selected CNN','Fixed-negative CNN','Selected median','Baseline median']
    for ax,key,title in zip(axes,['f1','recall','clean_ring_fa'],['Box F1','Recall at box IoU >= 0.5','Fraction of clean rings flagged']):
        values=[results['results'][k]['metrics'][key] for k in selected_names]
        ax.bar(range(4),values,color=['#3568a8','#dc7c32','#91b7da','#e7b88c']);ax.set_ylim(0,1)
        ax.set_xticks(range(4),labels,rotation=30,ha='right');ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v+.02,f'{v:.3f}',ha='center')
    fig.suptitle('Controlled comparison: same training population and 30 held-out backgrounds')
    fig.savefig(FIG/'confirmation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for label,spec,name,color in [('CNN',frozen['cnn'],results['selected_cnn'],'#3568a8'),('Median',frozen['median'],results['selected_median'],'#dc7c32')]:
        trials=spec['selection']['trials']
        axes[0].scatter([t['metrics']['clean_ring_fa'] for t in trials],[t['metrics']['recall'] for t in trials],alpha=.4,s=20,color=color,label=label)
        for point,marker in [('best','o'),('low_fa','s')]:
            result=results['results'].get(name+'_'+point)
            if result and result.get('available'):
                m=result['metrics'];axes[1].scatter(m['clean_ring_fa'],m['recall'],marker=marker,s=90,color=color,label=label+' '+point)
    for ax,title in zip(axes,['Validation grid: selection evidence','New test: frozen operating points']):
        ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Fraction of clean rings flagged',ylabel='Region recall',title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(FIG/'alarm_recall.png',dpi=170);plt.close(fig)
    with np.load(g.OUT/'diagnostics/component_arrays.npz') as a:
        fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
        for j,key in enumerate(['recessed','closed_fusion','lambertian_B0']):
            axes[0,j].imshow(a[key+'_height'] if key+'_height' in a else a['recessed_height'],cmap='viridis');axes[0,j].set_title(key+' height')
            axes[1,j].imshow(a[key],cmap='gray');axes[1,j].set_title(key+' TOPO')
            for ax in axes[:,j]:ax.axis('off')
        fig.suptitle('Assumption sensitivity: identical physical patch; panels independently scaled')
        fig.savefig(FIG/'assumptions.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axis=(np.arange(64)-31.5)*.05
        for name in ('recessed','closed_fusion'):
            axes[0].plot(axis,1000*a[name+'_height'][32],label=name)
            axes[1].plot(axis,a[name][32],label=name)
        axes[0].set_ylabel('Surface relief (micrometres)');axes[1].set_ylabel('Quiet TOPO (relative signal)')
        for ax in axes:ax.set_xlabel('Local radial coordinate (mm)');ax.legend();ax.grid(alpha=.2)
        fig.suptitle('An assumed weld shape becomes a slope-dependent detector response')
        fig.savefig(FIG/'weld_profile.png',dpi=170);plt.close(fig)
    error_path=g.OUT/'diagnostics/error_taxonomy.json'
    if error_path.exists():
        examples=list(g.read(error_path)['examples'].items())[:6]
        if examples:
            fig,axes=plt.subplots(len(examples),3,figsize=(10,3*len(examples)),squeeze=False,layout='constrained')
            for row,(name,example) in enumerate(examples):
                for col,key in enumerate(('target','mask','score')):
                    axes[row,col].imshow(example[key],cmap='gray' if col<2 else 'magma')
                    axes[row,col].axis('off');axes[row,col].set_title(key)
                axes[row,0].set_title(textwrap.fill(name.replace('_',' '),38)+'\n'+example['scene_id'],fontsize=9)
            fig.suptitle('First frozen-prediction example per observed failure category; score panels independently scaled')
            fig.savefig(FIG/'failure_examples.png',dpi=150);plt.close(fig)


def table_line(name,result):
    if not result.get('available',True):return f'| {name} | unavailable | | | | | |'
    m=result['metrics']
    return f'| {name} | {m["tp"]}/{m["fp"]}/{m["fn"]} | {m["f1"]:.3f} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["pixel_iou"]:.3f} | {m["clean_rings_flagged"]}/{m["clean_rings"]} |'


def run():
    frozen=evaluation.verify_freeze();results=g.read(g.OUT/'results.json');checks=g.read(g.OUT/'diagnostics/component_checks.json')
    make_figures(frozen,results)
    report=['# Circular TOPO thesis study, 4 October 2026','',
        'A controlled synthetic study of continuous circular weld inspection, restricted to neighbouring median and a small U-Net-style CNN. The generator, train/validation/test splits, model choices and evidence are versioned.',
        '', '## What was completed','',
        '- A periodic surface model, finite fusion/HAZ transitions and locally tangent keyhole pores, retaining the existing pore-response constants.',
        '- Continuous 96 x 5024 annuli; central 64 rows evaluated. Stride-32 predictions use surrounding context and are stitched before periodic connected components.',
        '- 60 training, 20 validation and 30 new test backgrounds, each with clean/keyhole/intrinsic variants: 330 scenes. Angles cover all quadrants in every split. No weak-signal filtering.',
        '- Six median variants and eight CNN variants screened on validation. Baseline and selected CNN variant repeated with three seeds. New test generation follows model freeze.',
        '- Both validation-F1 and validation clean-ring-alarm-constrained operating points. Box, pixel, class-specific and ring-level results; background-clustered uncertainty.',
        '', '![Native circular TOPO views](figures/circular_study_20261004/native_circles.png)' if (FIG/'native_circles.png').exists() else '',
        '', '## Scope and interpretation','',
        'The weld is a procedural height field: a recessed trace, crown/shoulders, toe transitions and textured variation around a circle. The pore changes height, yield and void fraction. The resulting TOPO is detector contrast, not height or depth. The alternative closed-fusion profile is an assumption sensitivity, not a measured reconstruction. Both CV methods output binary regions; pore-family results use simulator metadata and do not imply a learned keyhole/intrinsic classifier.',
        '', 'The main campaign retains the previous explicit assumed constant B for comparability with the circular baseline. It does not reinstate that coefficient as a validated CASINO fit or change the project-wide Lambertian fallback. B=0 is assessed separately at component level.',
        '', 'The revised texture is a new cylindrical random-field distribution. It replaces the old Cartesian texture and collapse-event construction; this is a combined geometry change, not an isolated periodic-boundary ablation. No numerical score establishes metallurgical acceptance or transfer to measured SEM.',
        '', 'The old 12-ring test population is diagnostic history. All confirmation figures below use 30 fresh backgrounds. Earlier campaign F1 values cannot be subtracted from these to claim an improvement: geometry, population and counting units changed.',
        '', 'This is a pore-enriched benchmark: two defective scenes for each clean scene, with one injected pore per defective scene. Precision and F1 depend on that mixture and are not production-prevalence estimates. Clean-ring alarms and class-specific recall are reported separately.',
        '', 'Unwrapping uses the known simulator ring centre, radius and pixel scale. Automatic localisation of an unknown real ring and robustness to centre/radius errors are outside this experiment. The clean label means no additional pore was injected; it carries no manufacturing-acceptance judgement.',
        '', '## Numerical and integrity gates','',
        'Automated checks cover periodic value/slope continuity, grid-independent physical surface samples, periodic box matching, post-merge area filtering, disjoint median context, gap-free stitching and split angle/size/offset coverage.',
        '', 'Native full-frame and padded-annulus quiet signals agree to the saved tolerance for all three classes on one fixed diagnostic background. A blur-support guard is checked for every generated scene. Only the annulus is valid exported image data. Noise realizations can differ because the uncomputed exterior changes the Poisson draw sequence.',
        '', '| Ring orientation | 0.05 mm relative L2 error | 0.025 mm relative L2 error |','|---|---:|---:|']
    for angle in (0.,45.,90.):
        rr=[r for r in checks['convergence'] if abs(r['angle_deg']-angle)<1e-6]
        report.append(f'| {angle:.0f} degrees | {rr[0]["relative_l2"]:.2%} | {rr[1]["relative_l2"]:.2%} |')
    report+=['','Reference: 0.0125 mm geometry/normals and native detector quadrature, with a fixed 0.05 mm Gaussian PSF and common area-integrated output. This checks three clean component patches, not the complete acquisition chain or absolute convergence.']
    pore_path=g.OUT/'diagnostics/pore_convergence.json'
    if pore_path.exists():
        pores=g.read(pore_path)['rows']
        report+=['','The smallest-pore component test compares the paired pore-minus-clean signal itself, using 0.5 mm diameter and 0.25 mm depth parameters at three orientations. It retains native pore occupancy/response constants with fixed surface planes and unit detector gains.',
            '', '| Pore family | 0.05 mm relative signal error range | 0.025 mm relative signal error range |','|---|---:|---:|']
        for kind in g.baseline.c.KINDS[1:]:
            aa=[r['relative_delta_l2'] for r in pores if r['kind']==kind and r['pitch_mm']==.05]
            bb=[r['relative_delta_l2'] for r in pores if r['kind']==kind and r['pitch_mm']==.025]
            report.append(f'| {kind} | {min(aa):.2%} to {max(aa):.2%} | {min(bb):.2%} to {max(bb):.2%} |')
        report+=['','Small pore signals have greater sampling sensitivity than the clean background. The 0.05 mm campaign is a declared computational-resolution benchmark, not a claim that every defect response is converged. Finer uniform rendering with common output integration is the next numerical improvement before quantitative physical-response claims.']
    report+=['', '## Held-out confirmation','',
        f'Selected CNN variant: **{frozen["cnn_variant"]}**, seed {frozen["cnn"]["seed"]}. Selected median: **{frozen["median"]["variant"]}**. Selection uses validation only.',
        '', '| Method, validation-F1 operating point | TP/FP/FN | Box F1 | Precision | Recall | Pixel IoU | Clean rings flagged |','|---|---:|---:|---:|---:|---:|---:|']
    names=[results['selected_cnn'],results['selected_median'],'historical_circular_cnn','historical_circular_median']
    for name in names:report.append(table_line(name,results['results'][name+'_best']))
    report+=['','Previous-model rows are frozen transfer using each old model\'s own normalization. Their predictions use the new complete-ring counting pipeline. They are domain-transfer tests, not a pure architecture comparison.',
        '', '![Confirmation](figures/circular_study_20261004/confirmation.png)','',
        '![Alarm and recall](figures/circular_study_20261004/alarm_recall.png)','',
        '| Method, validation alarm-constrained operating point | TP/FP/FN | Box F1 | Precision | Recall | Pixel IoU | Clean rings flagged |','|---|---:|---:|---:|---:|---:|---:|']
    for name in names[:2]:report.append(table_line(name,results['results'][name+'_low_fa']))
    report+=['','The alarm target is at most 10% of validation clean rings. The displayed test alarm rates are outcomes, not guarantees and not exactly matched across methods. With 30 clean test rings, one ring changes the estimate by 3.3 percentage points. A zero count would still have an approximately 9.5% one-sided 95% binomial upper bound.',
        '', '| Selected method | Keyhole TP / labelled regions | Intrinsic TP / labelled regions | F1 95% cluster interval |','|---|---:|---:|---|']
    for name in names[:2]:
        r=results['results'][name+'_best'];k=r['per_class']['keyhole_pore'];i=r['per_class']['intrinsic_pore'];ci=r['intervals']['f1']
        report.append(f'| {name} | {k["tp"]}/{k["tp"]+k["fn"]} | {i["tp"]}/{i["tp"]+i["fn"]} | {ci[0]:.3f} to {ci[1]:.3f} |')
    report+=['','Each bootstrap draw resamples whole backgrounds with all three class variants together. Overlapping tiles are not independent evidence.',
        '', '## Controlled CV comparisons','',
        'Each CNN alternative starts from rotating-negative training and changes one factor. Fixed-negative training is the reference. Hard-negative mining uses training windows only; augmentation adds paired shifts/flips and mild intensity changes; larger context uses 96 input with central64 supervision; focal changes the BCE factor; capacity changes base channels16 to24; the optional multichannel arm adds Tx/Ty.',
        '', '| CNN variant | Seed | Validation F1 | Test F1 (replicated arms and auxiliary arm) | Test clean rings flagged |','|---|---:|---:|---:|---:|']
    for c in frozen['cnn_candidates']:
        key=f'cnn_{c["variant"]}_{c["seed"]}_best';r=results['results'].get(key)
        report.append(f'| {c["variant"]} | {c["seed"]} | {c["selection"]["best"]["metrics"]["f1"]:.3f} | '+(f'{r["metrics"]["f1"]:.3f} | {r["metrics"]["clean_rings_flagged"]}/30 |' if r else 'not opened | not opened |'))
    paired_path=g.OUT/'paired_comparisons.json'
    if paired_path.exists():
        paired=g.read(paired_path)
        report+=['','| Frozen comparison | Test F1 difference | Paired-background 95% interval |','|---|---:|---|']
        for name,comparison in paired['comparisons'].items():
            m=comparison['metrics']['f1'];low,high=m['interval']
            report.append(f'| {name} | {m["difference"]:+.3f} | {low:+.3f} to {high:+.3f} |')
        report+=['','These descriptive intervals use 5000 paired background resamples and are conditional on the frozen models. An interval spanning zero does not resolve the sign of the difference. They do not include seed variability or establish measured-weld transfer; they are not multiplicity-adjusted hypothesis tests.',
            '', 'The selected capacity24 versus fixed-negative contrast combines rotating negative sampling and increased channel capacity. It cannot identify a capacity-only effect. The seed-1 validation comparison against rotating_negatives isolates the capacity change within this screening design.']
    report+=['','Unselected TOPO-only CNN screening arms remain validation-only to limit confirmation comparisons. The TOPO+Tx+Ty arm is a separate secondary information comparison and is ineligible to become the headline TOPO-only model. Architecture choice and the best seed are selected using validation; the seed spread, not only the selected seed, matters.',
        '', '| Median variant | Validation F1 | Test F1 | Test clean rings flagged |','|---|---:|---:|---:|']
    for m in frozen['median_candidates']:
        r=results['results']['median_'+m['variant']+'_best']['metrics']
        report.append(f'| {m["variant"]} | {m["selection"]["best"]["metrics"]["f1"]:.3f} | {r["f1"]:.3f} | {r["clean_rings_flagged"]}/30 |')
    report+=['','Median alternatives subtract a training residual centre, align neighbour profile centre/width, condition robust scale on training intensity quartiles, use eight disjoint neighbours, or adapt scale from neighbour dispersion within fixed bounds. The baseline is the old score construction refitted on this population. These are experimental variants, not presumed improvements.',
        '', '## Visibility and noise','',
        'Every pore has diameter, nominal pixels, mask area, angle, radial offset, paired quiet contrast and a noise proxy saved in diagnostics/visibility.json. Primary TOPO-only quiet inference is a frozen sensitivity test using the same normalizer, checkpoint and thresholds. The auxiliary signed-channel arm is a separate noisy-data comparison. Quiet and noisy channels are never mixed.',
        '', '| Quiet selected method, original operating point | TP/FP/FN | Box F1 | Precision | Recall | Pixel IoU | Clean rings flagged |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('quiet_selected_cnn','quiet_selected_median'):
        if name+'_best' in results['results']:report.append(table_line(name,results['results'][name+'_best']))
    report+=['', '![Paired examples](figures/circular_study_20261004/paired_examples.png)']
    error_path=g.OUT/'diagnostics/error_taxonomy.json'
    if error_path.exists():
        diagnostic=g.read(error_path)
        report+=['','## Where the frozen detector failed','',
            'These are observational categories, not proof of a physical cause. A low score within a geometric label may reflect model confidence, threshold choice or weak signal. A localization mismatch means a response did not meet the declared box criterion. No tuning follows this diagnosis.',
            '', '| Category | CNN scenes | Median scenes |','|---|---:|---:|']
        categories=sorted(set().union(*(set(r['counts']) for r in diagnostic['methods'].values())))
        for category in categories:
            report.append(f'| {category} | {diagnostic["methods"]["cnn"]["counts"].get(category,0)} | {diagnostic["methods"]["median"]["counts"].get(category,0)} |')
        report+=['','![Failure examples](figures/circular_study_20261004/failure_examples.png)']
    origin_path=g.OUT/'diagnostics/tile_origin_sensitivity.json'
    if origin_path.exists():
        diagnostic=g.read(origin_path)
        report+=['','## Dependence on the arbitrary tile origin','',
            'The first test background is shifted by 8,16 and24 angular columns, predictions are made, then the shift is reversed. These shifts preserve the pooling-grid phase. Physical pixels are identical, and labels are not scored. This tests tile-boundary sensitivity independently of detection F1.',
            '', '| Frozen seed-1 model | Worst mean absolute score change | Worst maximum absolute score change |','|---|---:|---:|']
        for variant in ('fixed_negatives','larger_context'):
            rr=[r for r in diagnostic['rows'] if r['variant']==variant]
            report.append(f'| {variant} | {max(r["mean_absolute_score_change"] for r in rr):.6g} | {max(r["maximum_absolute_score_change"] for r in rr):.6g} |')
    runtime_path=g.OUT/'runtime_benchmark.json'
    if runtime_path.exists():
        timings=g.read(runtime_path)
        report+=['','## Inference cost','',
            'Five warmed repetitions on the fixed first clean ring, four Torch/OpenCV CPU threads, excluding file/model loading and final region extraction. These are local software timings, not acquisition speed or a hardware deployment guarantee.',
            '', '| Selected method | Median seconds per whole-ring score map |','|---|---:|']
        for method,row in timings['timings'].items():report.append(f'| {method} | {row["median_seconds"]:.3f} |')
        report+=['','Training report `seconds` is invocation time; interrupted/resumed fits are not cumulative timing measurements.']
    report+=['', '## Process record and limits','',
        '- v2 was stopped before training because linear texture interpolation introduced slope discontinuities. Its partial data are excluded.',
        '- v3 was stopped before training because the validation angle-stratum cycle omitted one quadrant. v4 uses balanced index-modulo-four angle strata. The validated surface and the index-zero numerical diagnostic are unchanged.',
        '- cv_protocol_v2.json clarifies that the primary CNN must remain TOPO-only. This eligibility correction was recorded before auxiliary-arm validation and before test generation. Completed fits were retained only after verifying unchanged fitting-function hashes. The original CV protocol and source snapshot remain as history.',
        '- The published dataset is v4. No historical campaign or its frozen sources was edited.',
        '- Surface texture, process-to-shape equations, emission assumption, response constants, blur and noise remain engineering assumptions. The numerical checks do not turn them into measured evidence.',
        '- Real SEM/BSE images and representative height measurements are still required for external validation. Their absence is an evidence gap, not a task that can be completed with more synthetic data.',
        '', '## Reproduce and inspect','',
        'Use the repository .venv Python and WeldDefectDetection/run_circular_study_20261004.py. Stages: prepare, train_data, val_data, diagnostics, train, test_data, evaluate, report. The train stage creates normalization after development generation in the documented driver; see the start-here guide for exact commands. Existing source/scene/model digests are checked. Do not edit a frozen implementation and reuse its output directory.',
        '', f'Dataset and complete machine-readable results: `{g.OUT.relative_to(g.ROOT).as_posix()}`. Read dataset_manifest.json for portable relative paths. Read protocol.json and cv_protocol_v2.json before comparing scores. The methods guide explains the executed layers chronologically.',
        '', 'A credible thesis contribution here is the controlled analysis of geometry continuity, sampling, context and detector training under known synthetic conditions. A reported failed improvement or weak intrinsic-pore result is evidence to explain, not a reason to strengthen the injected signal.']
    REPORT.write_text('\n'.join(report)+'\n',encoding='utf-8')
    teach(frozen)
    dataset_readme=g.OUT/'README.md'
    dataset_readme.write_text('\n'.join([
        '# Continuous circular TOPO dataset v4','',
        '110 independent synthetic backgrounds: 60 train, 20 validation, 30 test. Each has clean, keyhole_pore and intrinsic_pore scenes. Keep all variants and tiles of a background in its declared split.',
        '', '## Load a scene','', '```python', 'import json', 'from pathlib import Path', 'import numpy as np',
        'root = Path(__file__).resolve().parent',
        "manifest = json.loads((root / 'dataset_manifest.json').read_text())",
        "record = manifest['records'][0]", "with np.load(root / record['path']) as scene:",
        "    topo = scene['raw'][0]  # 96 x 5024, raw detector contrast",
        "    tx, ty = scene['raw'][1], scene['raw'][2]",
        "    mask = scene['mask'][16:80]  # evaluated central 64 rows",
        "    quiet_topo = scene['quiet']  # paired diagnostic, not primary noisy input", '```','',
        'The snippet assumes it is saved in this dataset directory. Paths in the manifest are relative and portable. Metadata/checkpoint/source records preserve original absolute provenance paths; a relocated run must resolve those paths deliberately before replay.',
        '', 'Radius40 mm; radial sampling0.05 mm; angular arc pitch is recorded in the manifest. Angle starts at0 degrees and wraps after5024 columns. The extra radial rows support context. Pixels outside a plotted ring are not additional valid measurements.',
        '', 'Raw float signals are retained. Apply normalization.json for the trained model; do not apply a separate display-image normalization. Ground truth is the existing surface occupancy threshold, not an acceptance label.',
        '', 'Read protocol.json, cv_protocol_v2.json and frozen_models.json before comparing methods. results.json and predictions contain full-ring counts, not independent crop statistics. diagnostics records numerical, visibility and assumption tests.',
        '', 'The excluded preliminary v2/v3 attempts are not part of this manifest. No measured SEM/BSE validation is claimed.'
    ])+'\n',encoding='utf-8')
    source_zip=g.OUT/'analysis_sources.zip'
    source_paths=[*Path(__file__).parent.glob('*.py'),g.ROOT/'WeldDefectDetection/run_circular_study_20261004.py',
        g.ROOT/'WeldDefectDetection/tests/test_circular_study.py',REPORT,GUIDE,
        g.ROOT/'docs/CIRCULAR_STUDY_START_HERE_20261004.md',g.ROOT/'docs/CIRCULAR_THESIS_EVIDENCE_MAP_20261004.md']
    with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:archive.write(path,str(path.relative_to(g.ROOT)))
    files=[REPORT,GUIDE,dataset_readme,g.ROOT/'docs/CIRCULAR_STUDY_START_HERE_20261004.md',
        g.ROOT/'docs/CIRCULAR_THESIS_EVIDENCE_MAP_20261004.md',*FIG.glob('*.png'),g.OUT/'results.json',
        g.OUT/'dataset_manifest.json',g.OUT/'cnn_layer_audit.json',source_zip,g.OUT/'frozen_models.json',
        *list((g.OUT/'diagnostics').rglob('*.json')),
        *[g.OUT/f for f in ('runtime_benchmark.json','paired_comparisons.json','final_verification.json','cv_scope_amendment.json','freeze_observation.json') if (g.OUT/f).exists()]]
    g.write(g.OUT/'delivery_manifest.json',dict(files={str(p.resolve()):sha(p) for p in files}))
    print(REPORT,flush=True)


def teach(frozen):
    audit=g.read(g.OUT/'cnn_layer_audit.json');c=frozen['cnn']['config']
    lines=['# What the circular weld and CV methods actually do','',
        '## 1. Construct the weld before adding a defect','',
        'For a Cartesian point (x,y), the local radial coordinate is u = sqrt(x²+y²) - R and the angular coordinate is theta = atan2(y,x), with R = 40 mm. A clean weld is a prescribed height z(u,theta), not a physical welding simulation. Here clean means no additional pore was injected; it carries no manufacturing-acceptance judgement.',
        '', 'The recessed profile combines a narrow Gaussian depression, small crown/shoulders, finite toe transitions, periodic roughness and HAZ texture. Smooth angular fields vary width and depth by 0%, 5% or 10%. The texture is generated on a fixed physical cylinder, then interpolated at the requested points. This makes the same seed represent the same surface when output resolution changes. Cubic interpolation represents this assumed surface; it does not supply the physical shape or certify acceptance.',
        '', 'An injected pore supplies an occupancy o(x,y,z). The existing response decreases yield, increases void fraction and adds a negative surface-height perturbation. A geometric label is o > 0.08 in the inspected surface slab. Keyhole long axis and tail follow the ring tangent. Intrinsic pores use a compact local ellipse. These are controlled surrogate families, not recovered metallurgical pore shapes.',
        '', '## 2. Turn height into detector contrast','',
        'Finite differences estimate dz/dx and dz/dy in mm/mm. The unit normal is n = (-dz/dx, -dz/dy, 1) / sqrt(1+(dz/dx)²+(dz/dy)²). Detector quadrature uses this normal, material/yield/openness and the emission assumption to calculate four signals A,B,C,D. Blur and the raster acquisition model then act on those signals.',
        '', 'Tx = (A+D)-(B+C), Ty = (A+B)-(C+D), and TOPO = sqrt(Tx²+Ty²). A flat region can have little TOPO even when its absolute height is large. A steep edge can have strong TOPO. Therefore TOPO brightness cannot be read as pore depth. The main study retains an explicitly assumed constant emission B, with a separate B=0 sensitivity diagnostic.',
        '', '## 3. Prepare a continuous inspection image','',
        'The Cartesian image is sampled around all 360 degrees into 96 radial rows and 5024 angular columns. Rows16:80 are the evaluation band. Angular indices wrap. Images retain physical sampling metadata. Training-only quantiles set TOPO normalization, followed by a square-root tone curve. Tx/Ty use a separate signed scale in the optional three-channel experiment. Clipping rates are saved for every split.',
        '', '## 4. Neighbouring median, in order','',
        '1. Take a 64 x 64 target around the current angle. Take four disjoint neighbours at +/-64 and +/-128 columns. This is neighbouring spatial context, not temporal prediction.',
        '2. Estimate each neighbour\'s cross-weld intensity profile using its row median. Correct smooth profile differences and take the median across neighbours. This estimates what a normal local background looks like.',
        '3. Subtract the reference from the target and smooth the residual by one pixel. The residual is signed evidence of an unusual local response.',
        '4. Fit a residual centre c and robust scale s = 1.4826 median(|r-c|) using clean training rings. Smooth s and apply the declared 0.003 floor. The centred score is |r-c|/s; the legacy arm omits c. This is a relative-deviation score, not a calibrated probability or Gaussian significance test. Labels are used for training calibration, not supplied at inference.',
        '5. The alternatives adjust neighbour alignment, conditional scale, neighbour count or bounded context-only adaptation. They are compared separately on the same data.',
        '6. Keep each window\'s central32 columns and join the complete ring. Threshold, apply the selected closing, join connected components across the ring cut, then filter small regions. A seam-crossing region is counted once.',
        '', f'The selected median variant is **{frozen["median"]["variant"]}**. The neighbours8 variant extends the four-neighbour reference with windows at +/-192 and +/-256 columns, giving eight disjoint neighbours. Its exact scale maps, threshold and calibration population are in frozen_models.json.',
        '', '![Median steps](figures/circular_study_20261004/median_steps.png)','',
        '## 5. CNN inference, in order','',
        f'The selected network takes {c["channels"]} channel(s), {c["input_size"]} x {c["input_size"]} pixels, and has {audit["parameters"]:,} trainable parameters. The table below records actual executed layers and tensor shapes, rather than an assumed architecture diagram.',
        '', 'A convolution learns local filters by multiplying a small neighbourhood by learned weights and summing it. A 3 x 3 layer can respond to local edges, textures or combinations; assigning a particular physical meaning to a channel requires evidence. Batch normalization scales features using training statistics. ReLU keeps positive responses and sets negative responses to zero.',
        '', 'A convolution forms a weighted sum across its input channels and 3 x 3 spatial neighbourhood. Batch normalization applies gamma * (a - mean) / sqrt(variance + epsilon) + beta. During training, batch/spatial statistics update running estimates; inference uses those stored estimates. Gamma and beta are learned. ReLU is max(0,a). These operations repeat in each block; the table shows their channel counts and spatial sizes.',
        '', 'Two encoder blocks build local features. Max-pooling halves spatial dimensions and increases the physical context of later features. The bottleneck operates at one-quarter input width and height. Transposed convolutions increase resolution. Concatenating the corresponding encoder features supplies the decoder with detail retained before pooling. Decoder convolutions combine that detail with broader context.',
        '', 'A final 1 x 1 convolution produces one logit per pixel. Sigmoid converts logits to scores between 0 and 1; these are model scores, not automatically calibrated physical probabilities. The larger-context variant supervises and retains only the central 64 region of its 96 input. All variants contribute central 32 angular columns during whole-ring inference.',
        '', 'Increasing the input from 64 to 96 does not change kernel sizes or network depth. It mainly provides real surrounding pixels where a smaller window would rely on padding near retained-output edges. Increasing base channels from 16 to 24 is a separate capacity change: more feature channels and weights at the same spatial scales.',
        '', '| Executed layer | Type | Input tensor | Output tensor | Parameters |','|---|---|---|---|---:|']
    for row in audit['layers']:
        lines.append(f'| {row["name"]} | {row["type"]} | {row["input"]} | {row["output"]} | {row["parameters"]} |')
    lines+=['','Tensor order is batch, channels, radial pixels, angular pixels. Skip concatenations are functional operations between the named modules and therefore do not appear as separate leaf layers.',
        '', '![CNN features](figures/circular_study_20261004/cnn_layers.png)','',
        '## 6. CNN learning, in order','',
        '1. Build train-only positive and negative window populations. A batch contains examples from an epoch sample of600 positive and600 negative windows. Fixed-negative training reuses eight negative windows per scene; rotating-negative training redraws from the full negative population. Hard mining ranks additional known-negative training windows after epochs5,10,15.',
        '2. Run the image forward through all layers. Compare the central output with the geometric label. Weighted binary cross-entropy penalizes wrong pixel predictions; Dice loss also rewards overlap despite a small foreground. The focal alternative downweights already-easy pixel decisions with a squared factor.',
        '3. Backpropagation calculates each weight\'s contribution to the loss. AdamW updates weights; gradient clipping limits large updates and cosine scheduling lowers the learning rate. Repeat for20 epochs. Saved optimizer and random-generator state allow an interrupted training run to resume.',
        '4. At epoch 20, evaluate the complete validation rings. Choose threshold, minimum area and closing using validation only. Screen seven TOPO-only variants and one auxiliary signed-channel variant on seed 1; repeat the fixed-negative baseline and selected TOPO-only variant with two more seeds. Freeze models and their file hashes before generating the new test set.',
        '5. The test set is evaluated once under the frozen choices. A stronger-looking test result never causes retuning. A future change requires another version and another confirmation population.',
        '', 'For label y in {0,1} and sigmoid score p, the weighted pixel loss is -w*y*log(p) - (1-y)*log(1-p). The positive weight is min(20, sqrt((1-f)/f)), where f is half the mean foreground fraction among positive training windows, reflecting the balanced window sampler. Dice loss is 1 - (2*sum(p*y)+1)/(sum(p)+sum(y)+1). The implementation adds mean weighted BCE and Dice. The focal alternative multiplies each BCE term by (1-p_t)^2, where p_t is p for a foreground label and 1-p for background. It retains Dice and the same positive-weight rule.',
        '', '## 7. Interpret the output correctly','',
        'Both detectors output binary defect regions, not pore-type predictions. Keyhole/intrinsic breakdowns use known injected-class metadata. Bounding boxes are drawn around periodic connected regions. One-to-one matching requires box IoU >= 0.5. A loose box touching a pore can therefore be a false positive while the labelled pore is a false negative. Pixel IoU measures segmentation overlap separately. Clean-ring alarm rate asks whether any false region appeared anywhere on a clean ring; it is stricter than a small crop alarm rate.',
        '', 'The confidence intervals resample independent backgrounds, keeping each clean/pore family together. Thousands of overlapping windows do not create thousands of independent test welds. Synthetic detection accuracy, numerical convergence and measured physical validity are distinct questions.',
        '', '## 8. What measured evidence would add','',
        'Collect paired surface height maps and quadrant/BSE images of representative clean and defective circular welds, with pixel scale, detector geometry, voltage/current, focus and acquisition settings recorded. Split by physical specimen before calibration. Estimate profile/texture distributions on calibration specimens, freeze the simulator and CV models, then assess held-out specimens with independently reviewed labels. Start with image/contrast correspondence and false alarms; do not infer manufacturing acceptance from synthetic occupancy labels.']
    GUIDE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
