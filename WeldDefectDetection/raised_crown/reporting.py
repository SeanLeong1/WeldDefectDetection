"""Evidence-led report, figures and chronological method explanation."""
from pathlib import Path
import numpy as np
from . import campaign as c,rendering as r

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=c.ROOT/'docs/figures/raised_crown_results_20261005';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    previews={}
    for kind in ('open_keyhole','bulge','hidden'):
        with np.load(c.OUT/f'pilot/{kind}.npz') as a:previews[kind]=(a['height'].copy(),a['raw'][0].copy(),a['mask'].copy())
    high=float(np.quantile(np.concatenate([v[1].ravel()[::8] for v in previews.values()]),.995))
    for col,kind in enumerate(('open_keyhole','bulge','hidden')):
        height,raw,mask=previews[kind]
        cols=(np.arange(160)-80)%5024
        hmap=axes[0,col].imshow(height[:,cols],origin='lower',cmap='viridis',aspect='equal',extent=(-4,4,-4,4),vmin=0,vmax=3.1);axes[0,col].set_title(kind.replace('_',' '))
        tmap=axes[1,col].imshow(raw[:,cols],origin='lower',cmap='gray',aspect='equal',extent=(-4,4,-4,4),vmin=0,vmax=high)
        axes[2,col].imshow(mask[:,cols],origin='lower',cmap='gray',vmin=0,vmax=1,aspect='equal',extent=(-4,4,-4,4))
        for row in range(3):axes[row,col].set_xticks([-4,0,4]);axes[row,col].set_yticks([-4,0,4])
        axes[2,col].set_xlabel('Arc distance (mm)')
    for ax,label in zip(axes[:,0],('Surface height','Rendered TOPO','Visible surface label')):ax.set_ylabel(label)
    fig.colorbar(hmap,ax=axes[0,:],shrink=.7,label='Height (mm)')
    fig.colorbar(tmap,ax=axes[1,:],shrink=.7,label='TOPO (raw units)')
    fig.suptitle('Unwrapped segments of an 80 mm circular weld; vertical coordinate is radial offset (mm)\nBulge is an assumed displacement case; hidden pore has no surface label',fontsize=12)
    fig.savefig(folder/'surface_signal_labels.png',dpi=160);plt.close(fig)
    return folder

def report():
    from .quality import run as quality
    quality()
    from .diagnostics import run as diagnose
    diagnosis=diagnose()
    c.audit();results=c.read(c.OUT/'results.json');stress=c.read(c.OUT/'robustness_results.json');pilot=c.read(c.OUT/'pilot/gates.json')
    frozen=c.read(c.OUT/'frozen_models.json');audit=c.read(c.OUT/'audit.json');folder=figures()
    lines=['# Raised circular crown: results and limits, 5 October 2026','',
        'This is a separate synthetic study. It does not establish measured SEM realism, subsurface-pore sensitivity or manufacturing acceptance. Published electron-depth numbers were researched but not implemented.','',
        '## Geometry and signal model','',
        'The target crown is 3 mm above the original surface; the solid fusion region reaches 7 mm below it, with widths of 2.8 mm at the original surface and 1.2 mm at 5 mm depth. Crown footprint is provisionally 2.8 mm. Plate thickness is not required for the surface experiment.','',
        'The old assumed emission model rejects slopes above 60 degrees. This study uses the supported explicit uncalibrated Lambertian fallback, uniform alloy composition, geometric surface pore response and nonlocal ray visibility. Thus old-model transfer measures a joint distribution change. The paired recessed/raised diagnostic holds these new rendering assumptions fixed.','',
        'Open surface pores, prescribed volume-displacement bulges and enclosed pores without surface change are separate scenarios. The displacement identity is a mathematical assumption, not experimentally established weld mechanics.','',
        '![Surface, signal and labels](figures/raised_crown_results_20261005/surface_signal_labels.png)','',
        '## Data and selection','',
        f'Independent backgrounds by split: {audit["backgrounds"]}. Scene counts: {audit["counts"]}. Related conditions stay in the same split. Test and stress scenes were generated after model freeze.','',
        'The CNN architecture remains TinyDefectSegNet with base width 24, trained for 20 epochs on three seeds. The physical scored band widens to 6.4 mm at 0.05 mm sampling; overlapping 64-pixel tiles cover it without shrinking the weld. Median calibration uses clean training rings. Thresholds, component area and closing are selected on validation only.','',
        '## Confirmation','',
        '| Model | F1 | Precision | Recall | Clean rings flagged | F1 95% background interval |',
        '|---|---:|---:|---:|---:|---|']
    for name,item in results['results'].items():
        m=item['metrics'];ci=item['intervals']['f1']
        lines.append(f'| {name} | {m["f1"]:.3f} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["clean_rings_flagged"]}/{m["clean_rings"]} | [{ci[0]:.3f}, {ci[1]:.3f}] |')
    lines+=['','Zero alarms among 24 clean rings does not establish a zero population false-alarm rate. Under an independent identically sampled Bernoulli model, the one-sided 95% upper bound for 0/24 is 1 - 0.05^(1/24), approximately 11.7%. A bootstrap containing only observed zeros is degenerate and cannot express that uncertainty.','']
    lines+=['',f'Selected CNN: `{results["selected"]}`. Selection was made before confirmation; other seeds are replications, not candidates selected using test scores.','',
        'Historical transfer uses original weights, normalisation and operating points with an explicit radial tiling adapter. It is not a rerun of the original narrower coverage contract. Hidden controls are counted as negative surface truth and are also reported separately in the machine-readable results.','',
        'Clean and hidden/no-response controls share the same simulated surface signal. They are paired conditions, not additional independent negative backgrounds. The primary table includes all five declared conditions. For transparency, the following descriptive view excludes the duplicate hidden-control condition and retains the same frozen models and thresholds.','',
        '| Model | Four-condition F1 | Precision | Recall |','|---|---:|---:|---:|']
    for method,item in diagnosis.items():
        m=item['without_hidden_control']
        lines.append(f'| {method} | {m["f1"]:.3f} | {m["precision"]:.3f} | {m["recall"]:.3f} |')
    lines+=['','| Model | Surface scenario | True detections | False regions | Misses | Recall |','|---|---|---:|---:|---:|---:|']
    for method in (results['selected'],'median'):
        for kind in ('open_keyhole','open_intrinsic','bulge'):
            m=results['results'][method]['by_class'][kind]
            lines.append(f'| {method} | {kind} | {m["tp"]} | {m["fp"]} | {m["fn"]} | {m["recall"]:.3f} |')
    selection=c.OUT/'selection_diagnostic.json'
    if selection.exists():
        p=c.read(selection);worst=p['backgrounds'][0];outside=[k for k,v in p['features'].items() if v['outside_training_range']]
        lines+=['','### Validation-selection sensitivity','',
            f'The selected seed is {p["selected_seed"]}, with validation F1 {p["metrics"]["f1"]:.3f}. Replaying the frozen validation inference reproduces the selection metrics exactly. Background `{worst["background_id"]}` accounts for {worst["metrics"]["fp"]} of {p["metrics"]["fp"]} validation false regions. Its clean-image high clipping fraction is {p["high_clip_fraction"]:.5f}.','',
            f'Parameters outside the observed clean-training sample range: {outside}. This is a descriptive association, not proof of a causal mechanism or an out-of-spec acquisition. The selected seed is retained even if another seed scores better on confirmation.','',
            '![Validation false alarms](figures/raised_crown_results_20261005/validation_false_alarms.png)','']
    dwell=c.OUT/'dwell_diagnostic.json'
    if dwell.exists():
        p=c.read(dwell)
        lines+=['','### Controlled dwell diagnostic','',p['control']+'. Original validation pixels were reproduced exactly.','',
            '| Condition | Dwell (microseconds) | CNN false regions | Median false regions | Noise RMS |','|---|---:|---:|---:|---:|']
        for row in p['records']:
            lines.append(f'| {row["condition"]} | {row["dwell_time_us"]:.4f} | {row["cnn"]["fp"]} | {row["median"]["fp"]} | {row["noise_rms"]:.6f} |')
        lines+=['',p['limits']+' '+p['selection_effect']+'.','']
    lines+=['',
        '## Robustness','',
        '| Method / condition | F1 | Recall | Individual-pore recall |',
        '|---|---:|---:|---:|']
    for method,groups in stress['groups'].items():
        for condition,item in groups.items():
            lines.append(f'| {method} / {condition} | {item["metrics"]["f1"]:.3f} | {item["metrics"]["recall"]:.3f} | {item["individual_recall"]:.3f} |')
    lines+=['',stress['note'],'',
        'Predictions are mapped back to fixed native truth. Lost coverage counts as a miss. Instance recall is separate from union-region detection, so one merged region is not automatically credited with finding every overlapping pore. The image-based ring estimator is a bounded image-only baseline, not a validated general localisation system.','',
        '### Matched localisation comparison','',
        'The following table uses only clean and close-pair scenes for every condition. This avoids comparing injected errors on one population with nominal performance on a different population.','',
        '| Method | Condition | F1 | Recall |','|---|---|---:|---:|']
    from circular_study import metrics
    for method in ('cnn','median'):
        for condition in sorted(stress['groups'][method]):
            subset=[x for x in stress['rows'] if x['method']==method and x['error']['name']==condition and x['source_class'] in ('clean','pair_near')]
            m=metrics.summary(subset)
            lines.append(f'| {method} | {condition} | {m["f1"]:.3f} | {m["recall"]:.3f} |')
    estimated=[x['error'] for x in stress['rows'] if x['method']=='cnn' and x['source_class']=='clean' and x['error']['name']=='image_estimated']
    if estimated:
        centre=np.array([np.hypot(x['dx'],x['dy']) for x in estimated]);radius=np.array([abs(x['dr']) for x in estimated])
        lines+=['',f'On {len(estimated)} independent clean stress backgrounds, the image-only estimator has median/max centre error {np.median(centre):.3f}/{centre.max():.3f} mm and median/max absolute radius error {np.median(radius):.3f}/{radius.max():.3f} mm. These values are relative to the known synthetic ring and the estimator\'s bounded search prior.','']
    lines+=['','### Individual scene types at nominal localisation','',
        '| Method | Scene type | F1 | Recall | False regions |','|---|---|---:|---:|---:|']
    for method in ('cnn','median'):
        for kind,m in stress['groups'][method]['nominal']['by_class'].items():
            f1=f'{m["f1"]:.3f}' if m['tp']+m['fn'] else 'n/a'
            recall=f'{m["recall"]:.3f}' if m['tp']+m['fn'] else 'n/a'
            lines.append(f'| {method} | {kind} | {f1} | {recall} | {m["fp"]} |')
    paired=c.OUT/'paired_profile/results.json'
    if paired.exists():
        p=c.read(paired)
        lines+=['','### Paired profile diagnostic','',p['claim']+'. Held constant: '+p['held_constant']+'.','',
            '| Method | Profile | F1 | Recall |','|---|---|---:|---:|']
        for method in ('cnn','median'):
            for profile in ('raised','recessed'):
                m=metrics.summary([x for x in p['rows'] if x['method']==method and x['profile']==profile])
                lines.append(f'| {method} | {profile} | {m["f1"]:.3f} | {m["recall"]:.3f} |')
    small=c.OUT/'small_pore_diagnostic.json'
    if small.exists():
        p=c.read(small)
        lines+=['','### Post-freeze small-pore diagnosis','',
            'This diagnostic varies only the component-area filter on the same frozen scores and threshold. It does not change the deployed operating point.','',
            '| Minimum component area (pixels) | F1 | Recall |','|---:|---:|---:|']
        for area,m in p['component_area_comparison'].items():
            lines.append(f'| {area} | {m["f1"]:.3f} | {m["recall"]:.3f} |')
        production=[x for x in p['rows'] if x['min_area']==frozen['cnn']['selection']['best']['min_area']]
        below=sum(x['score_max_in_truth'] is None or x['score_max_in_truth']<x['threshold'] for x in production)
        activated_misses=sum(x['score_max_in_truth'] is not None and x['score_max_in_truth']>=x['threshold'] and x['fn']>0 for x in production)
        lines+=['',f'At the production operating point, {below}/{len(production)} small-pore scenes never reach threshold inside the truth footprint; another {activated_misses} have an above-threshold pixel there but still fail the region-matching criterion. This separates weak score response from unsuccessful region localisation. Lowering the area cutoff alone does not resolve those errors.','']
        lines+=['',p['limit']+'. Detailed phase/pitch footprint errors and within-truth score maxima are saved in `small_pore_diagnostic.json`.']
    sampling=c.OUT/'small_pore_signal_sampling.json'
    if sampling.exists():
        p=c.read(sampling)
        lines+=['','### Deterministic small-pore signal sampling','',p['controls']+'. '+p['limits']+'.','',
            '| Pixel pitch (mm) | Signal RMS | Relative RMS difference to 0.0125 mm |','|---:|---:|---:|']
        for row in p['records']:
            lines.append(f'| {row["pitch_mm"]} | {row["rms_signal"]:.6f} | {row["relative_rms_difference_to_12_5um"]:.3f} |')
    lines+=['','### Error concentration and radial position','',
        'These are descriptive post-freeze results. Associations with acquisition, position or clipping are not causal estimates. Surface geometry labels need not produce a detectable contrast under every detector orientation or noise realisation.','',
        '| Model | Radial pore offset (mm) | Recall | F1 |','|---|---:|---:|---:|']
    for method,item in diagnosis.items():
        for off,m in item['by_offset'].items():
            lines.append(f'| {method} | {off} | {m["recall"]:.3f} | {m["f1"]:.3f} |')
    lines+=['','| Model | Diameter bin (mm) | Recall | F1 |','|---|---|---:|---:|']
    for method,item in diagnosis.items():
        for size,m in item['by_size'].items():
            lines.append(f'| {method} | {size} | {m["recall"]:.3f} | {m["f1"]:.3f} |')
    for method,item in diagnosis.items():
        worst=item['backgrounds'][0]
        lines+=['',f'{method}: largest false-region count occurs on `{worst["background_id"]}`: {worst["metrics"]["fp"]} false regions across its five conditions, including {worst["clean_false_regions"]} on its clean ring. Its dwell is {worst["beam"]["dwell_time_us"]:.4f} microseconds. Clean-image high clipping fraction is {worst["high_clip_fraction"]:.5f}. Hidden-control rings flagged: {item["hidden_rings_flagged"]}/24.','']
    lines+=['',
        '## Verification and limitations','',
        f'Pilot gate: `{pilot["numerical_gate"]}`. Geometry anchors, periodicity, displacement-volume identity, hidden/clean image equivalence and visibility refinement were checked. Detailed values are in the saved pilot report.','',
        'Nonlocal surface visibility is included under the inherited parallel-direction/constant-local-working-distance detector approximation. No electron trajectories or material-specific information-depth kernel are simulated. Detector response, roughness and acquisition remain uncalibrated. The provided macrograph is qualitative inspiration, not a measured fit.','',
        'Sample size supports an exploratory synthetic benchmark, not a power or industrial reliability claim. Adjacent sectors are spatial and do not increase the number of independent weld specimens. Per-scene clipping, signal/noise and visible-area diagnostics are saved in `image_quality.json`; they do not filter scenes or alter thresholds.','',
        'Prior failed/development pilot versions remain on disk with source snapshots; only the active campaign linked below supplies these results. No historical model or dataset was overwritten.','',
        '## Reproduce','',
        'From the repository root:','',
        '```powershell',r'.venv\Scripts\python.exe WeldDefectDetection/run_raised_crown_20261005.py all',r'.venv\Scripts\python.exe WeldDefectDetection/complete_raised_delivery_20261005.py','```','',
        f'Active artifacts: `{c.OUT}`. See `protocol.json`, `pilot/gates.json`, `cv_protocol.json`, `frozen_models.json`, `dataset_manifest.json`, `results.json`, `robustness_results.json` and `audit.json`.','',
        '[SEM/EBW research and modelling decisions](SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md).',
        '[Chronological methods guide](RAISED_CROWN_METHODS_20261005.md).','']
    lines+=['[Evidence-led next experiment](RAISED_CROWN_NEXT_STEPS_20261005.md).','']
    (c.ROOT/'docs/RAISED_CROWN_RESULTS_20261005.md').write_text('\n'.join(lines),encoding='utf-8')
    methods(c.ROOT/'docs/RAISED_CROWN_METHODS_20261005.md')
    print('Reports and figures saved',flush=True)

def methods(path):
    path.write_text('''# Raised-crown methods, in execution order

## 1. Geometry before pixels

The analytic fusion outline is a solid-material label. Surface height h(u,theta) is a separate boundary connected to it. The nominal crown uses H[1-(u/a)^2]^3 on its compact support, with H=3 mm and provisional a=1.4 mm. Periodic width/height variations are explicitly assumed. A 7 mm fusion depth is never assigned to a surface-trench parameter.

## 2. Pore geometry and surface expression

An open pore cuts a cap-shaped bowl into the healthy surface. An enclosed pore is represented by a void in surface-following coordinates and can have no surface effect. The displacement experiment adds a compact bump whose integral equals an assumed fraction of the pore volume. The model cannot infer that fraction from pore volume alone. Overlapping openings use the lowest floor; bulges add prescribed volumes. Visible footprint masks and individual IDs are saved separately from internal geometry.

## 3. From height to detector signals

Cartesian finite differences estimate hx and hy. The unit normal is (-hx,-hy,1)/sqrt(1+hx^2+hy^2). Detector quadrature integrates the existing angular/yield response over each quadrant. The old B fit is not extrapolated: the explicit uncalibrated Lambertian fallback sets B=0. Ray horizons reject directions blocked by another part of the height surface. This adds geometric occlusion, not electron transport. The constant local working distance and parallel-direction convention remain assumptions.

The four signals are A, B, C, D. Tx=(A+D)-(B+C), Ty=(A+B)-(C+D), and TOPO=sqrt(Tx^2+Ty^2). TOPO is a contrast magnitude, not height. A single crest can create two bright slope bands. Native raster acquisition then adds the inherited unmeasured shot/readout/drift model. No new literature-derived penetration depth is used.

## 4. Coordinates and preprocessing

Sample a complete 160x5024 annulus directly from the Cartesian image at 0.05 mm radial pitch. Score central rows 16:144, a nominal 6.4 mm band. The outer rows provide context. Training pixels alone determine the percentile-based raw range and square-root tone curve. Every sample sharing a physical background stays in one split. Ring localisation errors re-extract the native image, and scores return to fixed true coordinates before scoring.

## 5. CNN, layer by layer

Each input is one TOPO channel, shape [batch,1,64,64]. A convolution block contains two 3x3 convolutions, each followed by batch normalisation and ReLU. Spatial dimensions are preserved by padding.

| Stage | Output per example | Role |
|---|---|---|
| Encoder 1, two convolutions | 24x64x64 | Local edges and intensity patterns |
| Max pool 1 | 24x32x32 | Reduce spatial resolution |
| Encoder 2, two convolutions | 48x32x32 | Combine local patterns |
| Max pool 2 | 48x16x16 | Broader effective context |
| Bottleneck, two convolutions | 96x16x16 | Coarse contextual representation |
| Transposed convolution | 48x32x32 | Upsample |
| Concatenate encoder-2 skip | 96x32x32 | Restore detail from earlier features |
| Decoder 2, two convolutions | 48x32x32 | Fuse context and detail |
| Transposed convolution | 24x64x64 | Return to input resolution |
| Concatenate encoder-1 skip | 48x64x64 | Restore fine spatial information |
| Decoder 1, two convolutions | 24x64x64 | Produce local segmentation features |
| 1x1 output convolution | 1x64x64 | Produce one logit per pixel |
| Sigmoid | 1x64x64 | Convert logits to model scores |

Training minimises weighted binary cross-entropy plus Dice loss with AdamW and a cosine learning-rate schedule. Half the sampled patches contain visible positive truth and half are negative. Training uses labels; inference does not. Twenty epochs are fixed in advance. Validation selects the operating point and one of three trained seeds. At inference, retain central 32x32 predictions from overlapping tiles, assemble the complete ring, then threshold and extract periodic connected regions. Scores are not calibrated probabilities of an internal pore or manufacturing rejection.

For a pixel logit z, the sigmoid score is p = 1/(1+exp(-z)); its training label y is 0 or 1. Weighted cross-entropy penalises -[w*y*log(p) + (1-y)*log(1-p)], averaged over pixels. The added Dice loss is 1 - (2*sum(p*y)+1)/(sum(p)+sum(y)+1), computed over the batch. The positive weight is derived from training-patch occupancy and capped at 20. AdamW starts at learning rate 0.001 with weight decay 0.0001; cosine scheduling ends at 0.00005. Gradients are clipped to norm 5. Each epoch samples 1,200 patches, with batch size 32. The network has 262,777 parameters.

Batch normalisation uses batch statistics during training and the learned running statistics during inference. Pooling reduces spatial resolution without learned weights. Transposed convolutions learn how to upsample; skip connections preserve earlier spatial details. ReLU keeps positive activations and maps negative ones to zero. The final threshold and component rule convert scores into reported regions; they are separate from the neural-network layers.

## 6. Neighbouring median, chronologically

For each 64x64 target window, take eight spatial neighbours at angular offsets +/-64, +/-128, +/-192 and +/-256 pixels. Compute the median along each row of the target and each neighbour. Smooth their row-profile difference with the existing nine-row Gaussian and add that correction to each neighbour. Then take the pixelwise median of the eight corrected neighbours. This is intensity-profile alignment, not geometric translation of the images. Subtract that reference from the target and smooth the residual with the inherited one-pixel Gaussian. Clean training windows estimate a robust residual centre and scale separately for each radial tile position. The score is the absolute centred residual divided by that scale.

As with the CNN, keep central 32x32 scores, assemble the full ring, and apply a validation-selected threshold/component rule. Nearby defects can contaminate reference windows; smooth ridge variation and mislocalisation can create residuals without a pore. This is a spatial reference method, not a temporal prediction model. It has no learned convolutional layers.

## 7. Evaluation and what it proves

Match predicted and true circular boxes one-to-one at IoU >=0.5. Report precision, recall, F1, pixel IoU and clean-ring alarms. Bootstrap independent backgrounds, not overlapping patches. Report individual-pore recall separately from detection of a merged overlap region. Enclosed no-response controls test whether the pipeline invents a surface cue. Their lack of a simulated signal is a declared renderer assumption, not a measured SEM information-depth result.

The comparison answers whether these methods detect the declared synthetic surface anomalies and tolerate the tested coordinate errors. Measured surface profiles, detector calibration and independent real TOPO images remain necessary for physical and transfer claims.

## 8. One sample traced through the implemented pipeline

![Pilot bulge traced through geometry, detector and both CV paths](figures/raised_crown_results_20261005/traced_sample.png)

This is the declared pilot bulge, not a newly selected confirmation example. Read the panels in row order: height and inclination; four quiet quadrant signals; quiet and acquired TOPO; training-normalized TOPO and surface truth; CNN scores and final connected regions; then median reference, smoothed residual, calibrated score and final regions. The numerical arrays, eight neighbour patches and provenance are saved in `traced_sample.npz` and `traced_sample.json` in the active study. Truth is displayed to explain the sample and never enters inference.

The illustrated tile is 64 x 64 pixels. Production inference retains only its central 32 x 32 scores; final regions come from stitching the complete ring before thresholding and component filtering. The four quadrant images precede the random acquisition effects; their common focusing factor is applied when forming the saved quiet TOPO. Colourbars identify the panel scales, so displayed brightness between panels is not a shared physical scale.

## 9. Follow the code

- [Geometry](../WeldDefectDetection/raised_crown/geometry.py): `height`, `fusion_half_width`, `fusion`, `pore_contains`, `surface_effect`.
- [Rendering and native extraction](../WeldDefectDetection/raised_crown/rendering.py): `Engine.volume`, `Engine.inject`, `render`, `extract`.
- [Nonlocal detector visibility](../WeldDefectDetection/raised_crown/shadow.py): `VisibilityQuadrature.horizon`, `integrated`, `compute`.
- [Actual CNN layer definitions](../WeldDefectDetection/src/algorithms/classical/cnn_detector.py): `TinyDefectSegNet`.
- [Training, normalisation and whole-ring inference](../WeldDefectDetection/raised_crown/learning.py): `normalize`, `populations`, `batch`, `train`, `predict_image`, `median_fit`, `median_image`.
- [Median neighbours and residual](../WeldDefectDetection/circular_study/median.py): `residual`; [reference profile correction](../WeldDefectDetection/box_development/methods.py): `aligned_reference`.
- [Frozen assessment](../WeldDefectDetection/raised_crown/assessment.py): `evaluate`, `stress`, `map_back`, `estimated_error`.
- [Teaching trace](../WeldDefectDetection/raised_crown/trace_sample.py): one pilot sample through both implemented methods.
''',encoding='utf-8')
