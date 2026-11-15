"""Evidence report derived from frozen JSON, with no selection or model updates."""
import math
from pathlib import Path
import numpy as np
from . import campaign as c, learning as learn


def f(value): return f'{value:.3f}'


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frozen=learn.verify_freeze(); result=c.read(c.OUT/'results.json'); robustness=c.read(c.OUT/'robustness_results.json')
    audit=c.read(c.OUT/'audit.json'); assert audit['status']=='passed'
    docs=c.ROOT/'docs'; figures=docs/'figures/raised_acquisition_20261006'; figures.mkdir(parents=True,exist_ok=True)
    selected=result['selected']; names=['frozen_v5_cnn',selected['coverage'],selected['hard_negative'],'median','frozen_v5_median']
    labels=['Frozen 5 Oct CNN','Coverage CNN','Hard-negative CNN','New median','Frozen 5 Oct median']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,label in zip(names,labels):
        bybin=result['results'][name]['by_dwell_bin']
        axes[0].plot(range(4),[bybin[str(b)]['f1'] for b in range(4)],marker='o',label=label)
        axes[1].plot(range(4),[bybin[str(b)]['clean_ring_fa'] for b in range(4)],marker='o',label=label)
    for ax in axes:
        ax.set_xticks(range(4),['0.1–0.5','0.5–2','2–8','8–20']); ax.set_xlabel('Dwell range (microseconds)'); ax.set_ylim(-.03,1.03); ax.grid(alpha=.2)
    axes[0].set_ylabel('Full-ring box F1'); axes[1].set_ylabel('Fraction of clean rings flagged'); axes[0].legend(fontsize=8)
    fig.suptitle('Fresh balanced acquisition challenge | six independent backgrounds per dwell range\nSynthetic results; operating points fixed before confirmation')
    fig.savefig(figures/'dwell_results.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for j,arm in enumerate(c.ARMS):
        specs=frozen['arms'][arm]['cnn_candidates']
        val=[s['selection']['best']['metrics']['f1'] for s in specs]
        test=[result['results'][f'{arm}_{s["seed"]}']['metrics']['f1'] for s in specs]
        for i,(v,t) in enumerate(zip(val,test)):
            ax.plot([j-.12,j+.12],[v,t],marker='o',alpha=.7,label=f'{arm}, seed {i+1}')
    ax.set_xticks([-.12,.12,.88,1.12],['Coverage\nvalidation','Coverage\nconfirmation','Hard negatives\nvalidation','Hard negatives\nconfirmation'])
    ax.set_ylabel('Box F1'); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.legend(fontsize=7,ncol=2); ax.set_title('Training-seed variability | all final-epoch models retained')
    fig.savefig(figures/'seed_variability.png',dpi=180); plt.close(fig)
    from raised_crown.learning import normalize,tile
    train=c.read(c.OUT/'datasets/train/index.json'); norm=c.read(c.OUT/'normalization.json')
    bank=frozen['arms']['hard_negative']['cnn']['mining'][-1]
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),layout='constrained')
    for b,ax in enumerate(axes):
        candidates=[(score,coord) for score,coord in zip(bank['scores'],bank['coordinates']) if train[coord[0]]['dwell_bin']==b]
        score,(ring,y,x)=max(candidates,key=lambda pair:pair[0]); meta=train[ring]
        with np.load(meta['npz_path']) as a: image=tile(normalize(a['raw'][0],norm),y,x)
        ax.imshow(image,origin='lower',cmap='gray',vmin=0,vmax=1)
        ax.set_title(f'Clean, dwell {meta["dwell_time_us"]:.3g} us\nMining score {score:.3f}'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Actual clean training patches retained after epoch 15\nHighest-ranked retained example in each dwell range; scores precede subsequent training')
    fig.savefig(figures/'mined_clean_examples.png',dpi=180); plt.close(fig)
    lines=['# Circular crown acquisition study, 6 October 2026','',
        'Completed campaign: `raised_acquisition_20261006_v1`. The circular crown, renderer, noise law, CNN architecture and neighbouring-median algorithm are unchanged. The experiment changes acquisition sampling and CNN negative-tile sampling.',
        '',f'Independent backgrounds: {audit["backgrounds"]}. Scenes: {audit["scenes"]}. Every split is disjoint and all prior artifacts in the frozen inventory are unchanged.',
        '', '## Fresh confirmation results','',
        'The confirmation set deliberately balances four dwell ranges from 0.1 to 20 microseconds. These are results on the same fresh images, not a comparison of scores from different test populations. CNN seeds and operating points were selected using validation only.',
        '', '| Method | Box F1 | 95% background interval | Precision | Recall | Clean rings flagged |',
        '|---|---:|---|---:|---:|---:|']
    for name,label in zip(names,labels):
        value=result['results'][name]; m=value['metrics']; ci=value['intervals']['f1']
        lines.append(f'| {label} | {f(m["f1"])} | [{f(ci[0])}, {f(ci[1])}] | {f(m["precision"])} | {f(m["recall"])} | {m["clean_rings_flagged"]}/{m["clean_rings"]} |')
    new_m=result['results']['median']['metrics']; old_m=result['results']['frozen_v5_median']['metrics']
    if new_m['f1']<old_m['f1']:
        lines += ['',f'The median recalibration is not an F1 improvement: {old_m["f1"]:.3f} to {new_m["f1"]:.3f}. '
            f'Recall changes from {old_m["recall"]:.3f} to {new_m["recall"]:.3f}, while false regions increase from {old_m["fp"]} to {new_m["fp"]}. '
            'Both results remain visible. The comparator is not switched retrospectively using confirmation.']
    selected_stress=robustness['groups'][frozen['selected_arm']]
    lines += ['',f'**Remaining limitation: ring localisation.** The primary confirmation uses declared ring coordinates. '
        f'On the same separate 84-scene robustness population, the selected model scores {selected_stress["nominal"]["metrics"]["f1"]:.3f} F1 with nominal coordinates '
        f'and {selected_stress["image_estimated"]["metrics"]["f1"]:.3f} with image-estimated coordinates. '
        'The acquisition improvement does not establish localisation robustness. Do not interpret the main-confirmation score as an end-to-end unknown-image result.']
    lines += ['',f'The GUI default is **{frozen["selected_arm"]}**, selected before confirmation. This choice is not replaced by whichever model has the highest confirmation score.',
        '', 'All confirmation metrics above include 120 scenes, including hidden controls. Those controls exactly repeat the clean surface signal and are grouped with their background in uncertainty calculations. They are not independent additional negative examples.',
        '', '| Method | F1 excluding duplicate hidden controls (96 scenes) | Hidden controls flagged |','|---|---:|---:|']
    for name,label in zip(names,labels):
        value=result['results'][name]
        lines.append(f'| {label} | {f(value["without_duplicate_hidden"]["f1"])} | {value["hidden_rings_flagged"]}/{value["hidden_negative_rings"]} |')
    failures={}
    for arm in c.ARMS:
        records=result['results'][selected[arm]]['rows']; backgrounds=[]
        for key in sorted({x['background_id'] for x in records}):
            subset=[x for x in records if x['background_id']==key]
            backgrounds.append(dict(background_id=key,dwell_time_us=subset[0]['dwell_time_us'],
                dwell_bin=subset[0]['dwell_bin'],fp=sum(x['fp'] for x in subset),fn=sum(x['fn'] for x in subset),
                tp=sum(x['tp'] for x in subset),clean_fp=sum(x['fp'] for x in subset if x['source_class']=='clean')))
        failures[arm]=sorted(backgrounds,key=lambda x:(x['fp'],x['fn']),reverse=True)
    c.write(c.OUT/'failure_analysis.json',dict(backgrounds=failures,
        interpretation='Descriptive post-freeze ranking; no model or threshold changes'))
    lines += ['', '## Concentration of remaining errors','',
        '| Arm | Highest-FP background | Dwell us | False regions across paired scenes | False regions on clean scene | Missed regions |',
        '|---|---|---:|---:|---:|---:|']
    for arm,backgrounds in failures.items():
        for row in backgrounds[:3]:
            lines.append(f'| {arm} | {row["background_id"]} | {row["dwell_time_us"]:.3g} | {row["fp"]} | {row["clean_fp"]} | {row["fn"]} |')
    lines += ['', 'These background rankings are descriptive. They must not be used to retune the frozen models or discard difficult confirmation scenes.']
    lines += ['', '## Surface-condition recovery','',
        '| Method | Open keyhole-shaped pore recall | Open intrinsic-pore recall | Assumed bulge recall |',
        '|---|---:|---:|---:|']
    for name,label in zip(names,labels):
        classes=result['results'][name]['by_class']
        lines.append(f'| {label} | '+ ' | '.join(f(classes[k]['recall']) for k in ('open_keyhole','open_intrinsic','bulge'))+' |')
    lines += ['', 'Each positive condition contains 24 scenes. Recall alone ignores false regions; use the overall precision and clean-ring alarm results alongside this table. Bulges remain an assumed surface-displacement case, not evidence of buried-pore visibility.']
    lines += ['', '## Paired comparisons','', '| Comparison | F1 difference | 95% paired interval | Recall difference | Clean alarm-rate difference |','|---|---:|---|---:|---:|']
    for label,values in result['paired_differences'].items():
        d=values['f1']; interval=d['interval_95']
        lines.append(f'| {label.replace("_"," ")} | {d["difference"]:+.3f} | [{interval[0]:+.3f}, {interval[1]:+.3f}] | {values["recall"]["difference"]:+.3f} | {values["clean_ring_fa"]["difference"]:+.3f} |')
    difference=result['paired_differences']['hard_minus_coverage']['f1']; interval=difference['interval_95']
    if interval[0]>0: conclusion='For the validation-selected fitted models, the paired bootstrap interval favours the hard-negative model on this synthetic population. This is not a guarantee of improvement across initialisations.'
    elif interval[1]<0: conclusion='The hard-negative procedure reduces F1 in this comparison; it should not be described as a demonstrated improvement.'
    else: conclusion='The paired interval includes zero. This campaign does not establish a reliable incremental F1 benefit from hard-negative mining.'
    lines += ['', conclusion,
        '', 'Intervals use 2,000 paired replicates, resampling independent backgrounds within dwell bins. Only six confirmation backgrounds are available per bin. These intervals condition on the fitted models and observed synthetic backgrounds, and do not include uncertainty about real-world domain shift. Three trained seeds show additional optimisation variability.',
        '', 'If a method flags 0/24 clean rings, its empirical bootstrap alarm interval is degenerate at zero. Even under an additional independent, identically distributed Bernoulli assumption, the one-sided 95% upper bound would be about 11.7%, not zero; the intentionally stratified design is not proof of a deployment alarm rate.',
        '', 'Dwell-bin summaries are stratified diagnostics, not a within-background causal dose-response. The inherited index-based layouts do not form a full factorial of seam position, crown variation and defect size. The matched training-arm comparison uses exactly the same scenes in both arms.',
        '', '![Dwell-stratified results](figures/raised_acquisition_20261006/dwell_results.png)',
        '', '## All seeds and validation choices','', '| Arm | Seed | Validation F1 | Confirmation F1 | Threshold | Area | Closing | Selected within arm |','|---|---:|---:|---:|---:|---:|---:|---|']
    for arm in c.ARMS:
        for s in frozen['arms'][arm]['cnn_candidates']:
            op=s['selection']['best']; m=result['results'][f'{arm}_{s["seed"]}']['metrics']
            lines.append(f'| {arm} | {s["seed"]} | {f(op["metrics"]["f1"])} | {f(m["f1"])} | {op["threshold"]} | {op["min_area"]} | {op["closing"]} | {s["seed"]==frozen["arms"][arm]["cnn"]["seed"]} |')
    lines += ['', '![Seed variability](figures/raised_acquisition_20261006/seed_variability.png)',
        '', '## Multiple defects and ring localisation','',
        'Stress includes clean rings, separated, near and overlapping pairs, triples, small openings and hidden controls. Nominal and image-estimated coordinates use all 84 scenes. The combined 0.25 mm x/y/radius error applies only to clean and near-pair scenes, so its matched nominal subset is listed separately.',
        '', '| Method | Coordinate condition | Box F1 | Region recall | Individual-defect recall | Clean rings flagged |','|---|---|---:|---:|---:|---:|']
    for method,conditions in robustness['groups'].items():
        for condition,value in conditions.items():
            m=value['metrics']; individual=f(value['individual_recall']) if 'individual_recall' in value else 'not aggregated'
            lines.append(f'| {method} | {condition} | {f(m["f1"])} | {f(m["recall"])} | {individual} | {m["clean_rings_flagged"]}/{m["clean_rings"]} |')
    lines += ['', 'Overlapping defects may merge into one visible footprint. Region recall and individual-defect recall therefore answer different questions; one merged prediction cannot be counted as several correctly localised defects.',
        '', '## What changed, chronologically','',
        '1. The original 0.1–20 microsecond acquisition range is divided into four strata before rendering. Each independent background gets a declared dwell value. The same unchanged renderer creates related clean and defective scenes. This improves coverage without weakening noise or increasing pore strength.',
        '2. Train-only quantiles define the same clipped square-root input transform. The median calibration is fitted only to clean training images. Validation and confirmation do not alter either transform.',
        '3. Both CNN arms retain the existing 262,777-parameter TinyDefectSegNet, four radial tiles, weighted binary cross-entropy plus Dice loss, and 20 epochs with 1,200 samples per epoch. Their positive samples and initial weights match by seed.',
        '4. The coverage arm samples all negative tiles uniformly. After epochs 5, 10 and 15, the second arm scores a fixed sample of clean training patches. Patches with high predicted defect probability are difficult negatives. Keeping 16 from each background prevents one noisy ring from taking the entire bank. Subsequent batches contain 600 positive, 300 uniform negative and 300 mined negative tiles. Mining adds forward-pass compute; it does not add optimizer updates.',
        '5. Validation selects the final-epoch seed and threshold/area/closing settings. Both arms and the median are frozen before fresh confirmation is generated. Test results cannot feed back into this experiment.',
        '6. Evaluation scores complete rings with periodic connectivity, one-to-one box matching at IoU 0.5, and background-level uncertainty. The GUI uses the same input contract and frozen choices, never truth masks.',
        '', '![Actual mined clean training patches](figures/raised_acquisition_20261006/mined_clean_examples.png)',
        '', 'For the unchanged CNN layers, geometry-to-image calculations and median operations, see the [chronological method guide](RAISED_CROWN_METHODS_20261005.md). The [prospective protocol](RAISED_ACQUISITION_PROTOCOL_20261006.md) records the comparison before fitting.',
        '', '## Interpretation and next steps','',
        'This tests a training-process improvement inside the existing circular synthetic thesis scope. Comparison with the old frozen CNN includes changed training draws, seeds and a larger validation set (24 rather than 12 backgrounds), so it is not a pure causal estimate of dwell coverage alone. The matched two-arm comparison specifically tests the mining schedule on this dataset.',
        '', 'The next controlled experiment should address localisation-aware training and validation: perturb native-image extraction coordinates, preserve the physical surface and fixed truth coordinates, and compare nominal-only training with coordinate-perturbed training. Select settings on development/validation and evaluate once on fresh backgrounds. The now-exposed stress scenes can motivate the design but cannot remain its untouched confirmation.',
        '', 'Also replicate across more independent low-dwell clean backgrounds and initialisations. The median needs a separate development study of reference contamination and acquisition-dependent residual calibration; its refit here did not improve F1. Do not retrospectively switch this campaign\'s comparator or thresholds. Real-image validation remains necessary for a physical inspection claim. Unfamiliar weld profiles and electron-transport simulation remain outside this increment.',
        '', 'GUI checks establish that generation, inference and export work and that labels are not used. They do not require every generated clean or hidden-control example to have zero false alarms. Zero observed alarms in 24 confirmation backgrounds is not a guarantee for subsequent examples.',
        '', '## Run and inspect','',
        'Use `Launch Weld Vision.cmd` or `launch_weld_studio.py`. The new acquisition tab exposes the frozen CNN arms and a dwell control, while the 5 October crown and older recessed workflows remain labelled historical. Generated examples are saved separately from research datasets.',
        '', '```powershell', '.venv\\Scripts\\python.exe WeldDefectDetection/run_raised_acquisition_20261006.py all','```',
        '', 'Evidence: `DataSimulation/data/output/raised_acquisition_20261006_v1/` contains the protocol, source hashes, training histories and mining coordinates, frozen models, scene manifests, predictions, robustness results and audit. Historical datasets, models and release pointers remain unchanged.',
        '', 'Synthetic evidence only. The raised crown is prescribed geometry. Pore-induced bulges are an explicit displacement hypothesis, not a validated welding-fluid model. No researched SEM/BSE penetration number was implemented.']
    path=docs/'RAISED_ACQUISITION_RESULTS_20261006.md'; path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    c.write(c.OUT/'report_artifacts.json',dict(report=str(path),sha256=c.sha(path),figures={str(p):c.sha(p) for p in figures.glob('*.png')}))
    print('Report',path,flush=True)
