"""Artifact-derived tables and figures; no manually supplied performance numbers."""
import json
from pathlib import Path
import numpy as np
from box_comparison.campaign import read,ARMS
from box_development.data import load


def build(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);r=read(root/'results.json');f=read(root/'frozen_models.json')
    gate=read(root/'pilot_gate.json');population=read(root/'development_population.json')
    overfit=read(root/'overfit_diagnostic.json')
    text=['# Surface-expressed pore detection campaign','',
        'This campaign uses clean welds, keyhole pores and intrinsic pores. Each defective sequence has exactly one pore type. The renderer remains an empirical synthetic model; B remains assumed, and measured-SEM validation is absent.',
        '', '## 1. Scope and generation','',
        '304SS, 60 keV, straight welds, 0.05 mm/pixel, 64x64 targets, nine adjacent frames and five target windows per sequence. Four neighbouring crops exclude the target. Existing pore-response coefficients, noise and geometric labels are unchanged.',
        'Keyhole diameter 0.50-1.20 mm and depth parameter 0.25-0.60 mm; intrinsic diameter 0.50-1.00 mm and depth parameter 0.25-0.65 mm. Depth is a simulator parameter, not the realized surface depression.',
        '', '## 2. Surface response pilot','',
        f"The 3x3 diameter/depth grid for each pore type passed operational response checks: {gate['passed']}. All 18 pore grid scenes were retained. Clean scenes were also rendered. Masks were checked against noiseless replays.",
        'The figures show defect-off, noiseless defect-on, noisy defect-on, absolute signal change and the geometric mask. Difference images are individually scaled and must not be read as equal absolute contrast.',
        '', '| Pore | Median mask fraction above 2 noise SD | Minimum/maximum realized height change (um) |', '|---|---:|---:|']
    for kind in ('keyhole_pore','intrinsic_pore'):
        rows=[x for x in gate['rows'] if x['kind']==kind]
        text.append(f"| {kind} | {np.median([x['fraction_above_2_noise_std'] for x in rows]):.3f} | {min(x['max_height_change_mm'] for x in rows)*1000:.3f} / {max(x['max_height_change_mm'] for x in rows)*1000:.3f} |")
    text+=['','The two-noise-SD comparison is descriptive. It did not change labels or remove low-contrast samples. Passing this pilot does not physically qualify the renderer.',
           '', '## 3. Development and selection','',f'Development membership: `{json.dumps(population)}`.',
           'The original training-only normalization was frozen throughout. Validation chooses thresholds, component filtering, checkpoints and the primary method. The development test split was not evaluated or used for selection.',
           f"A 16-target training memorization diagnostic reached pixel F1 {overfit['pixel_f1']:.3f}. This is not held-out performance.",
           'Three independent training seeds were run for each neural method, twenty epochs each. The classical search compared row-profile residuals and local-variation residuals. All methods use the same validation targets and unfiltered truth boxes.',
           '', '| Method | Validation box F1 | Selected threshold | Minimum area |', '|---|---:|---:|---:|']
    for a in ARMS:
        spec=f['models'][a];params=spec['spec']['parameters'] if spec['implementation']=='historical' else spec['parameters']
        text.append(f"| {a} | {spec['metrics']['box_f1']:.3f} | {params['threshold']} | {params['min_area']} |")
    text+=['',f"The primary method selected before confirmation was **{f['primary']}**.",
        '', '## 4. Fresh independent confirmation','',
        'Thirty sequences per source class, including clean welds, use a disjoint seed schedule. All five eligible target windows per sequence are evaluated. Matching requires IoU >= 0.5, one prediction per truth region. F1 = 2TP/(2TP+FP+FN).',
        '', '| Method | TP | FP | FN | Box F1 (95% sequence bootstrap CI) | Pixel IoU | Clean false alarms |', '|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        m=r['metrics'][a];ci=r['uncertainty']['intervals'][a]['box_f1']
        text.append(f"| {a} | {m['tp']} | {m['fp']} | {m['fn']} | {m['box_f1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {m['pixel_iou']:.3f} | {m['clean_false_alarms']}/{m['clean_targets']} |")
    text+=['','| Method | Keyhole box F1 | Intrinsic box F1 | Noiseless overall box F1 | Previous model on identical noisy inputs |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        text.append(f"| {a} | {r['per_class'][a]['keyhole_pore']['box_f1']:.3f} | {r['per_class'][a]['intrinsic_pore']['box_f1']:.3f} | {r['metrics']['quiet_'+a]['box_f1']:.3f} | {r['metrics']['previous_'+a]['box_f1']:.3f} |")
    text+=['', '## 5. What the results establish','',
        f"Synthetic image/pixel skill gate passed: **{r['synthetic_pixel_image_skill_gate']}**. This checks image balanced accuracy above chance and pixel IoU improvement over both fixed baselines. It does not certify reliable box localization.",
        f"Primary box F1 >= 0.7: **{r['metrics'][r['primary']]['box_f1']>=.7}**. No settings were altered after confirmation to meet this target.",
        'Confidence intervals use 2,000 paired sequence resamples. Multiple comparisons are exploratory. Paired noisy/noiseless performance separates one contribution of noise; weak geometry response and model limitations can remain.',
        '', '| Paired comparison | Box F1 difference 95% CI |', '|---|---:|']
    for name,values in r['uncertainty']['paired_difference_intervals'].items():
        ci=values['box_f1'];text.append(f'| {name} | [{ci[0]:.3f}, {ci[1]:.3f}] |')
    text+=['','## 6. Reproduction and inference','',
        'Run from the repository root:', '```powershell',
        r"& '.\.venv\Scripts\python.exe' '.\WeldDefectDetection\run_pore_campaign.py' --stage all",'```',
        'Completed stages are resumed only after their receipts, sources and dataset inventories verify. Existing unverified partial generation is preserved and stops the run.',
        'Label-free inference example:', '```powershell',
        r"& '.\.venv\Scripts\python.exe' '.\WeldDefectDetection\infer_pore_boxes.py' --approach cnn --input '.\DataSimulation\data\output\pore_campaign_20260921_v1\inference_smoke\input.npz' --output '.\pore_inference_boxes.json'",'```',
        'The input contains raw target/context crops and physical-domain metadata, without labels. Output is half-open xyxy boxes. All four inference paths were checked against the confirmation evaluator.',
        '', '## 7. Limitations','',
        'Surface anchoring and a nonempty geometric mask do not guarantee an easily visible pore. The volume occupancy cutoff and three-layer mask projection can extend beyond strong surface response. Internal pores, real SEM transfer, circular seams and physical electron transport remain outside this campaign.',
        'The reference crops are different neighbouring surface locations, not repeated observations of the target. Context can add mismatch and false alarms. The shuffled-context comparison tests dependence on the chosen neighbours.',
        'Historical seven-class campaigns remain unchanged. This scope was chosen after viewing their results; the new confirmation seed is independent, while the choice of research question is informed by earlier work.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    x=np.arange(4);fig,ax=plt.subplots(figsize=(9,4))
    for j,(prefix,label) in enumerate((('previous_','Previous model'),('','Pore-trained'),('quiet_','Pore-trained, noiseless'))):
        ax.bar(x+(j-1)*.25,[r['metrics'][prefix+a]['box_f1'] for a in ARMS],width=.24,label=label)
    ax.axhline(.7,color='gray',linestyle=':',label='0.7 project target');ax.set_ylim(0,1)
    ax.set_xticks(x,ARMS);ax.set_ylabel('Box F1, IoU >= 0.5');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    data=load(root/'datasets/confirmation','confirmation');preds={a:read(root/'predictions'/(a+'.json')) for a in ARMS}
    # Deterministic examples include misses and clean false alarms when they occur.
    primary=preds[r['primary']];ids=[]
    for kind in ('keyhole_pore','intrinsic_pore'):
        for condition in (lambda q:q['tp']>0,lambda q:q['fn']>0):
            choices=[i for i,q in enumerate(primary) if q['source_class']==kind and condition(q)]
            if choices:ids.append(choices[0])
    for condition in (lambda q:not q['target_positive'] and q['fp']>0,lambda q:not q['target_positive'] and q['fp']==0):
        choices=[i for i,q in enumerate(primary) if condition(q)]
        if choices:ids.append(choices[0])
    ids=list(dict.fromkeys(ids));fig,axes=plt.subplots(len(ids),4,figsize=(11,2.6*len(ids)),squeeze=False)
    from matplotlib.patches import Rectangle
    for row,i in enumerate(ids):
        for col,a in enumerate(ARMS):
            ax=axes[row,col];ax.imshow(data['target'][i],cmap='gray',vmin=0,vmax=1)
            for boxes,color in ((data['meta'][i]['truth_boxes'],'lime'),(preds[a][i]['boxes_xyxy'],'red')):
                for x0,y0,x1,y1 in boxes:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=1))
            ax.set_title(a,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(data['meta'][i]['source_class'],fontsize=8)
    fig.suptitle('Green: geometric truth; red: predicted boxes. Selected successes, misses and clean cases.')
    fig.tight_layout();fig.savefig(root/'examples.png',dpi=140);plt.close(fig)
