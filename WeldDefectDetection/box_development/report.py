from pathlib import Path
import numpy as np
from box_comparison.core import sha,write
from box_comparison.campaign import read,inventory


def build(root):
    from .campaign import check,require,ARMS
    from .data import load
    check(root);require(root,'verify')
    results=read(root/'results.json');frozen=read(root/'frozen_models.json');m=results['metrics'];ci=results['uncertainty']
    curves=read(root/'learning_curve.json');population=read(root/'training_populations.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,arm in zip(axes,('cnn','cnn_median')):
        for seed in sorted({r['seed'] for r in curves}):
            rows=sorted([r for r in curves if r['seed']==seed and r['arm']==arm],key=lambda r:r['multiplier'])
            ax.plot([r['training_sequences'] for r in rows],[r['metrics']['box_f1'] for r in rows],marker='o',label=str(seed))
        ax.set_title(arm);ax.set_xlabel('Independent training sequences');ax.set_ylabel('Validation box F1');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'learning_curve.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4)
    for offset,prefix,label in ((-.25,'original_','First campaign'),(0,'previous_','Second campaign'),(.25,'','Selected development')):
        ax.bar(x+offset,[m[prefix+a]['box_f1'] for a in ARMS],.25,label=label)
    ax.set_xticks(x,ARMS);ax.set_ylabel('Confirmation box F1');ax.legend();fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    data=load(root/'datasets/confirmation','confirmation')
    predictions={a:read(root/'predictions'/f'{a}.json') for a in ARMS}
    choices=[]
    for label in sorted({r['source_class'] for r in data['meta']}):
        indices=[i for i,r in enumerate(data['meta']) if r['source_class']==label]
        choices.append(max(indices,key=lambda i:int(data['mask'][i].sum())))
    fig,axes=plt.subplots(len(choices),5,figsize=(12,2*len(choices)))
    for row,i in enumerate(choices):
        for col,arm in enumerate(('truth',)+ARMS):
            ax=axes[row,col];ax.imshow(data['target'][i],cmap='gray',vmin=0,vmax=1,extent=(0,64,64,0))
            boxes=data['meta'][i]['truth_boxes'] if arm=='truth' else predictions[arm][i]['boxes_xyxy']
            for x0,y0,x1,y1 in boxes:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime' if arm=='truth' else 'orange'))
            ax.set_title(data['meta'][i]['source_class']+' / '+arm,fontsize=7);ax.axis('off')
    fig.tight_layout();fig.savefig(root/'examples.png',dpi=140);plt.close(fig)
    visibility=read(root/'visibility/report.json')
    lines=['# Five-step detector development campaign','',
        f'Primary approach selected on validation before confirmation: **{frozen["primary"]}**.',
        '', '## 1. Error and visibility audit','',
        'The audit used original development images. `error_audit.json` separates negligible overlap, inaccurate box extent and acceptable extent; official matching remains one-to-one at IoU >= 0.5.',
        f'The paired replay covered {visibility["sequences"]} training sequences and {visibility["crops"]} positive target crops. Geometry was reproduced before comparing noiseless defect-on and defect-off images. Noise-relative visibility is a descriptive proxy; no weak labels were removed.',
        '', '| Source class | Replay crops | Median fraction of mask above twice noise standard deviation |','|---|---:|---:|']
    for c in sorted({r['source_class'] for r in visibility['rows']}):
        rows=[r for r in visibility['rows'] if r['source_class']==c]
        lines.append(f'| {c} | {len(rows)} | {np.median([r["fraction_mask_change_above_2_noise_std"] for r in rows]):.3f} |')
    lines+=['','## 2. Local background and context variability','',
        'Thresholding tests local Gaussian backgrounds after row-profile subtraction. Median context corrects smooth profile differences in each neighbour before taking the median. Both divide residuals by a spatial variation map estimated from clean training windows. The map is frozen; inference reads no labels.',
        'The optional four-channel hybrid receives target, adjusted reference, absolute difference and a bounded noise-relative residual. The original three-channel hybrid remains a candidate.',
        '', '## 3. Nested training-size experiment','',
        '| Multiplier | Sequences | Targets | Positive targets |','|---|---:|---:|---:|']
    for k,v in population.items():lines.append(f'| {k}x | {v["sequences"]} | {v["targets"]} | {v["positive_targets"]} |')
    lines+=['','The original 210 validation targets are shared across all runs. The added datasets were designated training before generation; their local generator split labels are retained only as provenance. Original validation/test and all confirmation campaigns are excluded from training.',
        'Each data-size condition uses the same three seeds and 20 epochs. This holds epochs fixed, not optimizer steps or compute. The experiment therefore measures the practical effect of more data with proportionally more training, not a compute-matched causal data-only effect.',
        '', '| Approach | Size | Mean validation F1 across three seeds | Range |','|---|---:|---:|---:|']
    for a in ('cnn','cnn_median'):
        for n in (1,2,4):
            values=[r['metrics']['box_f1'] for r in curves if r['arm']==a and r['multiplier']==n]
            lines.append(f'| {a} | {n}x | {np.mean(values):.3f} | {min(values):.3f} to {max(values):.3f} |')
    lines+=['','![Learning curve](learning_curve.png)','','## 4. Box extent supervision','',
        'Where the prespecified validation error rule was met, a second output was trained on filled ground-truth rectangles, with loss weight 0.25. Validation chose whether this output or the segmentation output produced boxes. Pixel IoU always uses the segmentation output. Ground truth and the IoU acceptance rule were unchanged.',
        'This compact experiment adds a rectangle raster output to the existing network, not a separate instance detector. Touching predicted rectangles can still merge.',
        '', 'Selected models:','']
    for a,s in frozen['models'].items():
        lines.append(f'- {a}: '+(f'historical {s["origin"]} settings.' if s['implementation']=='historical' else
            f'{s["implementation"]}; variant {s.get("variant","local residual")}; size {s.get("multiplier","4") }x; rectangle output {s.get("rectangle",False)}; selected box channel {s.get("box_channel",0)}.'))
    lines+=['','## 5. Frozen independent confirmation','',
        'All three versions below were evaluated on the same fresh 70 sequences / 350 targets. Historical confirmation images were not reused. Actual seeds and raw hashes were audited against all prior datasets and expansion batches.',
        '', '| Method | First campaign F1 | Second campaign F1 | Selected F1 | Precision | Recall | TP / truth | Clean false alarms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        r=m[a];lines.append(f'| {a} | {m["original_"+a]["box_f1"]:.3f} | {m["previous_"+a]["box_f1"]:.3f} | {r["box_f1"]:.3f} | {r["box_precision"]:.3f} | {r["box_recall"]:.3f} | {r["tp"]}/{r["tp"]+r["fn"]} | {r["clean_false_alarms"]}/{r["clean_targets"]} |')
    lines+=['','![Confirmation comparison](comparison.png)','','Paired 95% sequence-bootstrap intervals for selected minus second-campaign box F1:','']
    for a in ARMS:
        lo,hi=ci['paired_difference_intervals'][a+'_minus_previous_'+a]['box_f1']
        lines.append(f'- {a}: [{lo:.3f}, {hi:.3f}], '+('positive improvement supported.' if lo>0 else 'no positive improvement established.' if hi>=0 else 'negative difference supported.'))
    lines+=['','Intervals use 2,000 paired sequence-cluster draws and are exploratory without multiplicity adjustment. All-zero bootstrap intervals for a detector with no successes do not prove zero population performance.',
        f'The prespecified primary image/pixel skill gate passed: **{results["synthetic_pixel_image_skill_gate"]}**. This requires balanced-accuracy lower bound above 0.5 and positive pixel-IoU differences over both simple baselines. It does not establish reliable box localisation.',
        '', '![Examples](examples.png)','','Examples show the largest labelled crop per source class, selected without consulting detector scores. Full per-class metrics and shuffled-context controls are in `results.json`.',
        '', '## Reproducibility and scope','',
        '`protocol.json`, `source_snapshot.zip`, dataset inventories, `training_membership.json`, `model_candidates.json` and `frozen_models.json` bind inputs, searches and selected settings. `verification.json` records hash checks and label-free inference agreement. The two historical campaigns remain unchanged.',
        'Constant B remains assumed. These experiments concern procedural straight-weld synthetic TOPO at 60 keV and 0.05 mm/pixel. They provide no new physical qualification, circular-weld evidence or measured SEM transfer evidence.',
        '', 'Run or resume from the repository root:','','```powershell',
        "& '.\\.venv\\Scripts\\python.exe' '.\\WeldDefectDetection\\develop_boxes.py' --stage all",'```','',
        'Run frozen label-free inference:','','```powershell',
        "& '.\\.venv\\Scripts\\python.exe' '.\\WeldDefectDetection\\infer_developed_boxes.py' --approach cnn `",
        f"  --input '{root / 'inference_smoke/input.npz'}' `",f"  --output '{root / 'example_boxes.json'}'",'```','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Manifest is rebuilt only after all deliverables have been written.
    files=inventory(root);files.pop(str((root/'FINAL_MANIFEST.json').resolve()),None)
    write(root/'FINAL_MANIFEST.json',dict(files=files))
