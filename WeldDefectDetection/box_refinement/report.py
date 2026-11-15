from pathlib import Path
import json
import numpy as np
from box_comparison.core import load_windows,sha,write
from box_comparison.campaign import read,verify_files


def build(root):
    from .campaign import check,require,ARMS,PREVIOUS
    check(root);require(root,'evaluate')
    if read(root/'verification.json')['status']!='ok':raise ValueError('Complete verification first.')
    frozen=read(root/'frozen_models.json');results=read(root/'results.json');metrics=results['metrics'];ci=results['uncertainty']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(4)
    for ax,key in zip(axes,('box_f1','pixel_iou')):
        ax.bar(x-.18,[metrics['previous_'+a][key] for a in ARMS],.36,label='Previous frozen')
        ax.bar(x+.18,[metrics[a][key] for a in ARMS],.36,label='Selected refinement')
        ax.set_xticks(x,ARMS,rotation=15);ax.set_ylim(0,1);ax.set_ylabel(key);ax.legend()
    fig.suptitle('Same fresh confirmation targets for both versions');fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    data=load_windows(root/'confirmation');classes=sorted({m['source_class'] for m in data['meta']})
    rows={a:read(root/'predictions'/f'{a}.json') for a in ARMS}
    fig,axes=plt.subplots(len(classes),5,figsize=(12,2.2*len(classes)))
    for i,label in enumerate(classes):
        indices=[k for k,m in enumerate(data['meta']) if m['source_class']==label]
        k=max(indices,key=lambda v:int(data['mask'][v].sum()))
        for j,arm in enumerate(('truth',)+ARMS):
            ax=axes[i,j];ax.imshow(data['target'][k],cmap='gray',vmin=0,vmax=1,extent=(0,64,64,0))
            boxes=data['meta'][k]['truth_boxes'] if arm=='truth' else rows[arm][k]['boxes_xyxy']
            for x0,y0,x1,y1 in boxes:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime' if arm=='truth' else 'orange',lw=1))
            ax.set_title(f'{label}\n{arm}',fontsize=8);ax.axis('off')
    fig.suptitle('Largest labelled target per source class, chosen without model scores');fig.tight_layout();fig.savefig(root/'examples.png',dpi=140);plt.close(fig)
    figures=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,arm in zip(axes,('cnn','cnn_median')):
        for p in sorted((root/'training').glob(f'{arm}_9216*.json')):
            r=read(p);h=[e for e in r['history'] if 'metrics' in e]
            ax.plot([e['epoch'] for e in h],[e['metrics']['box_f1'] for e in h],label=str(r['seed']))
        ax.set_title(arm);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
    axes[0].set_ylabel('Validation box F1');fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    text=['# Second development campaign: TOPO bounding boxes','',
        f'Primary method selected before new confirmation: **{frozen["primary"]}**. All selected parameters and weights were frozen before generation.',
        '', 'The table evaluates the previous frozen models and the selected refinements on exactly the same 350 fresh target crops from 70 independent sequences. The renderer, ground-truth masks, normalization and IoU >= 0.5 matching rule are unchanged.', '',
        '| Approach | Previous box F1 | Refined box F1 [95% interval] | Refined precision | Refined recall | Correct / labelled regions | Clean false alarms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        m=metrics[a];lo,hi=ci['intervals'][a]['box_f1']
        text.append(f'| {a} | {metrics["previous_"+a]["box_f1"]:.3f} | {m["box_f1"]:.3f} [{lo:.3f}, {hi:.3f}] | {m["box_precision"]:.3f} | {m["box_recall"]:.3f} | {m["tp"]}/{m["tp"]+m["fn"]} | {m["clean_false_alarms"]}/{m["clean_targets"]} |')
    text+=['','![Comparison](comparison.png)','','## Are the improvements supported?','',
        '| Paired comparison | Box F1 difference, 95% interval | Interpretation |','|---|---:|---|']
    for name,values in ci['paired_difference_intervals'].items():
        lo,hi=values['box_f1'];meaning='Positive difference supported' if lo>0 else ('Negative difference supported' if hi<0 else 'Interval includes zero')
        text.append(f'| {name} | [{lo:.3f}, {hi:.3f}] | {meaning} |')
    text+=['','Intervals use 2,000 paired sequence-cluster bootstrap draws. They are exploratory and not multiplicity adjusted. An all-zero interval with no observed successful boxes is a degenerate bootstrap result, not proof of exactly zero population performance.','',
        '## What changed','',
        '1. Thresholding subtracts each image row\'s median, suppressing the continuous weld band. The median method instead subtracts the row profile of the target-minus-context residual. Validation chooses Gaussian smoothing and whether to divide by a robust variation estimate. These operations use no labels at inference.',
        '2. Both CNNs use the existing small architecture with 40 epochs, equal expected sampling of positive and clean training targets, aligned horizontal flips, common mild gain/offset augmentation, AdamW and cosine learning-rate decay. Three seeds were trained per family.',
        '3. Validation chooses strong/weak threshold region growth, optional closing, minimum area and 0/2/4-pixel box padding. Padding changes only predicted rectangles; pixel-mask IoU is evaluated on the actual detected mask.',
        '4. The old model remains a candidate for every approach. The old CNN weights with revised box extraction are also a separate candidate, so training gains are distinguished from postprocessing gains.',
        '', '## Selected implementations','']
    for a,s in frozen['models'].items():
        text.append(f'- **{a}:** {s.get("training_origin",s["implementation"])}; parameters `{json.dumps(s["parameters"],sort_keys=True)}`'+(f'; seed {s["seed"]}, epoch {s["epoch"]}.' if 'seed' in s else '.'))
    text+=['','The selection histories and prior-weight retuning comparisons are in `training/`. Validation data were reused for this explicit new development cycle; previous confirmation images were not used to tune these settings.','',
        '![Training curves](training_curves.png)','','## Class results','',
        '| Sequence source class | Thresholding F1 | CNN F1 | Median F1 | CNN + median F1 |','|---|---:|---:|---:|---:|']
    for c in classes:
        if c!='clean':text.append('| '+c+' | '+' | '.join(f'{results["per_class"][a][c]["box_f1"]:.3f}' for a in ARMS)+' |')
    text+=['','Some sequences contain a second injected defect. These groups are source classes, not exclusive physical-instance classifications. Weak or small labelled regions remain in evaluation.','',
        '![Examples](examples.png)','','## Reproducibility and limits','',
        '- `protocol.json` records the hypotheses and bounded search before the new confirmation.',
        '- `source_snapshot.zip` archives the implementation; completed stages and datasets are hash-verified.',
        '- `frozen_models.json` preserves preprocessing, selected weights and box rules.',
        '- `confirmation_audit.json` checks disjoint actual seeds/content against development and both earlier confirmation datasets.',
        '- `predictions/` and `results.json` contain case-level and aggregate comparisons.',
        '- `inference_smoke/` contains label-free raw inputs and verified outputs for all four approaches.',
        '- The original campaign and checkpoints remain preserved.',
        '', 'This remains a synthetic straight-weld TOPO experiment using assumed constant B. It does not validate real SEM transfer or establish reliability on all defect classes. No settings were changed after viewing the fresh confirmation.',
        '', 'Run from the repository root:', '', '```powershell',
        "& '.\\.venv\\Scripts\\python.exe' '.\\WeldDefectDetection\\refine_boxes.py' --stage all",'```','',
        'The matching inference command is `WeldDefectDetection/infer_refined_boxes.py`, with `--approach`, `--input` and `--output`. Its input schema matches the original label-free inference command.','']
    (root/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    files=[p for p in root.rglob('*') if p.is_file() and p.name!='FINAL_MANIFEST.json']
    write(root/'FINAL_MANIFEST.json',dict(files={str(p.resolve()):sha(p) for p in files},physical_qualification=False))
    verify_files(read(PREVIOUS/'FINAL_MANIFEST.json')['files'])
