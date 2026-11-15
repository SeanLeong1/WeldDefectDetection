"""Supplementary fixed-rule diagnostics after frozen confirmation, with no tuning."""
from pathlib import Path
import argparse
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
import numpy as np
from box_comparison.campaign import read
from box_comparison.core import write,match_boxes
from box_development.campaign import DEFAULT,ARMS,check,require
from box_development.data import load


def build(root):
    check(root);require(root,'evaluate')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    data=load(root/'datasets/confirmation','confirmation')
    all_rows={a:read(root/'predictions'/f'{a}.json') for a in ARMS}
    diagnostics={};error_examples=[]
    for arm,rows in all_rows.items():
        diagnostics[arm]=dict(matches_by_diagnostic_iou={str(t):sum(match_boxes(r['boxes_xyxy'],r['truth_boxes'],t)['tp'] for r in rows)
            for t in (.1,.25,.5)},truth_regions=sum(len(r['truth_boxes']) for r in rows))
        # Recall by box size keeps matching against the full truth set.
        bins={'small_under_64':[],'medium_64_to_255':[],'large_256_plus':[]}
        from scipy.optimize import linear_sum_assignment
        from box_comparison.core import box_iou
        for r in rows:
            boxes,truth=r['boxes_xyxy'],r['truth_boxes'];matched=set()
            if boxes and truth:
                iou=np.array([[box_iou(p,t) for t in truth] for p in boxes]);eligible=iou>=.5
                pi,ti=linear_sum_assignment(-(eligible*(min(len(boxes),len(truth))+1+iou)))
                matched={int(j) for i,j in zip(pi,ti) if eligible[i,j]}
            for i,(x0,y0,x1,y1) in enumerate(truth):
                area=(x1-x0)*(y1-y0);key='small_under_64' if area<64 else 'medium_64_to_255' if area<256 else 'large_256_plus'
                bins[key].append(i in matched)
        diagnostics[arm]['truth_box_size_recall']={k:dict(regions=len(v),matched=sum(v),recall=sum(v)/len(v) if v else None) for k,v in bins.items()}
        for kind,key in (('false alarm','fp'),('miss','fn')):
            index=next((i for i,r in enumerate(rows) if r[key]>0),None)
            if index is not None:error_examples.append((arm,kind,index))
    fig,axes=plt.subplots(4,2,figsize=(8,14));axes=axes.ravel()
    for ax,(arm,kind,i) in zip(axes,error_examples):
        ax.imshow(data['target'][i],cmap='gray',vmin=0,vmax=1,extent=(0,64,64,0))
        for boxes,color in ((data['meta'][i]['truth_boxes'],'lime'),(all_rows[arm][i]['boxes_xyxy'],'orange')):
            for x0,y0,x1,y1 in boxes:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color))
        ax.set_title(f'{arm}: first {kind}\n{data["meta"][i]["window_id"]}',fontsize=7);ax.axis('off')
    for ax in axes[len(error_examples):]:ax.axis('off')
    fig.suptitle('Fixed-order errors: green truth, orange prediction');fig.tight_layout();fig.savefig(root/'error_examples.png',dpi=150);plt.close(fig)
    write(root/'localization_diagnostics.json',dict(official_iou=.5,diagnostic_only_lower_thresholds=True,methods=diagnostics))
    write(root/'error_examples.json',[dict(arm=a,kind=k,window_id=data['meta'][i]['window_id']) for a,k,i in error_examples])
    retries={}
    for name in ('train_1','train_2','train_3','confirmation'):
        seq=read(root/'datasets'/name/'sequence_manifest.json')
        retries[name]={}
        for c in sorted({r['planned_class'] for r in seq}):
            rows=[r for r in seq if r['planned_class']==c]
            retries[name][c]=dict(sequences=len(rows),retried_sequences=sum(r['generation_attempt']>1 for r in rows),
                discarded_attempts=sum(r['generation_attempt']-1 for r in rows),maximum_attempt=max(r['generation_attempt'] for r in rows))
    write(root/'generation_retries_by_class.json',retries)
    from box_development.campaign import DEV
    norm=read(DEV/'topo_normalization.json');clipping={}
    for name in ('train_1','train_2','train_3','confirmation'):
        below=above=pixels=0
        for path in (root/'datasets'/name).rglob('*.npy'):
            raw=np.load(path,allow_pickle=False)
            below+=int(np.sum(raw<norm['raw_low']));above+=int(np.sum(raw>norm['raw_high']));pixels+=raw.size
        clipping[name]=dict(raw_pixels=pixels,below_frozen_low_fraction=below/pixels,
                            above_frozen_high_fraction=above/pixels)
    write(root/'normalization_clipping.json',clipping)
    visibility=read(root/'visibility/report.json');examples=[]
    for label in sorted({r['source_class'] for r in visibility['rows']}):
        examples.append(next(r for r in visibility['rows'] if r['source_class']==label))
    fig,axes=plt.subplots(6,4,figsize=(11,15))
    for i,r in enumerate(examples):
        pair=np.load(root/'visibility'/(r['sample_id']+'.npz'))
        low=min(pair['defect_on'].min(),pair['defect_off'].min());high=max(pair['defect_on'].max(),pair['defect_off'].max())
        for j,key in enumerate(('defect_on','defect_off','delta','mask')):
            options=dict(cmap='gray',vmin=low,vmax=high) if j<2 else dict(cmap='magma')
            axes[i,j].imshow(pair[key],**options);axes[i,j].set_title(r['source_class']+' / '+key,fontsize=7);axes[i,j].axis('off')
    fig.suptitle('First replay crop per class; on/off share intensity limits, delta is individually scaled')
    fig.tight_layout();fig.savefig(root/'visibility_examples.png',dpi=140);plt.close(fig)
    text=['# Supplementary diagnostics','',
        'These diagnostics were computed after frozen confirmation. They do not change model settings, labels or acceptance thresholds.',
        '', '![Fixed-order errors](error_examples.png)','','The first false positive and first miss per approach are shown. They are reproducible examples, not an estimate of error frequency.',
        '', '![Paired renderer replay](visibility_examples.png)','','The on/off images in each row share intensity limits. Each difference image uses its own range to show spatial structure, so difference brightness cannot be compared quantitatively across rows.',
        '', '| Method | Matches at IoU 0.1 | IoU 0.25 | Official IoU 0.5 | Truth regions |','|---|---:|---:|---:|---:|']
    for a,d in diagnostics.items():
        v=d['matches_by_diagnostic_iou'];text.append(f'| {a} | {v["0.1"]} | {v["0.25"]} | {v["0.5"]} | {d["truth_regions"]} |')
    text+=['','Lower-IoU values diagnose rough localization only. `localization_diagnostics.json` also reports recall by unchanged truth-box area.',
        '`generation_retries_by_class.json` reports discarded generation attempts for every source class. Existing generator validity retries remain in force; passing a nonempty-mask check does not establish strong visual evidence.',
        '`normalization_clipping.json` records raw pixels outside the frozen normalization limits in every new dataset. The normalization is never refitted on confirmation images.']
    (root/'DIAGNOSTICS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT)
    build(parser.parse_args().root)
