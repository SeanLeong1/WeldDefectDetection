"""Frozen median threshold sensitivity and equal-criterion CNN comparison."""
from pathlib import Path
import argparse
import gc
import json
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import run_continuous_median_20261019 as previous
from raised_acquisition import campaign as original, learning, assessment
from raised_crown import learning as base, campaign as crown
from circular_study import metrics

OUT=ROOT/'DataSimulation/data/output/median_thresholds_20261019_v1'
THRESHOLDS=(3,4,6,8,10,12)
IOUS=(.5,.25)
SEED_BASE=2461900000
COUNT=24
read,write,sha=original.read,original.write,original.sha
cv2.setNumThreads(1)
torch.set_num_threads(4)


def match_at(pred,truth,period,cutoff):
    tp=0;total=0.
    if pred and truth:
        ious=np.array([[metrics.circular_iou(p,t,period) for t in truth] for p in pred])
        ok=ious>=cutoff
        ii,jj=linear_sum_assignment(-(ok*(min(len(pred),len(truth))+1+ious)))
        pairs=[(i,j) for i,j in zip(ii,jj) if ok[i,j]]
        tp=len(pairs);total=float(sum(ious[i,j] for i,j in pairs))
    return dict(tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,matched_iou_sum=total)


def checks():
    rng=np.random.default_rng(19102026)
    for _ in range(100):
        boxes=[]
        for n in (int(rng.integers(0,8)),int(rng.integers(0,8))):
            starts=rng.integers(0,100,(n,2));sizes=rng.integers(1,30,(n,2))
            boxes.append(np.c_[starts,starts+sizes].tolist())
        assert match_at(*boxes,100,.5)==metrics.match(*boxes,100)
        assert match_at(*boxes,100,.25)['tp']>=match_at(*boxes,100,.5)['tp']
    p=[[0,0,3,10]];t=[[0,0,10,10]]
    assert match_at(p,t,100,.5)['tp']==0 and match_at(p,t,100,.25)['tp']==1
    assert match_at(p+p,t,100,.25)['tp']==1
    assert match_at([[98,0,103,10]],[[0,0,5,10]],100,.25)['tp']==1
    return dict(status='passed',strict_match_parity_cases=100,relaxed_match_monotonic=True,
        one_to_one_matching=True,periodic_matching=True,criterion_changes_counts_not_predictions=True)


def summary(rows,intervals=False):
    pores=[x for x in rows if x['source_class']!='bulge']
    out=dict(primary_pores=metrics.summary(pores),all_conditions=metrics.summary(rows),
        by_class={k:metrics.summary([x for x in rows if x['source_class']==k]) for k in previous.KINDS})
    if intervals:
        out['primary_intervals']=assessment.grouped_intervals(pores)
        out['all_intervals']=assessment.grouped_intervals(rows)
    return out


def rows_for(score,mask,meta,op,retain=False):
    pred,boxes=metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
    truth=metrics.regions(mask)[1]
    common=dict(scene_id=meta['scene_id'],background_id=meta['background_id'],source_class=meta['source_class'],
        amplitude=meta['amplitude'],dwell_bin=meta['dwell_bin'],dwell_time_us=meta['dwell_time_us'],
        intersection=int(((pred>0)&(mask>0)).sum()),union=int(((pred>0)|(mask>0)).sum()),
        predicted_pixels=int(pred.sum()),truth_pixels=int(mask.sum()),predicted_regions=len(boxes))
    if retain:common.update(predicted_boxes=boxes,truth_boxes=truth)
    return {f'{iou:.2f}':dict(common,**match_at(boxes,truth,score.shape[1],iou)) for iou in IOUS}


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    original.verify();learning.verify_freeze();previous.verify_frozen()
    paths=[original.OUT/'frozen_models.json',original.OUT/'results.json',original.OUT/'normalization.json',
           previous.OUT/'frozen.json',previous.OUT/'results.json']
    protocol=dict(campaign=OUT.name,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        aim='Fair existing-median threshold sensitivity and frozen CNN comparison; no deliberate weakening',
        detection_thresholds=list(THRESHOLDS),fixed_median_postprocessing=dict(min_area=15,closing=3),
        matching_iou=list(IOUS),strict_primary_iou=.5,
        selection='Maximum strict pore-task validation F1, ties pixel IoU then fewer FP; threshold 12 retained as option',
        primary_population='Clean + open_keyhole + open_intrinsic',secondary_population='All four conditions including bulge',
        cnn='Original validation-selected coverage and hard-negative CNNs with their original operating settings; no retraining',
        confirmation=dict(backgrounds=COUNT,scenes=96,seed_base=SEED_BASE,classes=list(previous.KINDS),
            coordinates='Declared nominal ring',gate='Generate only after freeze'),
        reporting='Report every prespecified threshold at both IoUs; never select a threshold using confirmation',
        uncertainty='2000 paired background bootstrap replicates stratified by dwell bin',
        historic_hashes={str(p):sha(p) for p in paths},physical_validation=False)
    if (OUT/'protocol.json').exists():
        old=read(OUT/'protocol.json');protocol['created_utc']=old['created_utc'];assert old==protocol
    else:write(OUT/'protocol.json',protocol)
    seeds=set()
    for source in (original,crown):
        for split in source.COUNTS:seeds.update(r['seed'] for r in read(source.OUT/f'datasets/{split}/index.json'))
    seeds.update(r['seed'] for r in read(previous.OUT/'confirmation/index.json'))
    assert not seeds.intersection(range(SEED_BASE,SEED_BASE+COUNT))
    write(OUT/'self_checks.json',checks())
    print('Protocol and matching checks passed',flush=True)


def develop():
    if (OUT/'frozen.json').exists():verify();return
    assert not (OUT/'confirmation').exists()
    frozen=learning.verify_freeze();norm=read(original.OUT/'normalization.json')
    folder=OUT/'validation';folder.mkdir(exist_ok=True)
    allrows={str(t):{f'{iou:.2f}':[] for iou in IOUS} for t in THRESHOLDS}
    index=read(original.OUT/'datasets/val/index.json')
    for meta in index:
        assert meta['split']=='val' and sha(meta['npz_path'])==meta['npz_sha256']
        path=folder/(meta['scene_id']+'.json')
        if path.exists():receipt=read(path)
        else:
            with np.load(meta['npz_path']) as z:image=base.normalize(z['raw'][0],norm);mask=z['mask'][16:144]
            score=base.median_image(image,frozen['median']['spec'])
            receipt={str(t):rows_for(score,mask,meta,dict(threshold=t,min_area=15,closing=3)) for t in THRESHOLDS}
            write(path,receipt)
        for t in allrows:
            for iou in allrows[t]:allrows[t][iou].append(receipt[t][iou])
        print('VALIDATED',meta['scene_id'],flush=True)
    trials={t:{iou:summary(rows) for iou,rows in vals.items()} for t,vals in allrows.items()}
    selected=max(THRESHOLDS,key=lambda t:metrics.key(trials[str(t)]['0.50']['primary_pores']))
    write(OUT/'validation_results.json',dict(selected_threshold=selected,trials=trials))
    sources=set(original.generation_sources()+[Path(__file__),Path(previous.__file__),Path(base.__file__),
        Path(metrics.__file__),Path(assessment.__file__),Path(learning.__file__)])
    sources.update(Path(p) for p in read(original.OUT/'cv_protocol.json')['source_hashes'])
    files=set(sources)|set(folder.glob('*.json'))|{OUT/'protocol.json',OUT/'self_checks.json',OUT/'validation_results.json',
        original.OUT/'frozen_models.json',original.OUT/'normalization.json',original.OUT/'datasets/val/index.json'}
    files.update(Path(r['npz_path']) for r in index)
    cnns={name:frozen['arms'][name]['cnn'] for name in original.ARMS}
    files.update(Path(s['checkpoint']) for s in cnns.values())
    with zipfile.ZipFile(OUT/'sources_at_freeze.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT))
    write(OUT/'frozen.json',dict(selected_threshold=selected,median=frozen['median'],cnns=cnns,
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files={str(p):sha(p) for p in files}))
    print('FROZEN threshold',selected,'VALIDATION',trials[str(selected)]['0.50'],flush=True)


def verify():
    f=read(OUT/'frozen.json')
    for p,h in f['files'].items():assert sha(p)==h,f'Frozen artifact changed: {p}'
    return f


def background(index):
    # Reuse the unchanged paired renderer and dwell sampling with a new namespace.
    previous.OUT=OUT;previous.BASE_SEED=SEED_BASE;previous.COUNT=COUNT;previous.verify_frozen=verify
    previous.background(index)


def generate(workers=3):
    verify();folder=OUT/'confirmation';folder.mkdir(exist_ok=True)
    logs=OUT/'logs';logs.mkdir(exist_ok=True);pending=list(range(COUNT));active=[]
    while pending or active:
        while pending and len(active)<workers:
            i=pending.pop(0);receipt=folder/f'background_{i:03d}.json'
            if receipt.exists():
                for row in read(receipt):assert sha(row['npz_path'])==row['npz_sha256']
                continue
            handle=(logs/f'background_{i:03d}.log').open('a',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,str(Path(__file__)),'background','--index',str(i)],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((proc,handle,i))
        for proc,handle,i in active[:]:
            if proc.poll() is not None:
                handle.close();active.remove((proc,handle,i))
                if proc.returncode:raise RuntimeError(f'Background {i} failed; see logs')
                print('BACKGROUND COMPLETE',i,flush=True)
        if active:time.sleep(2)
    write(folder/'index.json',[r for i in range(COUNT) for r in read(folder/f'background_{i:03d}.json')])


def evaluate():
    f=verify();norm=read(original.OUT/'normalization.json');index=read(OUT/'confirmation/index.json')
    models={k:base.model_for(v) for k,v in f['cnns'].items()}
    names=[f'median_{t}' for t in THRESHOLDS]+list(models)
    allrows={k:{f'{iou:.2f}':[] for iou in IOUS} for k in names}
    folder=OUT/'predictions';folder.mkdir(exist_ok=True)
    for meta in index:
        path=folder/(meta['scene_id']+'.json')
        assert sha(meta['npz_path'])==meta['npz_sha256']
        if path.exists():
            receipt=read(path);assert receipt['freeze_sha256']==sha(OUT/'frozen.json')
            assert receipt['source_sha256']==meta['npz_sha256']
        else:
            with np.load(meta['npz_path']) as z:image=base.normalize(z['raw'][0],norm);mask=z['mask'][16:144]
            rows={};score=base.median_image(image,f['median']['spec'])
            for t in THRESHOLDS:rows[f'median_{t}']=rows_for(score,mask,meta,dict(threshold=t,min_area=15,closing=3),True)
            for name,model in models.items():
                rows[name]=rows_for(base.predict_image(image,model),mask,meta,f['cnns'][name]['selection']['best'],True)
            receipt=dict(freeze_sha256=sha(OUT/'frozen.json'),source_sha256=meta['npz_sha256'],rows=rows)
            write(path,receipt)
        for name in names:
            for iou in allrows[name]:allrows[name][iou].append(receipt['rows'][name][iou])
        print('SCORED',meta['scene_id'],flush=True)
    results={k:{iou:dict(**summary(rows,True),rows=rows) for iou,rows in vals.items()} for k,vals in allrows.items()}
    comparisons={}
    chosen=f'median_{f["selected_threshold"]}'
    for label,a,b in [('selected_minus_original',chosen,'median_12'),('hard_cnn_minus_selected','hard_negative',chosen),
                      ('coverage_cnn_minus_selected','coverage',chosen)]:
        comparisons[label]={}
        for iou in ('0.50','0.25'):
            aa=allrows[a][iou];bb=allrows[b][iou]
            comparisons[label][iou]=dict(primary_pores=assessment.paired_difference(previous.subset(aa),previous.subset(bb)),
                all_conditions=assessment.paired_difference(aa,bb))
    write(OUT/'results.json',dict(selected_threshold=f['selected_threshold'],results=results,paired_differences=comparisons,
        freeze_sha256=sha(OUT/'frozen.json'),physical_validation=False))
    verify()
    for p,h in read(OUT/'protocol.json')['historic_hashes'].items():assert sha(p)==h
    assert len(index)==96 and len({r['background_id'] for r in index})==24
    assert {r['seed'] for r in index}==set(range(SEED_BASE,SEED_BASE+COUNT))
    for name,vals in allrows.items():
        for a,b in zip(vals['0.50'],vals['0.25']):
            assert a['predicted_boxes']==b['predicted_boxes'] and b['tp']>=a['tp']
        assert results[name]['0.50']['primary_pores']['tp']+results[name]['0.50']['primary_pores']['fn']==48
    write(OUT/'audit.json',dict(status='passed',scenes=96,backgrounds=24,seed_overlap=False,
        frozen_inputs_unchanged=True,historical_artifacts_unchanged=True,selection_preceded_confirmation=True,
        same_predictions_at_both_ious=True,numerical_checks=read(OUT/'self_checks.json'),physical_validation=False))
    print('FINAL',json.dumps({k:{iou:results[k][iou]['all_conditions'] for iou in ('0.50','0.25')} for k in ('median_12',chosen,'coverage','hard_negative')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','prepare','develop','generate','background','evaluate'])
    p.add_argument('--workers',type=int,default=3);p.add_argument('--index',type=int);args=p.parse_args()
    if args.stage in ('all','prepare'):prepare()
    if args.stage in ('all','develop'):develop()
    if args.stage in ('all','generate'):generate(args.workers)
    if args.stage=='background':background(args.index)
    if args.stage in ('all','evaluate'):evaluate()


if __name__=='__main__':main()
