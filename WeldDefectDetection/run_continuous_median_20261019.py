"""Prospective circular-median ablation, isolated from frozen October 6 artifacts."""
from pathlib import Path
from dataclasses import replace, asdict
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from raised_acquisition import campaign as original, learning as original_learning, assessment
from raised_crown import learning as base, rendering as render, campaign as crown
from circular_study import metrics

OUT=ROOT/'DataSimulation/data/output/continuous_median_20261019_v1'
SPACINGS=(16,32,64)
THRESHOLDS=(3,4,6,8,12,20,40)
KINDS=('clean','open_keyhole','open_intrinsic','bulge')
BASE_SEED=2451900000
COUNT=24
read=original.read
write=original.write
sha=original.sha
cv2.setNumThreads(1)


def profile(image):
    """Continuous target profile; 65 angular samples then nine radial rows."""
    row=median_filter(image,size=(1,65),mode='wrap')
    return gaussian_filter1d(row,2.,axis=0,mode='reflect',truncate=2.).astype('f4')


def difference(image,spacing,corrected=False,profile_image=None):
    trend=(profile(image) if profile_image is None else profile_image) if corrected else 0.
    local=image-trend
    stack=np.stack([np.roll(local,k*spacing,axis=1) for k in (-4,-3,-2,-1,1,2,3,4)])
    reference=np.median(stack,axis=0)+trend
    return (image-reference).astype('f4')


def smooth(residual,sigma):
    if not sigma:return residual
    along=gaussian_filter1d(residual,sigma,axis=-1,mode='wrap',truncate=4.)
    return gaussian_filter1d(along,sigma,axis=-2,mode='reflect',truncate=4.).astype('f4')


def self_checks():
    rng=np.random.default_rng(191019)
    a=rng.random((40,320),dtype=np.float32)
    errors={}
    for corrected in (False,True):
        for sigma in (0,1):
            d=smooth(difference(a,16,corrected),sigma)
            moved=smooth(difference(np.roll(a,7,axis=1),16,corrected),sigma)
            error=float(np.max(abs(np.roll(d,7,axis=1)-moved)))
            assert error<1e-6
            errors[f'profile{corrected}_sigma{sigma}']=error
    flat=np.broadcast_to(np.linspace(0,1,40,dtype='f4')[:,None],(40,320)).copy()
    assert np.max(abs(difference(flat,16)))==0
    impulse=flat.copy();impulse[20,0]+=1
    assert abs(float(difference(impulse,16)[20,0])-1)<1e-6
    manual=np.median([a[8,(23-k*16)%320] for k in (-4,-3,-2,-1,1,2,3,4)])
    assert abs(difference(a,16)[8,23]-(a[8,23]-manual))<1e-7
    seam=np.zeros((20,80),'u1');seam[7:12,:3]=1;seam[7:12,-3:]=1
    assert len(metrics.regions(seam)[1])==1
    return dict(status='passed',circular_shift_max_errors=errors,target_excluded=True,
                constant_profile_cancels=True,periodic_region_connected=True,manual_median_matches=True)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    original.verify();original_learning.verify_freeze()
    historical={str(p):sha(p) for p in (original.OUT/'frozen_models.json',original.OUT/'normalization.json',
                                      original.OUT/'results.json',original.OUT/'dataset_manifest.json')}
    prior_seeds=set()
    for split in original.COUNTS:
        prior_seeds.update(x['seed'] for x in read(original.OUT/f'datasets/{split}/index.json'))
    for split in crown.COUNTS:
        prior_seeds.update(x['seed'] for x in read(crown.OUT/f'datasets/{split}/index.json'))
    assert not prior_seeds.intersection(range(BASE_SEED,BASE_SEED+COUNT))
    protocol=dict(campaign=OUT.name,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        aim='Evaluate continuous four-before/four-after median without changing renderer, noise or defect strength',
        development_source=str(original.OUT),development_splits=['train','val'],
        forbidden_selection_data=['prior test','prior stress','new confirmation'],
        primary='Box F1 at IoU >= 0.5 on clean + open_keyhole + open_intrinsic; ties pixel IoU then fewer FP',
        secondary=['all four conditions box F1','per-class recall','pixel IoU','clean-ring alarm rate'],
        spacings_px=list(SPACINGS),neighbour_offsets='[-4d,-3d,-2d,-d,+d,+2d,+3d,+4d], target excluded',
        step_px=1,profile_correction=[False,True],residual_sigma_px=[0,1],
        profile_rule='65-column running row median, Gaussian across 9 radial rows sigma=2; circular angular wrap',
        calibration='Per radial row: centre=median; scale=max(0.003,1.4826*MAD) over clean training residuals, every fourth angular pixel',
        normalization='Unchanged frozen train-only square-root normalization from original campaign',
        thresholds=list(THRESHOLDS),minimum_areas=[3,15],closing_sizes=[1,3],
        selection='Select one spacing/operating point per ablation arm, then primary winner, on validation only',
        confirmation=dict(backgrounds=COUNT,scenes=COUNT*len(KINDS),seed_base=BASE_SEED,
                          classes=list(KINDS),dwell_edges_us=list(original.EDGES),backgrounds_per_bin=6,
                          generation_gate='Only after selection freeze',coordinates='declared nominal ring'),
        inference='No truth inputs; masks used only for validation selection and post-freeze evaluation',
        uncertainty='2000 paired background bootstrap replicates stratified by dwell bin',
        previous_artifact_hashes=historical,physical_validation=False)
    p=OUT/'protocol.json'
    if not p.exists():write(p,protocol)
    else:
        saved=read(p);protocol['created_utc']=saved['created_utc'];assert saved==protocol
    write(OUT/'self_checks.json',self_checks())
    print('Protocol and numerical checks saved',flush=True)


def load_development():
    norm=read(original.OUT/'normalization.json')
    clean=[r for r in read(original.OUT/'datasets/train/index.json') if r['source_class']=='clean']
    valrows=read(original.OUT/'datasets/val/index.json')
    def images(rows):
        out=[]
        for row in rows:
            assert sha(row['npz_path'])==row['npz_sha256']
            with np.load(row['npz_path']) as z:out.append(base.normalize(z['raw'][0],norm))
        return np.stack(out)
    train=images(clean); val=images(valrows)
    masks=[]
    for row in valrows:
        with np.load(row['npz_path']) as z:masks.append(z['mask'][16:144])
    data=dict(mask=np.stack(masks),meta=valrows)
    data['truth']=[metrics.regions(m)[1] for m in data['mask']]
    return train,val,data


def subset(rows):return [x for x in rows if x['source_class']!='bulge']


def summarize(rows,intervals=False):
    out=dict(primary_pores=metrics.summary(subset(rows)),all_conditions=metrics.summary(rows),
             by_class={k:metrics.summary([x for x in rows if x['source_class']==k]) for k in KINDS})
    if intervals:
        out['primary_intervals']=assessment.grouped_intervals(subset(rows))
        out['all_intervals']=assessment.grouped_intervals(rows)
    return out


def score_rows(data,scores,op):
    rows=metrics.evaluate(data,scores,op,retain=True)
    for row,meta in zip(rows,data['meta']):
        row.update(dwell_bin=meta['dwell_bin'],dwell_time_us=meta['dwell_time_us'])
    return rows


def selection_key(result):return metrics.key(result['summary']['primary_pores'])


def develop():
    if (OUT/'frozen.json').exists():verify_frozen();return
    assert not (OUT/'confirmation').exists(),'Fresh confirmation cannot precede freeze'
    train,val,data=load_development()
    folder=OUT/'development';folder.mkdir(exist_ok=True)
    print('Loaded 24 clean training and 96 validation scenes',flush=True)
    for corrected in (False,True):
        needed=any(not (folder/f'profile{int(corrected)}_sigma{sigma}_d{d}.json').exists()
                   for d in SPACINGS for sigma in (0,1))
        if not needed:continue
        tp=np.stack([profile(x) for x in train]) if corrected else None
        vp=np.stack([profile(x) for x in val]) if corrected else None
        for spacing in SPACINGS:
            if all((folder/f'profile{int(corrected)}_sigma{s}_d{spacing}.json').exists() for s in (0,1)):continue
            print('Residuals',dict(profile=corrected,spacing=spacing),flush=True)
            tr=np.stack([difference(x,spacing,corrected,None if tp is None else tp[i]) for i,x in enumerate(train)])
            vr=np.stack([difference(x,spacing,corrected,None if vp is None else vp[i]) for i,x in enumerate(val)])
            for sigma in (0,1):
                name=f'profile{int(corrected)}_sigma{sigma}_d{spacing}';path=folder/(name+'.json')
                if path.exists():continue
                started=time.monotonic()
                trainres=smooth(tr,sigma)[:,16:144,::4]
                pool=trainres.transpose(1,0,2).reshape(128,-1)
                mu=np.median(pool,axis=1).astype('f4')
                scale=np.maximum(.003,1.4826*np.median(abs(pool-mu[:,None]),axis=1)).astype('f4')
                scores=abs(smooth(vr,sigma)[:,16:144]-mu[None,:,None])/scale[None,:,None]
                trials=[]
                for threshold in THRESHOLDS:
                    for closing in (1,3):
                        for area in (3,15):
                            op=dict(threshold=threshold,min_area=area,closing=closing)
                            rows=score_rows(data,scores,op)
                            trials.append(dict(op=op,summary=summarize(rows)))
                    print(name,'threshold',threshold,'complete',flush=True)
                best=max(trials,key=selection_key)
                bestrows=score_rows(data,scores,best['op'])
                result=dict(name=name,profile=corrected,sigma=sigma,spacing=spacing,
                    centre=mu.tolist(),scale=scale.tolist(),best=best,trials=trials,
                    validation_rows=bestrows,seconds=time.monotonic()-started)
                write(path,result)
                print('VALIDATION',name,best['summary']['primary_pores'],flush=True)
                del scores,trainres,pool;gc.collect()
            del tr,vr;gc.collect()
        del tp,vp;gc.collect()
    candidates=[read(folder/f'profile{int(p)}_sigma{s}_d{d}.json') for p in (False,True) for s in (0,1) for d in SPACINGS]
    arms={f'profile{int(p)}_sigma{s}':max([x for x in candidates if x['profile']==p and x['sigma']==s],key=lambda x:selection_key(x['best'])) for p in (False,True) for s in (0,1)}
    selected=max(arms,key=lambda k:selection_key(arms[k]['best']))
    sourcepaths=original.generation_sources()+[Path(__file__),Path(base.__file__),Path(metrics.__file__),Path(assessment.__file__)]
    files=list(set(sourcepaths))+list(folder.glob('*.json'))+[OUT/'protocol.json',OUT/'self_checks.json',original.OUT/'normalization.json',original.OUT/'frozen_models.json']
    for split in ('train','val'):
        idx=original.OUT/f'datasets/{split}/index.json';files.append(idx)
        files.extend(Path(x['npz_path']) for x in read(idx))
    freeze=dict(selected_arm=selected,arms=arms,baseline=read(original.OUT/'frozen_models.json')['median'],
                frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files={str(p):sha(p) for p in files})
    with zipfile.ZipFile(OUT/'sources_at_freeze.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in set(sourcepaths):archive.write(path,path.relative_to(ROOT))
    write(OUT/'frozen.json',freeze)
    print('FROZEN',selected,arms[selected]['name'],arms[selected]['best'],flush=True)


def verify_frozen():
    f=read(OUT/'frozen.json')
    for path,digest in f['files'].items():assert sha(path)==digest,f'Frozen input changed: {path}'
    return f


def new_dwell(index):
    b=index%4;rank=index//4;rng=np.random.default_rng(BASE_SEED+index+701)
    q=(rank+rng.uniform(.1,.9))/6
    return float(np.exp(np.log(original.EDGES[b])+q*np.log(original.EDGES[b+1]/original.EDGES[b]))),b


def background(index):
    frozen=verify_frozen();folder=OUT/'confirmation';folder.mkdir(exist_ok=True)
    seed=BASE_SEED+index;dwell,bin_id=new_dwell(index)
    cfg=render.old.load_config_bundle(ROOT/'DataSimulation/config')
    sample,_=render.oldg.baseline.c.sampled(cfg,seed,index)
    sample=replace(sample,beam=replace(sample.beam,dwell_time_us=dwell))
    amplitude=(0.,.05,.1)[(index//3+index)%3];engine=None;records=[]
    for kind in KINDS:
        sid=f'fresh-{index:03d}-{kind}';path=folder/(sid+'.npz');mp=path.with_suffix('.json')
        if mp.exists():
            row=read(mp);assert sha(path)==row['npz_sha256'];records.append(row);continue
        if engine is None:engine=render.Engine(seed,amplitude)
        started=time.monotonic();changed=replace(sample,defects=crown.plans(seed,index)[kind])
        arrays,meta=render.render(engine,changed)
        assert bool(arrays['mask'].any())==(kind!='clean')
        np.savez_compressed(path,**arrays)
        row=dict(scene_id=sid,background_id=f'fresh-{index:03d}',split='confirmation',seed=seed,index=index,
                 source_class=kind,amplitude=amplitude,dwell_bin=bin_id,dwell_time_us=dwell,
                 parameters=asdict(changed),metadata=meta,npz_path=str(path),npz_sha256=sha(path),
                 freeze_sha256=sha(OUT/'frozen.json'),seconds=time.monotonic()-started)
        write(mp,row);records.append(row);print('GENERATED',sid,round(row['seconds'],1),flush=True)
        del arrays;gc.collect()
    write(folder/f'background_{index:03d}.json',records)


def generate(workers=3):
    verify_frozen();folder=OUT/'confirmation';folder.mkdir(exist_ok=True)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    pending=list(range(COUNT));active=[]
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
    rows=[r for i in range(COUNT) for r in read(folder/f'background_{i:03d}.json')]
    write(folder/'index.json',rows)


def evaluate():
    frozen=verify_frozen();metas=read(OUT/'confirmation/index.json');norm=read(original.OUT/'normalization.json')
    folder=OUT/'predictions';folder.mkdir(exist_ok=True)
    allrows={k:[] for k in ['baseline',*frozen['arms']]}
    for meta in metas:
        receipt=folder/(meta['scene_id']+'.json')
        if receipt.exists():
            saved=read(receipt);assert saved['freeze_sha256']==sha(OUT/'frozen.json')
            for k,v in saved['rows'].items():allrows[k].append(v)
            continue
        assert sha(meta['npz_path'])==meta['npz_sha256']
        with np.load(meta['npz_path']) as z:image=base.normalize(z['raw'][0],norm);mask=z['mask'][16:144]
        data=dict(mask=mask[None],meta=[meta]);rows={};trend=profile(image);cache={}
        for method in allrows:
            if method=='baseline':
                s=base.median_image(image,frozen['baseline']['spec']);op=frozen['baseline']['selection']['best']
            else:
                arm=frozen['arms'][method];key=(arm['spacing'],arm['profile'])
                if key not in cache:cache[key]=difference(image,arm['spacing'],arm['profile'],trend)
                res=smooth(cache[key],arm['sigma'])[16:144]
                s=abs(res-np.array(arm['centre'],'f4')[:,None])/np.array(arm['scale'],'f4')[:,None];op=arm['best']['op']
            row=score_rows(data,s[None],op)[0];allrows[method].append(row);rows[method]=row
        write(receipt,dict(freeze_sha256=sha(OUT/'frozen.json'),source_sha256=meta['npz_sha256'],rows=rows))
        print('SCORED',meta['scene_id'],flush=True)
    results={k:dict(**summarize(v,True),rows=v) for k,v in allrows.items()}
    chosen=frozen['selected_arm']
    comparisons={k:dict(primary=assessment.paired_difference(subset(v),subset(allrows['baseline'])),
                        all_conditions=assessment.paired_difference(v,allrows['baseline'])) for k,v in allrows.items() if k!='baseline'}
    write(OUT/'results.json',dict(selected_arm=chosen,results=results,paired_differences_vs_baseline=comparisons,
                                physical_validation=False,freeze_sha256=sha(OUT/'frozen.json')))
    verify_frozen()
    for p,h in read(OUT/'protocol.json')['previous_artifact_hashes'].items():assert sha(p)==h
    prior={x['seed'] for split in original.COUNTS for x in read(original.OUT/f'datasets/{split}/index.json')}
    assert not prior.intersection(x['seed'] for x in metas)
    assert len(metas)==96 and len({x['background_id'] for x in metas})==24
    write(OUT/'audit.json',dict(status='passed',scenes=len(metas),backgrounds=24,prior_seed_overlap=False,
        frozen_inputs_unchanged=True,historical_artifacts_unchanged=True,selection_preceded_confirmation=True,
        numerical_checks=read(OUT/'self_checks.json'),physical_validation=False))
    print('FINAL',chosen,results[chosen]['primary_pores'],comparisons[chosen]['primary'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','prepare','develop','generate','background','evaluate'])
    p.add_argument('--index',type=int);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    if args.stage in ('all','prepare'):prepare()
    if args.stage in ('all','develop'):develop()
    if args.stage in ('all','generate'):generate(args.workers)
    if args.stage=='background':background(args.index)
    if args.stage in ('all','evaluate'):evaluate()


if __name__=='__main__':main()
