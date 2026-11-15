"""Isolated median follow-up: validation selection, frozen fresh confirmation.

Never modifies the historical datasets, renderer, or trained CNN.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,argparse,itertools,subprocess,concurrent.futures,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from count_study import campaign as c,learning as learn
from raised_crown import learning as base
from circular_study import metrics
import median_localisation_20261110 as old
cv2.setNumThreads(1)
OUT=ROOT/'DataSimulation/data/output/median_exploration_20261115_v1'
REPORT=ROOT/'output/median_exploration_20261115'
OLD=c.OUT
SEED_BASE=5111500000
N_FRESH=48
METHODS=('original_static','original_adaptive','short_static','short_adaptive','gauss_static','gauss_adaptive')
THRESHOLDS=(4.,6.,8.,12.,16.,24.)
CLOSINGS=(1,3)
AREAS=(3,8)
read,write,sha=c.read,c.write,c.sha
NORM=read(OLD/'normalization.json')
CAL_OLD=read(old.OUT/'development/calibration_sigma1.json')
ORIGINAL_OP=dict(method='original_static',threshold=16.,closing=3,area=8,shape='original',merge=False,snap=False,aspect=1.3)

def utc():return datetime.now(timezone.utc).isoformat()

def prepare():
    c.verify()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sources=set(Path(p) for p in read(OLD/'protocol.json')['source_hashes'])
    sources.update([Path(__file__),Path(learn.__file__),Path(metrics.__file__),Path(base.__file__),Path(old.__file__),OLD/'normalization.json',old.OUT/'development/calibration_sigma1.json'])
    sources.update(OLD/f'datasets/{s}/index.json' for s in ('train','val'))
    cnn=read(OLD/'frozen_models.json')['arms']['mixed_96']['cnn'];sources.add(Path(cnn['checkpoint']))
    existing=set();acq=set()
    for p in (ROOT/'DataSimulation/data/output').glob('*/datasets/*/index.json'):
        try:
            rows=read(p)
            if isinstance(rows,list):
                existing.update(r['seed'] for r in rows if 'seed' in r)
                acq.update(r['acquisition_seed'] for r in rows if 'acquisition_seed' in r)
        except (ValueError,TypeError):pass
    assert not set(range(SEED_BASE,SEED_BASE+N_FRESH))&existing
    assert not {SEED_BASE+i+r*100000000 for i in range(N_FRESH) for r in (0,1)}&acq
    protocol=dict(study=OUT.name,created_utc=utc(),sources={str(p):sha(p) for p in sorted(sources)},
      selection='Original training/validation only. Historical confirmation is descriptive context, never used in selection. Fresh confirmation generated only after freeze.',
      methods=METHODS,thresholds=THRESHOLDS,areas=AREAS,closings=CLOSINGS,
      processing='Gaussian input sigma=1. Original references +/-64,128,192,256; short references +/-32,48,64,80; gaussian background sigma=32 columns. Static row MAD fitted to clean training repeat0, all levels; adaptive row MAD estimated from current image, floor .003.',
      region_variants='Original 2-row inward shift/1px pad; width-derived height, aspect 1/1.3/1.6, same inward shift; optional vertically aligned fragment merge; optional snapping to known three radial positions, reported separately as a restricted prior.',
      objective='Maximise mean per-count box F1; ties fewer false regions, then fixed iteration order. Also select separate count-specific policies without radial snapping.',
      fresh_backgrounds=N_FRESH,fresh_seed_base=SEED_BASE,physical_validation=False,
      diagnostic='Validation repeat0 at64: measured versus noise-free original median; paired clean subtraction oracle, relative thresholds .1/.25/.5, original and width box reconstruction. Oracle is not deployable.',
      cnn=cnn)
    p=OUT/'protocol.json'
    if p.exists():assert read(p)['sources']==protocol['sources']
    else:write(p,protocol)
    return read(p)

def verify(frozen=False):
    p=read(OUT/'protocol.json')
    for s,h in p['sources'].items():assert sha(s)==h,s
    if frozen:
        f=read(OUT/'frozen.json')
        for s,h in f['dependencies'].items():assert sha(s)==h,s
        return f
    return p

def residual(image,kind):
    if kind=='gauss':return image-gaussian_filter1d(image,32.,axis=1,mode='wrap')
    ks=learn.OFFSETS if kind=='original' else (-80,-64,-48,-32,32,48,64,80)
    return image-np.median(np.stack([np.roll(image,-k,axis=1) for k in ks]),axis=0)

def normalized(raw):return base.normalize(gaussian_filter(np.asarray(raw),1,mode=('reflect','wrap')),NORM)

def calibration_worker(row):
    x=normalized(np.load(row['raw_path']))
    return {k:residual(x,k)[:,::32].astype('f4') for k in ('short','gauss')}

def calibrate():
    verify();target=OUT/'calibration.json'
    if target.exists():return read(target)
    rows=[r for r in read(OLD/'datasets/train/index.json') if r['source_class']=='clean' and r['repeat']==0]
    samples={'short':[],'gauss':[]}
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for i,result in enumerate(pool.map(calibration_worker,rows)):
            for k,v in result.items():samples[k].append(v)
            if (i+1)%48==0:print('CALIBRATION',i+1,'/',len(rows),flush=True)
    result={'original':CAL_OLD}
    for k,parts in samples.items():
        a=np.concatenate(parts,axis=1);centre=np.median(a,axis=1,keepdims=True)
        scale=np.maximum(.003,1.4826*np.median(abs(a-centre),axis=1,keepdims=True))
        result[k]=dict(centre=centre.tolist(),scale=scale.tolist(),sampled_clean_scenes=len(rows),column_stride=32)
    write(target,result);return result

def score_maps(raw,cal,methods=METHODS):
    x=normalized(raw);out={}
    for k in ('original','short','gauss'):
        if not any(m.startswith(k+'_') for m in methods):continue
        d=residual(x,k)
        for mode in ('static','adaptive'):
            name=k+'_'+mode
            if name not in methods:continue
            if mode=='static':centre=np.asarray(cal[k]['centre']);scale=np.asarray(cal[k]['scale'])
            else:
                centre=np.median(d,axis=1,keepdims=True)
                scale=np.maximum(.003,1.4826*np.median(abs(d-centre),axis=1,keepdims=True))
            out[name]=(abs(d-centre)/scale)[16:144].astype('f4')
    return out

def components(score,threshold,closing):
    # Existing periodic component implementation, with component area preserved.
    return old.compact_components(score,threshold,closing)

def merge_fragments(boxes):
    work=[list(b) for b in boxes]
    # Only vertical fragments with substantial angular overlap; no truth input.
    for _ in range(6):
        changed=False
        bins={}
        for j,b in enumerate(work):
            for xx in range(int(b[0])//32,int(b[2]-1)//32+1):
                for yy in range(int(b[1])//16,int(b[3]-1)//16+1):bins.setdefault((xx%157,yy),set()).add(j)
        for i in range(len(work)):
            a=work[i]
            neighbours=set()
            for xx in range(int(a[0])//32,int(a[2]-1)//32+1):
                for yy in range(int(a[1]-8)//16,int(a[3]+8)//16+1):neighbours.update(bins.get((xx%157,yy),()))
            for j in sorted(j for j in neighbours if j>i):
                b=work[j];best=None
                for shift in (-5024,0,5024):
                    bb=[b[0]+shift,b[1],b[2]+shift,b[3]]
                    overlap=max(0,min(a[2],bb[2])-max(a[0],bb[0]))
                    gap=max(0,max(a[1],bb[1])-min(a[3],bb[3]))
                    u=[min(a[0],bb[0]),min(a[1],bb[1]),max(a[2],bb[2]),max(a[3],bb[3])]
                    if overlap>=.5*min(a[2]-a[0],bb[2]-bb[0]) and gap<=8 and max(u[2]-u[0],u[3]-u[1])<=40:best=u;break
                if best is not None:
                    w=best[2]-best[0];best[0]%=5024;best[2]=best[0]+w
                    work[i]=best;work.pop(j);changed=True;break
            if changed:break
        if not changed:break
    return work

def reconstruct(comps,op):
    boxes=[b[:4] for b in comps if b[4]>=op['area']]
    if op['merge']:boxes=merge_fragments(boxes)
    return reconstruct_boxes(boxes,op)

def reconstruct_boxes(boxes,op,as_array=False):
    if not len(boxes):return np.empty((0,4),int) if as_array else []
    b=np.asarray(boxes,float);cx=(b[:,0]+b[:,2])/2;cy=(b[:,1]+b[:,3])/2
    cy=cy+2*np.sign(64-cy);width=b[:,2]-b[:,0]+2
    height=b[:,3]-b[:,1]+2 if op['shape']=='original' else np.maximum(3.,width/op['aspect'])
    if op['snap']:
        options=np.array([43.,64.,85.]);cy=options[np.argmin(abs(cy[:,None]-options),axis=1)]
    left=np.floor(cx-width/2);right=np.ceil(cx+width/2)
    top=np.maximum(0,np.floor(cy-height/2));bottom=np.minimum(128,np.ceil(cy+height/2))
    out=np.stack([left%5024,top,left%5024+np.minimum(5024,right-left),bottom],axis=1)[bottom>top].astype(int)
    return out if as_array else out.tolist()

def truth_of(row):
    with np.load(row['geometry_path']) as z:return metrics.regions(z['mask'][16:144])[1]

def val_worker(row):
    target=OUT/'validation_components'/f"{row['scene_id']}.json"
    if target.exists():return str(target)
    cal=read(OUT/'calibration.json');maps=score_maps(np.load(row['raw_path']),cal)
    cache={}
    for name,s in maps.items():
        for t,close in itertools.product(THRESHOLDS,CLOSINGS):cache[f'{name}:{t:g}:{close}']=components(s,t,close)
    write(target,dict(meta={k:row[k] for k in ('scene_id','background_id','source_class','count_multiplier','diameter_mm','repeat')},truth=truth_of(row),components=cache))
    return str(target)

def validation():
    verify();calibrate();(OUT/'validation_components').mkdir(exist_ok=True)
    rows=read(OLD/'datasets/val/index.json')
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for i,_ in enumerate(pool.map(val_worker,rows)):
            if (i+1)%48==0:print('VALIDATION MAPS',i+1,'/',len(rows),flush=True)
    print('VALIDATION MAPS COMPLETE',flush=True)

def result_row(meta,truth,pred):
    if len(truth)<=1:
        best=0.
        if len(pred) and truth:
            a=np.asarray(pred);b=truth[0]
            vertical=np.maximum(0,np.minimum(a[:,3],b[3])-np.maximum(a[:,1],b[1]))
            horizontal=sum(np.maximum(0,np.minimum(a[:,2],b[2]+shift)-np.maximum(a[:,0],b[0]+shift)) for shift in (-5024,0,5024))
            inter=vertical*horizontal;union=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])+(b[2]-b[0])*(b[3]-b[1])-inter
            best=float(np.max(np.divide(inter,union,out=np.zeros(len(pred),float),where=union>0)))
        tp=int(best>=.5);match=dict(tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,matched_iou_sum=best if tp else 0.)
    else:match=metrics.match(pred,truth,5024)
    return dict(**meta,**match,predicted_boxes=pred,truth_boxes=truth,predicted_regions=len(pred))

def summary(rows):
    by={}
    for m in (1,4,64):
        rr=[r for r in rows if r['count_multiplier']==m]
        if not rr:continue
        tp,fp,fn=[sum(r[k] for r in rr) for k in ('tp','fp','fn')]
        clean=[r for r in rr if r['source_class']=='clean']
        by[str(m)]=dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),clean_alarms=sum(r['predicted_regions']>0 for r in clean),clean_scenes=len(clean))
    return dict(by_count=by,macro_f1=float(np.mean([v['f1'] for v in by.values()])),fp=sum(v['fp'] for v in by.values()))

def op_variants():
    shapes=[dict(shape='original',merge=False,snap=False,aspect=1.3)]
    shapes += [dict(shape='width',merge=merge,snap=snap,aspect=aspect) for aspect,merge,snap in itertools.product((1.,1.3,1.6),(False,True),(False,True))]
    for method,t,close,area,shape in itertools.product(METHODS,THRESHOLDS,CLOSINGS,AREAS,shapes):
        yield dict(method=method,threshold=t,closing=close,area=area,**shape)

def evaluate_cache(cache,op):
    key=f"{op['method']}:{op['threshold']:g}:{op['closing']}"
    out=[]
    for r in cache:
        boxkey=(key,op['area'],op['merge'])
        stored=r.setdefault('_boxes',{})
        if boxkey not in stored:
            boxes=[b[:4] for b in r['components'][key] if b[4]>=op['area']]
            stored[boxkey]=np.asarray(merge_fragments(boxes) if op['merge'] else boxes).reshape(-1,4)
        out.append(result_row(r['meta'],r['truth'],reconstruct_boxes(stored[boxkey],op,as_array=True)))
    return out

def select():
    verify();path=OUT/'selection.json'
    if path.exists():return read(path)
    cache=[read(OUT/'validation_components'/f"{r['scene_id']}.json") for r in read(OLD/'datasets/val/index.json')]
    trials=[]
    for i,op in enumerate(op_variants()):
        res=summary(evaluate_cache(cache,op));trials.append(dict(op=op,**res))
        if (i+1)%156==0:
            print('SELECTION',i+1,'best',max(x['macro_f1'] for x in trials),flush=True)
        if (i+1)%312==0:
            for r in cache:r.pop('_boxes',None)
    key=lambda r:(r['macro_f1'],-r['fp'])
    classical=[r for r in trials if not r['op']['snap']]
    prior=[r for r in trials if r['op']['snap']]
    selections=dict(classical=max(classical,key=key),position_prior=max(prior,key=key),
      original_score_restructure=max([r for r in classical if r['op']['method']=='original_static'],key=key),
      per_count={str(m):max(classical,key=lambda r:(r['by_count'][str(m)]['f1'],-r['by_count'][str(m)]['fp'])) for m in (1,4,64)},
      baseline=dict(op=ORIGINAL_OP,**summary(evaluate_cache(cache,ORIGINAL_OP))),
      best_by_score={m:max([r for r in classical if r['op']['method']==m],key=key) for m in METHODS},
      candidate_count=len(trials),trials=trials)
    assert abs(selections['baseline']['macro_f1']-.3997219998497546)<1e-12,'Baseline replay differs'
    write(path,selections)
    print('SELECTED',json.dumps({k:v for k,v in selections.items() if k not in ('trials','best_by_score')}),flush=True)
    return selections

def freeze():
    verify();assert not (OUT/'datasets/test').exists()
    s=select();paths=[Path(__file__),OUT/'protocol.json',OUT/'calibration.json',OUT/'selection.json']
    f=dict(frozen_utc=utc(),policies={k:s[k] for k in ('classical','position_prior','original_score_restructure','per_count','baseline')},
           dependencies={str(p):sha(p) for p in paths})
    write(OUT/'frozen.json',f);print('FROZEN',f['frozen_utc'],flush=True)

def background(index):
    verify(True)
    c.OUT=OUT;c.BASES={**c.BASES,'test':SEED_BASE};c.COUNTS={**c.COUNTS,'test':N_FRESH}
    c.verify=lambda:verify(True);learn.verify_freeze=lambda:verify(True)
    c.background('test',index)

def generate():
    verify(True);logs=OUT/'logs';logs.mkdir(exist_ok=True)
    pending=[i for i in range(N_FRESH) if not (OUT/f'datasets/test/{i:03d}/index.json').exists()];active=[]
    while pending or active:
        while pending and len(active)<3:
            i=pending.pop(0);fh=(logs/f'fresh_{i:03d}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'background','--index',str(i)],stdout=fh,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((p,fh,i))
        for p,fh,i in active[:]:
            if p.poll() is not None:
                fh.close();active.remove((p,fh,i))
                if p.returncode:raise RuntimeError(f'Fresh background failed: {i}; see log')
                print('FRESH BACKGROUND COMPLETE',i,flush=True)
        if active:time.sleep(2)
    rows=[r for i in range(N_FRESH) for r in read(OUT/f'datasets/test/{i:03d}/index.json')]
    assert len(rows)==N_FRESH*18
    write(OUT/'datasets/test/index.json',rows)

def confirm_worker(row):
    target=OUT/'confirmation_scenes'/f"{row['scene_id']}.json"
    if target.exists():return str(target)
    policies=read(OUT/'frozen.json')['policies'];cal=read(OUT/'calibration.json')
    ops={k:v['op'] for k,v in policies.items() if k!='per_count'}
    ops['per_count']=policies['per_count'][str(row['count_multiplier'])]['op']
    maps=score_maps(np.load(row['raw_path']),cal,{op['method'] for op in ops.values()});truth=truth_of(row)
    meta={k:row[k] for k in ('scene_id','background_id','source_class','count_multiplier','diameter_mm','repeat')}
    cache={};results={}
    for name,op in ops.items():
        key=(op['method'],op['threshold'],op['closing'])
        if key not in cache:cache[key]=components(maps[op['method']],op['threshold'],op['closing'])
        results[name]=result_row(meta,truth,reconstruct(cache[key],op))
    write(target,results);return str(target)

def confirm():
    verify(True);(OUT/'confirmation_scenes').mkdir(exist_ok=True)
    rows=read(OUT/'datasets/test/index.json')
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for i,_ in enumerate(pool.map(confirm_worker,rows)):
            if (i+1)%72==0:print('CONFIRM MEDIAN',i+1,'/',len(rows),flush=True)
    methods={k:[] for k in read(OUT/'frozen.json')['policies']}
    for row in rows:
        for k,v in read(OUT/'confirmation_scenes'/f"{row['scene_id']}.json").items():methods[k].append(v)
    write(OUT/'confirmation_median.json',{k:dict(**summary(v),rows=v) for k,v in methods.items()})
    print('CONFIRM MEDIAN RESULTS',json.dumps({k:summary(v) for k,v in methods.items()}),flush=True)

def cnn():
    verify(True);target=OUT/'confirmation_cnn.json'
    if target.exists():return
    spec=read(OUT/'protocol.json')['cnn'];model=base.model_for(spec)
    metas=read(OUT/'datasets/test/index.json');folder=OUT/'cnn_scene_results';folder.mkdir(exist_ok=True);rows=[]
    for i,r in enumerate(metas):
        p=folder/f"{r['scene_id']}.json"
        if p.exists():rows.append(read(p));continue
        raw=np.load(r['raw_path']);score=base.predict_image(base.normalize(raw,NORM),model)
        op=spec['selection']['best'];_,pred=metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
        result=result_row({k:r[k] for k in ('scene_id','background_id','source_class','count_multiplier','diameter_mm','repeat')},truth_of(r),pred)
        write(p,result);rows.append(result)
        if (i+1)%36==0:print('CONFIRM CNN',i+1,'/',len(metas),flush=True)
    write(target,dict(**summary(rows),rows=rows));print('CNN RESULT',json.dumps(summary(rows)),flush=True)

def diagnostic():
    verify();target=OUT/'diagnostic.json'
    if target.exists():return
    cal=read(OUT/'calibration.json');rows=[r for r in read(OLD/'datasets/val/index.json') if r['repeat']==0 and r['count_multiplier']==64]
    outputs={};stages=[]
    for i,r in enumerate(rows):
        with np.load(r['geometry_path']) as z:quiet=z['quiet'];mask=z['mask'][16:144]
        with np.load(Path(r['geometry_path']).with_name('clean_geometry.npz')) as z:clean=z['quiet']
        truth=metrics.regions(mask)[1];meta={k:r[k] for k in ('scene_id','background_id','source_class','count_multiplier','diameter_mm','repeat')}
        raw=np.load(r['raw_path']);measured=score_maps(raw,cal,('original_static',))['original_static']
        quiet_score=score_maps(quiet,cal,('original_static',))['original_static']
        difference=abs(normalized(quiet)-normalized(clean))[16:144]
        oracle=difference/max(float(difference.max()),1e-12)
        for label,s,t,shape in [('measured',measured,16,'original'),('quiet',quiet_score,16,'original')]+[(f'oracle_{t}_{shape}',oracle,t,shape) for t,shape in itertools.product((.1,.25,.5),('original','width'))]:
            op={**ORIGINAL_OP,'threshold':t,'shape':shape,'area':3 if label.startswith('oracle') else 8}
            pred=reconstruct(components(s,t,3),op);outputs.setdefault(label,[]).append(result_row(meta,truth,pred))
        if truth:
            _,before=metrics.binary(measured,16,1,3);_,after=metrics.binary(measured,16,8,3)
            overlap=lambda boxes:any(metrics.circular_iou(p,truth[0],5024)>0 for p in boxes)
            corrected=reconstruct(components(measured,16,3),ORIGINAL_OP)
            stages.append(dict(scene_id=r['scene_id'],diameter_mm=r['diameter_mm'],peak_inside=float(measured[mask>0].max()),above_threshold_pixels_inside=int(((measured>=16)&(mask>0)).sum()),
              overlaps_before_area=overlap(before),overlaps_after_area=overlap(after),matches_before_correction=metrics.match(after,truth,5024)['tp'],matches_after_correction=metrics.match(corrected,truth,5024)['tp']))
        if (i+1)%24==0:print('DIAGNOSTIC',i+1,'/',len(rows),flush=True)
    write(target,dict(scope='Validation repeat0 at64 only; paired-clean subtraction oracle unavailable at inference and normalised using each diagnostic image maximum. Not a deployable detector.',results={k:dict(**summary(v),rows=v) for k,v in outputs.items()},stages=stages))

def selfcheck():
    # Meaningful geometry and leakage guards before expensive selection.
    m=np.zeros((128,5024),'u1');m[40:45,:3]=1;m[40:45,-3:]=1
    _,boxes=metrics.regions(m);assert len(boxes)==1 and boxes[0][2]-boxes[0][0]==6
    assert metrics.match(boxes,boxes,5024)['tp']==1
    comps=[[100,40,120,43,60],[101,48,119,51,54]]
    assert len(merge_fragments([b[:4] for b in comps]))==1
    op={**ORIGINAL_OP,'shape':'width','merge':True,'snap':False,'aspect':1.3}
    b=reconstruct(comps,op);assert len(b)==1 and b[0][3]-b[0][1]>3
    assert np.all(score_maps(np.ones((160,5024),'f4')*.1,{'original':CAL_OLD},('original_adaptive',))['original_adaptive']==0)
    assert len(list(op_variants()))==1872
    write(OUT/'selfcheck.json',dict(passed=True,checks=['periodic component and exact IoU','fragment merge','width reconstruction','constant-image adaptive residual','bounded candidate count']))
    print('SELFCHECK PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','development','select','freeze','background','generate','confirm','cnn','diagnostic','selfcheck']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();selfcheck()
    elif a.action=='development':calibrate();validation();select();diagnostic()
    elif a.action=='background':background(a.index)
    else:globals()[a.action]()
