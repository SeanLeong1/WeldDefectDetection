"""Separate median experiment: bounded development, frozen fresh confirmation.

Historical sources and datasets are read-only. No truth is passed to inference.
Run prepare, development, freeze, generate, confirm, report in that order.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time, argparse, itertools, subprocess, concurrent.futures
import numpy as np
import cv2
from scipy.ndimage import gaussian_laplace, maximum_filter
import median_exploration_20261115 as ex

ROOT=ex.ROOT
OUT=ROOT/'DataSimulation/data/output/median_scale_templates_20261115_v1'
REPORT=ROOT/'output/median_scale_templates_20261115'
SEED_BASE=6111500000
N_FRESH=48
WORKERS=3
FLOORS=(.003,.001,.0003)
THRESHOLDS=(4.,8.,12.,16.,24.)
SIGMAS=(1.,2.,4.,8.)
REFS=('long','short','near','periodic')
META=('scene_id','background_id','source_class','count_multiplier','diameter_mm','repeat','amplitude')
read,write,sha=ex.read,ex.write,ex.sha
cv2.setNumThreads(1)

def prepare():
    ex.verify(True)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return verify()
    seen=set();acq=set()
    for p in (ROOT/'DataSimulation/data/output').glob('*/datasets/*/index.json'):
        try:
            rr=read(p)
            if isinstance(rr,list):
                seen.update(r['seed'] for r in rr if 'seed' in r)
                acq.update(r['acquisition_seed'] for r in rr if 'acquisition_seed' in r)
        except (ValueError,TypeError):pass
    assert not set(range(SEED_BASE,SEED_BASE+N_FRESH))&seen
    assert not {SEED_BASE+i+r*100000000 for i in range(N_FRESH) for r in (0,1)}&acq
    sources=set(Path(p) for p in read(ex.OUT/'protocol.json')['sources'])
    sources|={Path(__file__),ex.OUT/'frozen.json',ex.OUT/'calibration.json'}
    p=dict(created_utc=ex.utc(),study=OUT.name,sources={str(p):sha(p) for p in sorted(sources)},
        development='Original 96 train and 32 validation backgrounds. Previous tests are context only. Reused validation, exploratory selection.',
        fresh=dict(backgrounds=N_FRESH,seed_base=SEED_BASE,scenes=18*N_FRESH,gate='All policies and dependency hashes frozen before generation.'),
        selection='Sequential: box-only grid on original score; score/scale/reference grid holding selected boxes fixed; independent multiscale and template grids. Shared winner and count-specific winners by F1, fewer FP, deterministic order.',
        box_grid=dict(gap=[0,4,10],area=[3,8],shift=['fixed','row'],pad_fraction=[0,.10,.25],shape=['union','width1.6']),
        references=dict(long=[64,128,192,256],short=[32,48,64,80],near=[16,24,32,48],periodic='five harmonics, three Tukey IRLS updates, stride16 fit'),
        scores=dict(floors=FLOORS,thresholds=THRESHOLDS,calibration='per-image row MAD; additionally long-reference count-specific clean-training MAD',closing=3),
        multiscale='sigma^2 times Laplacian of Gaussian at 1,2,4,8 on signed median residual; extrema over scale and 3x3 space; row MAD significance; diameter 2sqrt(2)sigma times validation-selected factor; periodic NMS',
        templates='Signed paired quiet pore-minus-clean training patches, averaged within size x radial-position x pore-family bins; unit-energy filters; mask-derived prototype box dimensions. Full-image correlation, no test truth or radial snapping.',
        template_grid=dict(thresholds=[4,6,8,12,16,24],box_factors=[.8,1.,1.2]),
        limitations='Synthetic, known ring coordinates, one pore per defective image. Templates use labelled simulator-derived training priors. No physical model changes.',
        seed_collision_check='passed')
    write(OUT/'protocol.json',p);return p

def verify(frozen=False):
    p=read(OUT/'protocol.json')
    for f,h in p['sources'].items():assert sha(f)==h,f
    if frozen:
        p=read(OUT/'frozen.json')
        for f,h in p['dependencies'].items():assert sha(f)==h,f
    return p

def periodic_residual(x):
    n=x.shape[1];tt=2*np.pi*np.arange(n)/n
    basis=np.stack([np.ones(n)]+[f(k*tt) for k in range(1,6) for f in (np.cos,np.sin)],axis=1)
    b=basis[::16];y=x[:,::16].astype(float)
    coef=np.linalg.lstsq(b,y.T,rcond=None)[0].T
    for _ in range(3):
        err=y-coef@b.T
        med=np.median(err,axis=1,keepdims=True)
        scale=np.maximum(1e-5,1.4826*np.median(abs(err-med),axis=1,keepdims=True))
        u=(err-med)/(4.685*scale);w=np.maximum(0,1-u*u)**2;w[abs(u)>=1]=0
        lhs=np.einsum('ni,rn,nj->rij',b,w,b)+np.eye(11)[None]*1e-8
        rhs=np.einsum('ni,rn->ri',b,w*y)
        coef=np.linalg.solve(lhs,rhs[...,None])[...,0]
    return (x-coef@basis.T).astype('f4')

def residual(x,ref):
    if ref=='periodic':return periodic_residual(x)
    ks={'long':(64,128,192,256),'short':(32,48,64,80),'near':(16,24,32,48)}[ref]
    return x-np.median(np.stack([np.roll(x,s*k,axis=1) for k in ks for s in (-1,1)]),axis=0)

def standardize(d,floor,cal=None):
    if cal is None:
        centre=np.median(d,axis=1,keepdims=True)
        scale=1.4826*np.median(abs(d-centre),axis=1,keepdims=True)
    else:centre=np.asarray(cal['centre']);scale=np.asarray(cal['scale'])
    return abs(d-centre)/np.maximum(floor,scale)

def train_worker(row):
    if row['source_class']=='clean':
        return dict(count=row['count_multiplier'],sample=residual(ex.normalized(np.load(row['raw_path'])),'long')[:,::32])
    with np.load(row['geometry_path']) as z:q=ex.normalized(z['quiet']);mask=z['mask'][16:144]
    with np.load(Path(row['geometry_path']).with_name('clean_geometry.npz')) as z:cl=ex.normalized(z['quiet'])
    truth=ex.metrics.regions(mask)[1][0];cx=int(np.floor((truth[0]+truth[2]-1)/2));cy=int(np.floor((truth[1]+truth[3]-1)/2))
    cols=(cx+np.arange(-24,25))%5024;ys=cy+16+np.arange(-16,17)
    patch=(q-cl)[ys[:,None],cols[None,:]]
    patch=patch-patch.mean(axis=1,keepdims=True)
    size=int(np.digitize(row['diameter_mm'],[.3,.6]));pos=int(np.argmin(abs(np.array([43,64,85])-cy)))
    return dict(key=f"{row['source_class']}_{size}_{pos}",patch=patch,box=[truth[2]-truth[0],truth[3]-truth[1]],
                centre_offset=[(truth[0]+truth[2]-1)/2-cx,(truth[1]+truth[3]-1)/2-cy],scene_id=row['scene_id'])

def calibrate():
    verify()
    if (OUT/'training_calibration.json').exists():return
    rr=[r for r in read(ex.OLD/'datasets/train/index.json') if r['repeat']==0 and (r['source_class']=='clean' or r['count_multiplier']==64)]
    samples={str(c):[] for c in (1,4,64)};groups={}
    with concurrent.futures.ProcessPoolExecutor(WORKERS) as pool:
        for i,v in enumerate(pool.map(train_worker,rr)):
            if 'sample' in v:samples[str(v['count'])].append(v['sample'])
            else:groups.setdefault(v['key'],[]).append(v)
            if (i+1)%60==0:print('TRAIN CALIBRATION',i+1,'/',len(rr),flush=True)
    cal={}
    for k,parts in samples.items():
        a=np.concatenate(parts,axis=1);centre=np.median(a,axis=1,keepdims=True)
        cal[k]=dict(centre=centre.tolist(),scale=(1.4826*np.median(abs(a-centre),axis=1,keepdims=True)).tolist())
    bank=[];meta=[]
    for key,parts in sorted(groups.items()):
        # Equal influence for each geometry, independent of response amplitude.
        patches=[p['patch']/max(np.linalg.norm(p['patch']),1e-12) for p in parts]
        t=np.mean(patches,axis=0);t/=max(np.linalg.norm(t),1e-12);bank.append(t.astype('f4'))
        meta.append(dict(key=key,n=len(parts),width=float(np.median([p['box'][0] for p in parts])),height=float(np.median([p['box'][1] for p in parts])),
                         centre_offset=np.median([p['centre_offset'] for p in parts],axis=0).tolist(),training_scenes=[p['scene_id'] for p in parts]))
    np.savez_compressed(OUT/'templates.npz',bank=np.stack(bank))
    write(OUT/'training_calibration.json',dict(count_calibration=cal,templates=meta,training_only=True))

def box_union(a,b):
    best=None
    for shift in (-5024,0,5024):
        bb=[b[0]+shift,b[1],b[2]+shift,b[3]]
        dx=max(0,max(a[0],bb[0])-min(a[2],bb[2]));dy=max(0,max(a[1],bb[1])-min(a[3],bb[3]))
        dist=float(np.hypot(dx,dy));u=[min(a[0],bb[0]),min(a[1],bb[1]),max(a[2],bb[2]),max(a[3],bb[3])]
        if best is None or dist<best[0]:best=(dist,u)
    return best

def merge(comps,gap,area):
    # Retain small fragments before merging, then apply total-area requirement.
    if not gap:return [b[:4] for b in comps if b[4]>=area]
    boxes=[list(b[:4]) for b in comps];mass=[b[4] for b in comps];parent=list(range(len(boxes)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    bins={};pairs=set()
    for i,b in enumerate(boxes):
        for x in range(int(b[0]-gap)//32,int(b[2]+gap)//32+1):
            for y in range(int(b[1]-gap)//32,int(b[3]+gap)//32+1):
                key=(x%157,y)
                pairs.update((j,i) for j in bins.get(key,()))
                bins.setdefault(key,[]).append(i)
    edges=[]
    for i,j in pairs:
        d,u=box_union(boxes[i],boxes[j])
        if d<=gap and max(u[2]-u[0],u[3]-u[1])<=40:edges.append((d,i,j))
    for _,i,j in sorted(edges):
        i,j=root(i),root(j)
        if i==j:continue
        d,u=box_union(boxes[i],boxes[j])
        if d<=gap and max(u[2]-u[0],u[3]-u[1])<=40:
            w=u[2]-u[0];u[0]%=5024;u[2]=u[0]+w;boxes[i]=u;mass[i]+=mass[j];parent[j]=i
    return [b for i,b in enumerate(boxes) if root(i)==i and mass[i]>=area]

def shift_rows(cy,kind):
    if kind=='fixed':return 2*np.sign(64-cy)
    if kind=='none':return np.zeros_like(cy)
    # Frozen training diagnostic: opposite of measured contrast displacement.
    return np.interp(cy,[0,43,64,85,128],[1.54,1.54,0,-1.17,-1.17])

def boxes_from_components(comps,op):
    boxes=merge(comps,op['gap'],op['area'])
    if not boxes:return []
    a=np.asarray(boxes,float);cx=(a[:,0]+a[:,2])/2;cy=(a[:,1]+a[:,3])/2
    w=a[:,2]-a[:,0];h=a[:,3]-a[:,1];cy+=shift_rows(cy,op['shift'])
    pad=1+op['pad_fraction']*np.maximum(w,h);w+=2*pad
    h=h+2*pad if op['shape']=='union' else np.maximum(3,w/1.6)
    return rounded_boxes(cx,cy,w,h)

def rounded_boxes(cx,cy,w,h):
    l=np.floor(cx-w/2);r=np.ceil(cx+w/2);t=np.maximum(0,np.floor(cy-h/2));b=np.minimum(128,np.ceil(cy+h/2))
    return np.stack([l%5024,t,l%5024+np.minimum(5024,r-l),b],axis=1)[b>t].astype(int).tolist()

def peak_candidates(values,scores,widths,heights,offsets=None):
    peaks=values==maximum_filter(values,size=(3,3,3),mode=('nearest','reflect','wrap'))
    ss,yy,xx=np.nonzero(peaks&(scores>=4))
    if not len(ss):return []
    v=scores[ss,yy,xx];order=np.argsort(-v,kind='stable')
    # No top-k truncation; all above-threshold maxima retained for later NMS.
    out=[]
    for i in order:
        s=int(ss[i]);x=float(xx[i]);y=float(yy[i]);ox,oy=(0.,0.) if offsets is None else offsets[s]
        out.append([float(v[i]),x+.5+ox,y+.5+oy,float(widths[s]),float(heights[s]),s])
    return out

def multiscale(d,ref):
    vals=[];scores=[]
    for s in SIGMAS:
        response=-s*s*gaussian_laplace(d,s,mode=('reflect','wrap'))
        vals.append(abs(response)[16:144]);scores.append(standardize(response,.001)[16:144])
    w=[2*np.sqrt(2)*s for s in SIGMAS]
    return peak_candidates(np.stack(vals),np.stack(scores),w,w)

def template_candidates(d,bank,meta):
    # Pad angularly before correlation; score crop is far from radial borders.
    padded=np.pad(d,((16,16),(24,24)),mode='wrap')
    vals=[]
    for t in bank:
        response=cv2.matchTemplate(padded,t,cv2.TM_CCORR)
        centre=np.median(response,axis=1,keepdims=True)
        scale=np.maximum(.003,1.4826*np.median(abs(response-centre),axis=1,keepdims=True))
        vals.append(((response-centre)/scale)[16:144])
    v=np.stack(vals)
    # Template index is categorical: compare all templates, not adjacent bins.
    winning=np.argmax(v,axis=0);best=np.max(v,axis=0)
    spatial=best==maximum_filter(best,size=3,mode=('reflect','wrap'))
    yy,xx=np.nonzero(spatial&(best>=4));order=np.argsort(-best[yy,xx],kind='stable');out=[]
    for i in order:
        y,x=int(yy[i]),int(xx[i]);s=int(winning[y,x]);m=meta[s];ox,oy=m['centre_offset']
        out.append([float(best[y,x]),x+.5+ox,y+.5+oy,m['width'],m['height'],s])
    return out

def nms_peaks(candidates,op):
    selected=[];bins={}
    for score,x,y,w,h,s in candidates:
        if score<op['threshold']:continue
        w*=op['factor'];h*=op['factor']/op.get('aspect',1.)
        if op['family']=='multiscale':y+=float(shift_rows(np.asarray(y),op['shift']))
        box=rounded_boxes(np.asarray([x]),np.asarray([y]),np.asarray([w]),np.asarray([h]))
        if not box:continue
        b=box[0];near=set()
        for xx in range(int(b[0])//32,int(b[2])//32+1):
            for yy in range(int(b[1])//32,int(b[3])//32+1):near.update(bins.get((xx%157,yy),()))
        if any(ex.metrics.circular_iou(b,selected[j],5024)>.2 for j in near):continue
        j=len(selected);selected.append(b)
        for xx in range(int(b[0])//32,int(b[2])//32+1):
            for yy in range(int(b[1])//32,int(b[3])//32+1):bins.setdefault((xx%157,yy),[]).append(j)
    return selected

def score_key(ref,mode,floor):return f'{ref}_{mode}_{floor:g}'

def scene_cache(row):
    x=ex.normalized(np.load(row['raw_path']));cal=read(OUT/'training_calibration.json')
    with np.load(OUT/'templates.npz') as z:bank=z['bank']
    rr={k:residual(x,k) for k in REFS};comps={}
    original=standardize(rr['long'],.003,ex.CAL_OLD)[16:144]
    comps['baseline:16']=ex.components(original,16,3)
    for ref,d in rr.items():
        for mode in (('adaptive','count') if ref=='long' else ('adaptive',)):
            cc=cal['count_calibration'][str(row['count_multiplier'])] if mode=='count' else None
            for floor in FLOORS:
                score=standardize(d,floor,cc)[16:144]
                for t in THRESHOLDS:comps[f'{score_key(ref,mode,floor)}:{t:g}']=ex.components(score,t,3)
    previous=read(ex.OUT/'frozen.json')['policies']['classical']['op']
    score=standardize(rr['short'],.003)[16:144]
    prev=ex.reconstruct(ex.components(score,previous['threshold'],previous['closing']),previous)
    return dict(meta={k:row[k] for k in META},truth=ex.truth_of(row),components=comps,previous=prev,
        multiscale={ref:multiscale(rr[ref],ref) for ref in ('long','near')},
        templates={ref:template_candidates(rr[ref],bank,cal['templates']) for ref in ('long','near')})

def cache_worker(row):
    p=OUT/'validation_cache'/f"{row['scene_id']}.json"
    if not p.exists():write(p,scene_cache(row))
    return p.stat().st_size

def predict(cache,op):
    f=op['family']
    if f=='baseline':return ex.reconstruct(cache['components']['baseline:16'],ex.ORIGINAL_OP)
    if f=='previous':return cache['previous']
    if f in ('multiscale','templates'):return nms_peaks(cache[f][op['reference']],op)
    return boxes_from_components(cache['components'][f"{op['score']}:{op['threshold']:g}"],op)

def evaluate(caches,op):
    return [ex.result_row(c['meta'],c['truth'],predict(c,op)) for c in caches]

def choose(trials,count=None):
    return max(trials,key=lambda r:(r['macro_f1'] if count is None else r['by_count'][str(count)]['f1'],-r['fp']))

def select():
    verify()
    if (OUT/'selection.json').exists():return read(OUT/'selection.json')
    caches=[read(OUT/'validation_cache'/f"{r['scene_id']}.json") for r in read(ex.OLD/'datasets/val/index.json')]
    trials=[]
    def trial(op):
        res=dict(op=op,**ex.summary(evaluate(caches,op)));trials.append(res);return res
    baseline=trial(dict(family='baseline'));previous=trial(dict(family='previous'))
    assert abs(baseline['macro_f1']-.3997219998497546)<1e-12
    assert abs(previous['macro_f1']-read(ex.OUT/'selection.json')['classical']['macro_f1'])<1e-12
    boxtrials=[]
    for gap,area,shift,pad,shape in itertools.product((0,4,10),(3,8),('fixed','row'),(0,.1,.25),('union','width1.6')):
        boxtrials.append(trial(dict(family='components',score='baseline',threshold=16.,gap=gap,area=area,shift=shift,pad_fraction=pad,shape=shape)))
    box=choose(boxtrials);print('BOX SELECTED',json.dumps(box),flush=True)
    scoretrials=[]
    for ref in REFS:
        for mode in (('adaptive','count') if ref=='long' else ('adaptive',)):
            for floor,t in itertools.product(FLOORS,THRESHOLDS):
                scoretrials.append(trial({**box['op'],'score':score_key(ref,mode,floor),'threshold':t}))
        print('REFERENCE SELECTED',ref,choose(scoretrials)['macro_f1'],flush=True)
    scaletrials=[]
    for ref,t,factor,aspect,shift in itertools.product(('long','near'),(4,6,8,12,16,24),(.75,1.,1.5),(1.,1.6),('none','row')):
        scaletrials.append(trial(dict(family='multiscale',reference=ref,threshold=t,factor=factor,aspect=aspect,shift=shift)))
    print('MULTISCALE SELECTED',json.dumps(choose(scaletrials)),flush=True)
    templtrials=[]
    for ref,t,factor in itertools.product(('long','near'),(4,6,8,12,16,24),(.8,1.,1.2)):
        templtrials.append(trial(dict(family='templates',reference=ref,threshold=t,factor=factor)))
    print('TEMPLATE SELECTED',json.dumps(choose(templtrials)),flush=True)
    allnew=boxtrials+scoretrials+scaletrials+templtrials
    policies=dict(baseline=baseline,previous=previous,box_only=box,
        adaptive=choose([r for r in scoretrials if r['op']['score'].startswith('long_adaptive')]),
        count_calibrated=choose([r for r in scoretrials if r['op']['score'].startswith('long_count')]),
        reference=choose([r for r in scoretrials if not r['op']['score'].startswith('long_')]),
        multiscale=choose(scaletrials),templates=choose(templtrials),selected=choose(allnew),
        per_count={str(c):choose(allnew,c) for c in (1,4,64)})
    # Locked diagnostic ablations of the selected component policy.
    refbest=policies['reference']['op']
    for name,change in [('no_merge',dict(gap=0)),('fixed_shift',dict(shift='fixed')),('fixed_padding',dict(pad_fraction=0)),('area8',dict(area=8)),('long_reference',dict(score=refbest['score'].replace(refbest['score'].split('_')[0],'long',1)))]:
        policies['ablation_'+name]=trial({**refbest,**change})
    result=dict(policies=policies,candidate_count=len(trials),trials=trials)
    write(OUT/'selection.json',result);return result

def development():
    verify();calibrate();(OUT/'validation_cache').mkdir(exist_ok=True)
    rr=read(ex.OLD/'datasets/val/index.json')
    with concurrent.futures.ProcessPoolExecutor(WORKERS) as pool:
        for i,_ in enumerate(pool.map(cache_worker,rr)):
            if (i+1)%24==0:print('VALIDATION',i+1,'/',len(rr),flush=True)
    select()

def freeze():
    verify();assert not (OUT/'datasets/test').exists(),'Cannot refreeze after confirmation generation'
    s=select();paths=[OUT/n for n in ('protocol.json','selection.json','training_calibration.json','templates.npz')]+[Path(__file__)]
    write(OUT/'frozen.json',dict(frozen_utc=ex.utc(),policies=s['policies'],dependencies={str(p):sha(p) for p in paths}))
    print('FROZEN',flush=True)

def background(index):
    verify(True);ex.c.OUT=OUT;ex.c.BASES={**ex.c.BASES,'test':SEED_BASE};ex.c.COUNTS={**ex.c.COUNTS,'test':N_FRESH}
    ex.c.verify=lambda:verify(True);ex.learn.verify_freeze=lambda:verify(True)
    ex.c.background('test',index)

def generate():
    verify(True);logs=OUT/'logs';logs.mkdir(exist_ok=True)
    pending=[i for i in range(N_FRESH) if not (OUT/f'datasets/test/{i:03d}/index.json').exists()];active=[]
    while pending or active:
        while pending and len(active)<WORKERS:
            i=pending.pop(0);f=(logs/f'fresh_{i:03d}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'background','--index',str(i)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active.append((p,f,i))
        for p,f,i in active[:]:
            if p.poll() is not None:
                f.close();active.remove((p,f,i));assert p.returncode==0,f'Background {i} failed';print('FRESH',i,flush=True)
        if active:time.sleep(2)
    rr=[r for i in range(N_FRESH) for r in read(OUT/f'datasets/test/{i:03d}/index.json')]
    assert len(rr)==N_FRESH*18;write(OUT/'datasets/test/index.json',rr)

def confirm_worker(row):
    path=OUT/'confirmation_scenes'/f"{row['scene_id']}.json"
    if path.exists():return
    # Computing all predefined maps is harmless; only frozen policies are scored.
    cache=scene_cache(row);policies=read(OUT/'frozen.json')['policies'];results={}
    for k,p in policies.items():
        op=p[str(row['count_multiplier'])]['op'] if k=='per_count' else p['op']
        results[k]=ex.result_row(cache['meta'],cache['truth'],predict(cache,op))
    write(path,results)

def confirm():
    verify(True);(OUT/'confirmation_scenes').mkdir(exist_ok=True)
    rr=read(OUT/'datasets/test/index.json')
    with concurrent.futures.ProcessPoolExecutor(WORKERS) as pool:
        for i,_ in enumerate(pool.map(confirm_worker,rr)):
            if (i+1)%24==0:print('CONFIRM',i+1,'/',len(rr),flush=True)
    methods={k:[] for k in read(OUT/'frozen.json')['policies']}
    for r in rr:
        for k,v in read(OUT/'confirmation_scenes'/f"{r['scene_id']}.json").items():methods[k].append(v)
    write(OUT/'confirmation.json',{k:dict(**ex.summary(v),rows=v) for k,v in methods.items()})
    print('CONFIRM COMPLETE',flush=True)

def selfcheck():
    assert abs(float(shift_rows(np.asarray(64.),'row')))<1e-12
    assert len(merge([[100,40,110,43,30],[101,52,111,55,30]],10,8))==1
    assert len(merge([[100,40,110,43,30],[101,52,111,55,30]],4,8))==2
    seam=merge([[5020,40,5024,43,12],[0,48,4,51,12]],10,8)
    assert len(seam)==1 and seam[0][2]-seam[0][0]==8
    n=5024;tt=np.arange(n)*2*np.pi/n;x=np.tile(.3+.05*np.cos(3*tt)+.02*np.sin(5*tt),(160,1)).astype('f4')
    assert np.max(abs(periodic_residual(x)))<1e-6
    # Correlation centre and wrap: a template inserted across the angular seam.
    rng=np.random.default_rng(10);t=rng.normal(size=(33,49)).astype('f4');t-=t.mean(1,keepdims=True);t/=np.linalg.norm(t)
    d=np.zeros((160,5024),'f4');d[np.arange(64,97)[:,None],(np.arange(-24,25)%5024)[None,:]]=t
    pp=template_candidates(d,np.asarray([t]),[dict(width=10,height=8,centre_offset=[0,0])])
    assert pp and pp[0][1]==.5 and pp[0][2]==64.5
    write(OUT/'selfcheck.json',dict(passed=True,checks=['10px merge and 4px separation','periodic seam merge','zero centre-row shift','periodic harmonic reconstruction','template correlation coordinate and seam']))
    print('SELFCHECK PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','development','select','freeze','background','generate','confirm','selfcheck']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();selfcheck()
    elif a.action=='background':background(a.index)
    else:globals()[a.action]()
