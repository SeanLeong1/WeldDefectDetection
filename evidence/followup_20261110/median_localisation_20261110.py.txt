"""Prospective median localisation follow-up; original campaign is read-only.

Run: .venv/Scripts/python.exe WeldDefectDetection/median_localisation_20261110.py all
Artifacts and restart receipts live in median_localisation_20261110_v1.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys, json, time, argparse, itertools, subprocess, concurrent.futures
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'WeldDefectDetection'), str(ROOT/'DataSimulation')]
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from count_study import campaign as c, learning as learn
from raised_crown import rendering as render, geometry as geo, learning as base
from circular_study import metrics

OUT = ROOT/'DataSimulation/data/output/median_localisation_20261110_v1'
OLD = c.OUT
DEV = OLD/'development/mixed_96'
SIGMAS = (0., .5, 1., 2., 4., 8.)
THRESHOLDS = (2., 4., 6., 8., 12., 16., 24., 32., 48., 64., 96.)
AREAS = (3, 8, 15, 24, 40, 64, 96)
CLOSINGS = (1, 3)
FRESH_BASE = 3111000000
FRESH_COUNT = 24
read, write, sha = c.read, c.write, c.sha
cv2.setNumThreads(1)

def utc(): return datetime.now(timezone.utc).isoformat()

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    c.verify()
    paths = {Path(__file__), OLD/'frozen_models.json', OLD/'normalization.json'}
    paths.update(Path(p) for p in read(OLD/'development_protocol.json')['source_hashes'])
    paths.update(Path(p) for p in read(OLD/'protocol.json')['source_hashes'])
    for split in ('train', 'val', 'test'): paths.add(OLD/f'datasets/{split}/index.json')
    paths.update(DEV.glob('median*.json'))
    protocol = dict(study=OUT.name, created_utc=utc(),
        purpose='Separate registration error, contrast localisation error, and finite tuning grid limits.',
        development='Original 96 training and 32 validation backgrounds. Old test is historical only, never used in this selection.',
        calibration='Recompute clean-training per-row median/MAD for all six Gaussian sigmas; verify against original.',
        alignment='Reconstruct native nearest-neighbour geometric masks from prescribed pore coordinates; compare exact pixels. Quantify paired noiseless contrast on training only.',
        search=dict(sigmas=SIGMAS, thresholds=THRESHOLDS, areas=AREAS, closings=CLOSINGS,
            stage1='Full expanded grid. Exact original connected-region predictions and metrics.',
            stage2='Best base configuration per sigma, then box calibration. Translation-only, extent-only and combined ablations.',
            radial_shift_px=[-6,-4,-2,0,2,4,6], angular_shift_px=[-4,-2,0,2,4],
            radial_rule=['constant', 'toward_crown'], size_scale=[1.,1.5,2.], padding_px=[0,1,2,3],
            box_semantics='Keep one rectangle per retained connected component; shift/expand about its centre; clip radial bounds and wrap angular positions. No truth-dependent inference or merging.',
            objective='Mean box F1 across 1x/4x/64x, then pixel IoU, fewer false positives, simpler correction. Corrected output masks are filled rectangles.'),
        confirmation=dict(backgrounds=FRESH_COUNT, seed_base=FRESH_BASE, scenes=FRESH_COUNT*18,
            generation_gate='Generate only after frozen.json exists and its dependency hashes pass.',
            comparators=['original mixed-96 median','retuned median','shift-only median','extent-only median','combined median','original frozen mixed-96 CNN'],
            scope='Exploratory synthetic confirmation; 24 independent backgrounds; paired noise variants are not independent specimens.'),
        sources={str(p):sha(p) for p in sorted(paths)}, physical_validation=False)
    target=OUT/'protocol.json'
    if target.exists():
        previous=read(target)
        assert previous['sources']==protocol['sources'], 'A registered dependency changed.'
        return previous
    seeds=set();acquisition=set()
    for path in (ROOT/'DataSimulation/data/output').glob('*/datasets/*/index.json'):
        try:
            rows=read(path)
            if isinstance(rows,list):
                seeds.update(r['seed'] for r in rows if 'seed' in r)
                acquisition.update(r['acquisition_seed'] for r in rows if 'acquisition_seed' in r)
        except (ValueError, TypeError, KeyError): pass
    new={FRESH_BASE+i for i in range(FRESH_COUNT)}
    assert not new & seeds
    assert not {s+r*100000000 for s in new for r in (0,1)} & acquisition
    protocol['historical_seed_collision_check']='passed'
    write(target,protocol)
    return protocol

def verify():
    p=read(OUT/'protocol.json')
    for path,digest in p['sources'].items(): assert sha(path)==digest,path
    return p

def diagnostics():
    verify()
    target=OUT/'alignment.json'
    if target.exists(): return read(target)
    rows=[r for r in learn.selected_rows('train') if r['repeat']==0 and r['count_multiplier']==64 and r['source_class']!='clean']
    # Reconstruct Cartesian nearest-pixel coordinates independently of extract().
    n=render.old.N; theta=np.arange(5024)*2*np.pi/5024
    radius=40+(np.arange(160)-79.5)*.05
    xx=radius[:,None]*np.cos(theta); yy=radius[:,None]*np.sin(theta)
    # Preserve the renderer's addition/division order at half-pixel ties.
    # Algebraic reassociation changes round-off at the angular seam.
    xx=(np.floor((xx+(n-1)*.025)/.05+.5)-(n-1)/2)*.05
    yy=(np.floor((yy+(n-1)*.025)/.05+.5)-(n-1)/2)*.05
    uu=np.hypot(xx,yy)-40; tt=np.arctan2(yy,xx)
    records=[]; mismatches=0
    for row in rows:
        p=row['parameters']['defects'][0]['params']
        q,_,_=geo.pore_field(uu,tt,p)
        expected=np.sqrt(np.maximum(1-q,0))>.35
        with np.load(row['geometry_path']) as z: mask=z['mask'].astype(bool); quiet=z['quiet']
        with np.load(Path(row['geometry_path']).with_name('clean_geometry.npz')) as z: clean=z['quiet']
        difference=np.abs(quiet-clean)
        mismatch=int(np.count_nonzero(mask!=expected));mismatches+=mismatch
        # Oracle crop used only to diagnose signal footprint, never for inference.
        centre=int(round(p['angle_rad']*5024/(2*np.pi)))%5024
        cols=(np.arange(-24,25)+centre)%5024
        m=mask[:,cols]; d=difference[:,cols]
        y,x=np.nonzero(m); ymin=max(0,int(y.min())-8); ymax=min(160,int(y.max())+9)
        d[:ymin]=0;d[ymax:]=0
        gy,gx=np.indices(d.shape);mass=float(d.sum())
        dy=float((gy*d).sum()/max(mass,1e-20)-y.mean())
        dx=float((gx*d).sum()/max(mass,1e-20)-x.mean())
        peak=np.unravel_index(d.argmax(),d.shape)
        records.append(dict(scene_id=row['scene_id'],radial_offset_mm=p['radial_offset_mm'],angle_rad=p['angle_rad'],
            diameter_mm=p['diameter_mm'],mask_mismatch_pixels=mismatch,contrast_centroid_dy_px=dy,
            contrast_centroid_dx_px=dx,peak_inside_mask=bool(m[peak]),
            contrast_fraction_inside=float(d[m].sum()/max(mass,1e-20)),geometry_path=row['geometry_path']))
    groups={}
    for off in (-1.05,0.,1.05):
        rr=[r for r in records if r['radial_offset_mm']==off]
        groups[str(off)]=dict(n=len(rr),median_dy=float(np.median([r['contrast_centroid_dy_px'] for r in rr])),
            dy_10_90=np.quantile([r['contrast_centroid_dy_px'] for r in rr],[.1,.9]).tolist(),
            median_dx=float(np.median([r['contrast_centroid_dx_px'] for r in rr])),
            median_fraction_inside=float(np.median([r['contrast_fraction_inside'] for r in rr])),
            peak_inside_fraction=float(np.mean([r['peak_inside_mask'] for r in rr])))
    result=dict(geometries=len(records),native_nearest_pixel_mask_mismatches=mismatches,
        registration_check='passed' if mismatches==0 else 'investigate',by_radial_position=groups,rows=records,
        interpretation='Signal centroid is measured in a truth-centred diagnostic crop. It does not prescribe an inference correction.')
    write(target,result);print('ALIGNMENT',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result

def compact_components(score,threshold,closing):
    mask=(score>=threshold).astype('u1')
    if closing>1:
        mask=cv2.morphologyEx(np.pad(mask,((0,0),(closing,closing)),mode='wrap'),cv2.MORPH_CLOSE,np.ones((closing,closing),'u1'))[:,closing:-closing]
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if not (mask[:,0].any() and mask[:,-1].any()):
        return [[int(x),int(y),int(x+w),int(y+h),int(a)] for x,y,w,h,a in stats[1:] if a>=min(AREAS)]
    # Merge periodic components with identical connectivity to metrics.regions.
    parent=np.arange(n)
    def root(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for y in range(mask.shape[0]):
        if lab[y,0]:
            for yy in range(max(0,y-1),min(mask.shape[0],y+2)):
                if lab[yy,-1]:parent[root(lab[yy,-1])]=root(lab[y,0])
    mapping=np.array([root(i) for i in range(n)])
    seamroots=set(mapping[lab[:,0][lab[:,0]>0]]) & set(mapping[lab[:,-1][lab[:,-1]>0]])
    components=[]
    for k in range(1,n):
        if mapping[k]!=k:continue
        members=np.flatnonzero(mapping==k)
        area=int(stats[members,4].sum())
        if area<min(AREAS):continue
        if k in seamroots:
            ys,xs=np.where(np.isin(lab,members));columns=np.unique(xs)
            gaps=np.diff(np.r_[columns,columns[0]+mask.shape[1]]);j=int(np.argmax(gaps))
            x=int(columns[(j+1)%len(columns)]);w=int(mask.shape[1]-gaps[j]+1)
            box=[x,int(ys.min()),x+w,int(ys.max()+1)]
        else:
            x,y,w,h,_=stats[k];box=list(map(int,(x,y,x+w,y+h)))
        components.append([*box,area])
    return components

def fit_sigma(sigma):
    verify();tag=f'{sigma:g}';target=OUT/f'development/sigma{tag}.json'
    if target.exists():return
    norm=read(OLD/'normalization.json');train=learn.Data(learn.selected_rows('train','mixed_96'),norm)
    spec=learn.median_fit(train,sigma)
    previous=read(DEV/f'median_sigma{tag}.json')['spec']
    assert np.array_equal(spec['centre'],previous['centre']) and np.array_equal(spec['scale'],previous['scale'])
    write(OUT/f'development/calibration_sigma{tag}.json',spec)
    del train
    data=learn.Data(learn.selected_rows('val'),norm)
    source=DEV/f'median_sigma{tag}_val.npy';receipt=read(source.with_suffix('.done.json'))
    assert receipt['scene_ids']==[r['scene_id'] for r in data.rows]
    scores=np.load(source,mmap_mode='r')
    # Recompute one independent saved-score replay for each count level.
    for i in (0,1,2):
        actual=learn.median_score(data.image(i,sigma),spec)
        assert np.array_equal(actual,scores[i]), 'Saved score replay mismatch'
    trials=[]
    for threshold,closing in itertools.product(THRESHOLDS,CLOSINGS):
        path=OUT/f'components/s{tag}_t{threshold:g}_c{closing}.json'
        if path.exists():components=read(path)
        else:
            components=[compact_components(scores[i],threshold,closing) for i in range(len(data.rows))]
            write(path,components)
        # Primary objective can be evaluated directly from exact original boxes.
        for area in AREAS:
            pred=[[z[:4] for z in cc if z[4]>=area] for cc in components]
            summary=box_objective(data.rows,data.truth,pred)
            trials.append(dict(sigma=sigma,threshold=threshold,closing=closing,min_area=area,**summary))
        print('GRID',tag,threshold,closing,flush=True)
    maxf=max(t['macro_f1'] for t in trials)
    tied=[t for t in trials if abs(t['macro_f1']-maxf)<1e-12]
    for t in tied:
        rr=learn.evaluate_scores(data,scores,t);t['pixel_iou']=metrics.summary(rr)['pixel_iou']
    best=max(tied,key=lambda t:(t['macro_f1'],t['pixel_iou'],-t['fp']))
    write(target,dict(spec=spec,calibration_recomputed_equal=True,score_replay=True,best=best,trials=trials))
    print('BEST',tag,json.dumps(best),flush=True)

def box_objective(rows,truth,preds):
    sums={m:np.zeros(3,dtype=int) for m in c.MULTIPLIERS}
    for row,t,p in zip(rows,truth,preds):
        match=metrics.match(p,t,5024);sums[row['count_multiplier']]+=np.array([match['tp'],match['fp'],match['fn']])
    by={str(m):dict(tp=int(v[0]),fp=int(v[1]),fn=int(v[2]),f1=float(2*v[0]/max(2*v[0]+v[1]+v[2],1))) for m,v in sums.items()}
    return dict(macro_f1=float(np.mean([v['f1'] for v in by.values()])),fp=sum(v['fp'] for v in by.values()),by_count=by)

def transform(boxes,correction):
    result=[]
    for x0,y0,x1,y1 in boxes:
        cx=(x0+x1)/2;cy=(y0+y1)/2
        direction=1 if correction['radial_rule']=='constant' else np.sign(64-cy)
        cx+=correction['dx'];cy+=correction['dy']*direction
        w=(x1-x0)*correction['scale']+2*correction['pad'];h=(y1-y0)*correction['scale']+2*correction['pad']
        x=int(np.floor(cx-w/2));right=int(np.ceil(cx+w/2));y=max(0,int(np.floor(cy-h/2)));bottom=min(128,int(np.ceil(cy+h/2)))
        if bottom>y:result.append([x%5024,y,x%5024+min(5024,right-x),bottom])
    return result

def box_mask(boxes):
    out=np.zeros((128,5024),'u1')
    for x,y,xx,yy in boxes:out[y:yy,np.arange(x,xx)%5024]=1
    return out

def correction_identity():return dict(radial_rule='constant',dy=0,dx=0,scale=1.,pad=0)

def corrected_pixel_iou(data,preds):
    inter=union=0
    for i,p in enumerate(preds):
        a=box_mask(p)>0;b=data.masks[i][16:144]>0
        inter+=int((a&b).sum());union+=int((a|b).sum())
    return inter/max(union,1)

def tune_corrections():
    verify();target=OUT/'selection.json'
    if target.exists():return read(target)
    data=learn.Data(learn.selected_rows('val'),read(OLD/'normalization.json'))
    candidates=[read(OUT/f'development/sigma{s:g}.json')['best'] for s in SIGMAS]
    retuned=max(candidates,key=lambda t:(t['macro_f1'],t['pixel_iou'],-t['fp']))
    selected={};all_trials=[]
    for family in ('shift','extent','combined'):
        trials=[]
        for config in candidates:
            path=OUT/f"components/s{config['sigma']:g}_t{config['threshold']:g}_c{config['closing']}.json"
            boxes=[[z[:4] for z in cc if z[4]>=config['min_area']] for cc in read(path)]
            shifts=[('constant',0,0)] if family=='extent' else [(rule,dy,dx) for rule in ('constant','toward_crown') for dy in (-6,-4,-2,0,2,4,6) for dx in (-4,-2,0,2,4) if not (rule=='toward_crown' and dy==0)]
            extents=[(1.,0)] if family=='shift' else list(itertools.product((1.,1.5,2.),(0,1,2,3)))
            for (rule,dy,dx),(scale,pad) in itertools.product(shifts,extents):
                corr=dict(radial_rule=rule,dy=dy,dx=dx,scale=scale,pad=pad)
                pred=[transform(b,corr) for b in boxes]
                trials.append(dict(family=family,base=config,correction=corr,**box_objective(data.rows,data.truth,pred)))
            print('CORRECTION',family,config['sigma'],flush=True)
        maxf=max(t['macro_f1'] for t in trials)
        tied=[t for t in trials if abs(t['macro_f1']-maxf)<1e-12]
        for t in tied:
            a=t['base'];path=OUT/f"components/s{a['sigma']:g}_t{a['threshold']:g}_c{a['closing']}.json"
            pred=[transform([z[:4] for z in cc if z[4]>=a['min_area']],t['correction']) for cc in read(path)]
            t['pixel_iou']=corrected_pixel_iou(data,pred)
        def key(t):
            r=t['correction'];return (t['macro_f1'],t['pixel_iou'],-t['fp'],-abs(r['dy'])-abs(r['dx'])-r['pad']-r['scale'])
        selected[family]=max(tied,key=key);all_trials.extend(trials)
        print('SELECTED',family,json.dumps(selected[family]),flush=True)
    result=dict(retuned=retuned,**selected)
    result['chosen_family']=max(result,key=lambda k:(result[k]['macro_f1'],result[k]['pixel_iou'],-result[k]['fp']))
    write(OUT/'correction_trials.json',all_trials);write(target,result)
    return result

def freeze():
    verify();selection=read(OUT/'selection.json');path=OUT/'frozen.json'
    if path.exists():return verify_frozen()
    assert not (OUT/'datasets/test').exists()
    files=[OUT/'protocol.json',OUT/'alignment.json',OUT/'selection.json',Path(__file__)]
    files+=list((OUT/'development').glob('*.json'))
    original=read(OLD/'frozen_models.json')['arms']['mixed_96']
    checkpoint=Path(original['cnn']['checkpoint'])
    assert sha(checkpoint)==original['cnn']['checkpoint_sha256']
    files.append(checkpoint)
    write(path,dict(frozen_utc=utc(),selection=selection,original=original,
        dependencies={str(p):sha(p) for p in files},source_protocol_sha256=sha(OUT/'protocol.json')))
    return verify_frozen()

def verify_frozen():
    verify();f=read(OUT/'frozen.json')
    for path,digest in f['dependencies'].items():assert sha(path)==digest,path
    return f

def fresh_background(index):
    verify_frozen()
    # Delegate unchanged generation to the original function, redirecting only
    # output, seeds and the two verification callbacks in this worker process.
    c.OUT=OUT;c.BASES={**c.BASES,'test':FRESH_BASE};c.COUNTS={**c.COUNTS,'test':FRESH_COUNT}
    c.verify=verify_frozen;learn.verify_freeze=verify_frozen
    c.background('test',index)

def generate():
    verify_frozen();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    active=[];pending=[i for i in range(FRESH_COUNT) if not (OUT/f'datasets/test/{i:03d}/index.json').exists()]
    while pending or active:
        while pending and len(active)<3:
            i=pending.pop(0);fh=(logs/f'fresh_{i:03d}.log').open('a',encoding='utf8')
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'background','--index',str(i)],stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((p,fh,i))
        for p,fh,i in active[:]:
            if p.poll() is not None:
                fh.close();active.remove((p,fh,i))
                if p.returncode:raise RuntimeError(f'Fresh generation failed {i}; see log')
                print('FRESH',i,flush=True)
        if active:time.sleep(2)
    write(OUT/'datasets/test/index.json',[r for i in range(FRESH_COUNT) for r in read(OUT/f'datasets/test/{i:03d}/index.json')])

def corrected_rows(data,scores,config,correction):
    rows=[]
    for i,meta in enumerate(data.rows):
        _,boxes=metrics.binary(scores[i],config['threshold'],config['min_area'],config['closing'])
        pred=transform(boxes,correction);mask=box_mask(pred);truth=data.masks[i][16:144]
        inter=int(((mask>0)&(truth>0)).sum());union=int(((mask>0)|(truth>0)).sum())
        rows.append(dict(scene_id=meta['scene_id'],background_id=meta['background_id'],source_class=meta['source_class'],count_multiplier=meta['count_multiplier'],
            diameter_mm=meta['diameter_mm'],repeat=meta['repeat'],dwell_bin=meta['dwell_bin'],amplitude=meta['amplitude'],
            **metrics.match(pred,data.truth[i],5024),intersection=inter,union=union,predicted_pixels=int(mask.sum()),truth_pixels=int(truth.sum()),
            predicted_regions=len(pred),predicted_boxes=pred,truth_boxes=data.truth[i]))
    return rows

def evaluate():
    frozen=verify_frozen();data=learn.Data(read(OUT/'datasets/test/index.json'),read(OLD/'normalization.json'))
    folder=OUT/'confirmation';folder.mkdir(exist_ok=True)
    selection=frozen['selection'];original=frozen['original'];specs={'original':original['median']['selection']['best'],'retuned':selection['retuned']}
    specs['original']={**specs['original'],'sigma':original['median']['spec']['sigma']}
    for family in ('shift','extent','combined'):specs[family]=selection[family]['base']
    for sigma in sorted({s['sigma'] for s in specs.values()}):
        spec=read(OUT/f'development/calibration_sigma{sigma:g}.json')
        scores=learn.score_file(folder/f'median_sigma{sigma:g}.npy',data,lambda i:learn.median_score(data.image(i,sigma),spec))
        for name,op in specs.items():
            if op['sigma']!=sigma or (folder/f'{name}.json').exists():continue
            rows=learn.evaluate_scores(data,scores,op) if name in ('original','retuned') else corrected_rows(data,scores,op,selection[name]['correction'])
            write(folder/f'{name}.json',dict(name=name,operating_point=op,**learn.result_summary(rows)))
            print('RESULT',name,json.dumps(learn.objective(rows)),flush=True)
    path=folder/'cnn.json'
    if not path.exists():
        spec=original['cnn'];model=base.model_for(spec)
        scores=learn.score_file(folder/'cnn.npy',data,lambda i:base.predict_image(data.image(i),model))
        rows=learn.evaluate_scores(data,scores,spec['selection']['best'])
        write(path,dict(name='cnn',unchanged_frozen_model=True,**learn.result_summary(rows)))
        print('RESULT cnn',json.dumps(learn.objective(rows)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','diagnose','fit','correct','freeze','background','generate','evaluate','all']);p.add_argument('--sigma',type=float,default=1);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='diagnose':diagnostics()
    elif a.action=='fit':fit_sigma(a.sigma)
    elif a.action=='correct':tune_corrections()
    elif a.action=='freeze':freeze()
    elif a.action=='background':fresh_background(a.index)
    elif a.action=='generate':generate()
    elif a.action=='evaluate':evaluate()
    else:
        prepare();diagnostics()
        for sigma in SIGMAS:fit_sigma(sigma)
        tune_corrections();freeze();generate();evaluate()

if __name__=='__main__':main()
