"""Prospective median region-formation study; previous studies stay unchanged.

Run with the project Python: median_region_growth_20261110.py all
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse, heapq, itertools, json, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'WeldDefectDetection'), str(ROOT/'DataSimulation')]
import median_localisation_20261110 as previous
from count_study import campaign as c, learning as learn
from raised_crown import learning as base
from circular_study import metrics

OUT = ROOT/'DataSimulation/data/output/median_region_growth_20261110_v1'
OLD = c.OUT
FRESH_BASE = 4111000000
FRESH_COUNT = 24
PERIOD = 5024
HIGH = (12., 16., 24.)
LOW_FRACTIONS = (.25, .5, .75)
RADII = (3, 6)
AREAS = (3, 8, 15)
GAPS = (0, 2, 4)
MAX_EXTENT = 40
read, write, sha = c.read, c.write, c.sha
cv2.setNumThreads(1)


def utc():
    return datetime.now(timezone.utc).isoformat()


def periodic_components(mask, seeds, signed):
    """Eight-connected components with a circular horizontal seam.

    Records retain signed evidence; neither sign is required for detection.
    """
    mask = np.asarray(mask, 'u1')
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n == 1:
        return []
    parent = np.arange(n)
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for y in range(mask.shape[0]):
        if lab[y, 0]:
            for yy in range(max(0, y-1), min(mask.shape[0], y+2)):
                if lab[yy, -1]:
                    parent[root(lab[yy, -1])] = root(lab[y, 0])
    mapping = np.array([root(k) for k in range(n)])
    seed_counts = np.bincount(mapping[lab[seeds]], minlength=n)
    pos = np.bincount(mapping[lab[signed > 0]], minlength=n)
    neg = np.bincount(mapping[lab[signed < 0]], minlength=n)
    groups = {}
    for k in range(1, n):
        groups.setdefault(int(mapping[k]), []).append(k)
    output = []
    width = mask.shape[1]
    for k, ids in groups.items():
        if not seed_counts[k]:
            continue
        ss = stats[ids]
        y = int(ss[:, 1].min()); bottom = int((ss[:, 1]+ss[:, 3]).max())
        if len(ids) == 1:
            x, _, w, _, _ = ss[0]
            x, w = int(x), int(w)
        else:
            # Only seam-connected labels need a wrapped minimum-width box.
            columns = np.unique(np.where(np.isin(lab[y:bottom], ids))[1])
            gaps = np.diff(np.r_[columns, columns[0]+width])
            j = int(np.argmax(gaps))
            x = int(columns[(j+1) % len(columns)])
            w = int(width-gaps[j]+1)
        output.append(dict(box=[x, y, x+w, bottom], area=int(ss[:, 4].sum()),
                           seeds=int(seed_counts[k]), positive=int(pos[k]), negative=int(neg[k]), fragments=1))
    return output


def grow_components(signed, high, low, radius):
    score = np.abs(signed)
    seeds = score >= high
    if not seeds.any():
        return []
    if radius:
        pad = np.pad(seeds.astype('u1'), ((0, 0), (radius, radius)), mode='wrap')
        near = cv2.dilate(pad, np.ones((2*radius+1, 2*radius+1), 'u1'),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)[:, radius:-radius] > 0
        allowed = (score >= low) & near
    else:
        allowed = seeds
    # A weak region must connect to a strong seed, within its bounded neighbourhood.
    return periodic_components(allowed, seeds, signed)


def merge_pair(a, b, gap, period):
    """Join nearby fragments sharing >= half the smaller transverse span."""
    ax, ay, ar, ab = a['box']
    best = None
    for offset in (-period, 0, period):
        bx, by, br, bb = b['box']; bx += offset; br += offset
        gx = max(0, max(ax, bx)-min(ar, br))
        gy = max(0, max(ay, by)-min(ab, bb))
        ox = max(0, min(ar, br)-max(ax, bx))
        oy = max(0, min(ab, bb)-max(ay, by))
        aligned = (gy <= gap and ox >= .5*min(ar-ax, br-bx)) or (gx <= gap and oy >= .5*min(ab-ay, bb-by))
        union = [min(ax, bx), min(ay, by), max(ar, br), max(ab, bb)]
        if not aligned or max(gx, gy) > gap or max(union[2]-union[0], union[3]-union[1]) > MAX_EXTENT:
            continue
        rank = (max(gx, gy), (union[2]-union[0])*(union[3]-union[1]), union)
        if best is None or rank < best:
            best = rank
    return best


def form_regions(components, min_area, gap, period=PERIOD):
    work = [dict(v) for v in components if max(v['box'][2]-v['box'][0], v['box'][3]-v['box'][1]) <= MAX_EXTENT]
    work.sort(key=lambda v: tuple(v['box']))
    if gap:
        active=dict(enumerate(work));bins={};heap=[]
        def cells(box,expand=0):
            x,y,right,bottom=box;keys=set()
            for shift in (-period,0,period):
                left=max(0,x+shift-expand);rr=min(period,right+shift+expand)
                if rr>left:
                    for xx in range(int(left)//32,(int(rr)-1)//32+1):
                        for yy in range((y-expand)//32,(bottom+expand-1)//32+1):keys.add((xx,yy))
            return keys
        def insert(i):
            for key in cells(active[i]['box']):bins.setdefault(key,set()).add(i)
        def queue(i,j):
            a,b=active[i],active[j]
            if tuple(a['box'])>tuple(b['box']):i,j=j,i;a,b=b,a
            rank=merge_pair(a,b,gap,period)
            if rank is not None:
                distance,area,box=rank
                heapq.heappush(heap,(distance,area,tuple(box),tuple(a['box']),tuple(b['box']),i,j))
        for i in active:insert(i)
        for i in active:
            neighbours=set().union(*(bins.get(key,set()) for key in cells(active[i]['box'],gap+1)))
            for j in neighbours:
                if j>i:queue(i,j)
        next_id=len(active)
        while heap:
            _,_,box,_,_,i,j=heapq.heappop(heap)
            if i not in active or j not in active:continue
            box=list(box);a,b=active[i],active[j]
            width = box[2]-box[0]; box[0] %= period; box[2] = box[0]+width
            merged = dict(box=box, **{k:a[k]+b[k] for k in ('area','seeds','positive','negative','fragments')})
            for k in (i,j):
                for key in cells(active[k]['box']):bins[key].discard(k)
                del active[k]
            active[next_id]=merged;insert(next_id)
            neighbours=set().union(*(bins.get(key,set()) for key in cells(box,gap+1)))
            for k in neighbours:
                if k!=next_id:queue(next_id,k)
            next_id+=1
        work=sorted(active.values(),key=lambda v:tuple(v['box']))
    return [v for v in work if v['area'] >= min_area]


def configs():
    for high in HIGH:
        yield dict(high=high, low=high, radius=0)
        for fraction, radius in itertools.product(LOW_FRACTIONS, RADII):
            yield dict(high=high, low=high*fraction, radius=radius)


def tag(config):
    return f"h{config['high']:g}_l{config['low']:g}_r{config['radius']}"


def prepare():
    c.verify(); previous.verify_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'protocol.json'
    if target.exists():
        return verify()
    old_frozen = read(OLD/'frozen_models.json')['arms']['mixed_96']
    files = {Path(__file__), Path(previous.__file__), OLD/'frozen_models.json', OLD/'normalization.json',
             previous.OUT/'frozen.json', previous.OUT/'selection.json',
             OLD/'development/mixed_96/median_sigma1.json', Path(old_frozen['cnn']['checkpoint']),
             ROOT/'WeldDefectDetection/tests/test_median_region_growth_20261110.py'}
    files.update(Path(p) for p in read(OLD/'protocol.json')['source_hashes'])
    files.update(Path(p) for p in read(OLD/'development_protocol.json')['source_hashes'])
    files.update(OLD/f'datasets/{split}/index.json' for split in ('train','val','test'))
    seeds = set(); acquisitions = set()
    for path in (ROOT/'DataSimulation/data/output').glob('*/datasets/*/index.json'):
        try:
            rows = read(path)
            if isinstance(rows, list):
                seeds.update(r['seed'] for r in rows if 'seed' in r)
                acquisitions.update(r['acquisition_seed'] for r in rows if 'acquisition_seed' in r)
        except (ValueError, KeyError, TypeError):
            continue
    new = set(range(FRESH_BASE, FRESH_BASE+FRESH_COUNT))
    assert not new & seeds
    assert not {s+repeat*100000000 for s in new for repeat in (0,1)} & acquisitions
    protocol = dict(created_utc=utc(), study=OUT.name,
        question='Can bounded weak-region growth and constrained fragment merging improve geometric pore localisation?',
        training='Reuse clean-training calibration from 96 backgrounds; no CNN retraining.',
        validation='576 scenes / 32 backgrounds. Historical confirmations motivate this study but are excluded from selection.',
        background='Unchanged eight-neighbour median, train-only normalisation and per-row median/MAD; Gaussian sigma fixed at 1 pixel.',
        signed_evidence='Retain signed calibrated residual and positive/negative pixel counts. Detection accepts either polarity; no mandatory bright-dark pair.',
        search=dict(high=HIGH, low_fractions=LOW_FRACTIONS, growth_radius_px=RADII, min_area=AREAS, merge_gap_px=GAPS,
                    max_extent_px=MAX_EXTENT, max_extent_mm=MAX_EXTENT*.05,
                    merge_rule='Greedy nearest compatible fragments; >=50% smaller transverse overlap; bounded union; area cutoff after merging.',
                    growth_rule='8-connected low-threshold pixels within Chebyshev radius of a high-threshold seed; must connect to seed.',
                    assumption='Bounded isolated surface pores at known 0.05 mm pitch and known annular coordinates; crowded pores not validated.'),
        families=['seed_only','growth','growth_merge'],
        objective='Mean detection F1 across 1x/4x/64x, then filled-box pixel IoU, fewer false positives, simpler configuration.',
        metric='Unchanged one-to-one circular box IoU >=0.5. Pixel overlap for new methods describes filled boxes, not segmentation.',
        selection='Best per family plus overall validation winner; all frozen before new confirmation generation.',
        confirmation=dict(backgrounds=FRESH_COUNT, seed_base=FRESH_BASE, scenes=FRESH_COUNT*18,
                          comparators=['original median','previous position+size median','frozen mixed-96 CNN'],
                          uncertainty='2000 paired background bootstrap resamples, conditional on selected methods; no multiplicity correction.'),
        historical_seed_collision_check='passed', sources={str(p):sha(p) for p in sorted(files)}, physical_validation=False)
    write(target, protocol)
    return protocol


def verify():
    p = read(OUT/'protocol.json')
    for path, digest in p['sources'].items():
        assert sha(path) == digest, path
    return p


def signed_score(image, spec):
    return ((learn.median_residual(image)-np.asarray(spec['centre'],'f4')) / np.asarray(spec['scale'],'f4'))[16:144]


def signed_scores(data, folder):
    spec = read(OLD/'development/mixed_96/median_sigma1.json')['spec']
    return learn.score_file(folder/'signed_sigma1.npy', data, lambda i:signed_score(data.image(i,1), spec))


def objective_key(trial):
    p = trial['config']
    return (trial['macro_f1'],trial.get('pixel_iou',-1),-trial['fp'], -p['gap'], -p['radius'], p['low'], p['min_area'])


def tune():
    verify()
    if (OUT/'selection.json').exists():
        return read(OUT/'selection.json')
    data = learn.Data(learn.selected_rows('val'), read(OLD/'normalization.json'))
    folder = OUT/'development'; folder.mkdir(exist_ok=True)
    scores = signed_scores(data, folder)
    source = np.load(OLD/'development/mixed_96/median_sigma1_val.npy', mmap_mode='r')
    for i in range(len(data.rows)):
        assert np.array_equal(np.abs(scores[i]),source[i]), ('signed score replay',i)
    del source
    trials = []
    for cfg in configs():
        cache = folder/f'components_{tag(cfg)}.json'
        if cache.exists():
            cc = read(cache)
        else:
            cc = [grow_components(scores[i],**cfg) for i in range(len(data.rows))]
            write(cache, cc)
        for gap in ((0,) if cfg['radius']==0 else GAPS):
            merged=[form_regions(comps,1,gap) for comps in cc]
            for area in AREAS:
                conf = dict(**cfg, min_area=area, gap=gap)
                family = 'seed_only' if cfg['radius']==0 else 'growth' if gap==0 else 'growth_merge'
                preds = [[v['box'] for v in regs if v['area']>=area] for regs in merged]
                trials.append(dict(family=family,config=conf,**previous.box_objective(data.rows,data.truth,preds)))
        print('TUNED',tag(cfg),len(trials),flush=True)
    best = {}
    for family in ('seed_only','growth','growth_merge'):
        subset = [t for t in trials if t['family']==family]
        maxf = max(t['macro_f1'] for t in subset)
        for t in subset:
            if abs(t['macro_f1']-maxf)<1e-12:
                cfg=t['config'];cc=read(folder/f'components_{tag(cfg)}.json')
                preds=[[v['box'] for v in form_regions(comps,cfg['min_area'],cfg['gap'])] for comps in cc]
                t['pixel_iou']=previous.corrected_pixel_iou(data,preds)
        best[family]=max(subset,key=objective_key)
    chosen=max(best,key=lambda k:objective_key(best[k]))
    write(folder/'trials.json',trials)
    result=dict(selected=best,chosen_family=chosen,trial_count=len(trials),signed_score_replay_scenes=len(data.rows))
    write(OUT/'selection.json',result)
    print('SELECTION',json.dumps(result),flush=True)
    return result


def freeze():
    verify()
    if (OUT/'frozen.json').exists():
        return verify_frozen()
    assert not (OUT/'datasets/test').exists()
    files=[OUT/'protocol.json',OUT/'selection.json',OUT/'development/trials.json']
    write(OUT/'frozen.json',dict(frozen_utc=utc(),selection=read(OUT/'selection.json'),
        original=read(OLD/'frozen_models.json')['arms']['mixed_96'],
        previous=read(previous.OUT/'selection.json')['combined'],
        dependencies={str(p):sha(p) for p in files}))
    return verify_frozen()


def verify_frozen():
    verify(); f=read(OUT/'frozen.json')
    for path,digest in f['dependencies'].items():
        assert sha(path)==digest,path
    return f


def background(index):
    verify_frozen()
    c.OUT=OUT;c.BASES={**c.BASES,'test':FRESH_BASE};c.COUNTS={**c.COUNTS,'test':FRESH_COUNT}
    c.verify=verify_frozen;learn.verify_freeze=verify_frozen
    c.background('test',index)


def generate():
    verify_frozen();logs=OUT/'logs';logs.mkdir(exist_ok=True)
    active=[];pending=[i for i in range(FRESH_COUNT) if not (OUT/f'datasets/test/{i:03d}/index.json').exists()]
    try:
        while pending or active:
            while pending and len(active)<3:
                i=pending.pop(0);fh=(logs/f'fresh_{i:03d}.log').open('a',encoding='utf8')
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'background','--index',str(i)],
                    stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((proc,fh,i))
            for proc,fh,i in active[:]:
                if proc.poll() is not None:
                    fh.close();active.remove((proc,fh,i))
                    if proc.returncode:raise RuntimeError(f'Fresh generation failed {i}; inspect log')
                    print('FRESH',i,flush=True)
            if active:time.sleep(2)
    finally:
        for proc,fh,_ in active:
            if proc.poll() is None:proc.terminate()
            fh.close()
    write(OUT/'datasets/test/index.json',[r for i in range(FRESH_COUNT) for r in read(OUT/f'datasets/test/{i:03d}/index.json')])


def prediction(signed, config):
    comps=grow_components(signed,config['high'],config['low'],config['radius'])
    return form_regions(comps,config['min_area'],config['gap'])


def result_row(meta, truth, truth_boxes, regions):
    pred=[v['box'] for v in regions];mask=previous.box_mask(pred)>0
    return dict(scene_id=meta['scene_id'],background_id=meta['background_id'],source_class=meta['source_class'],
        count_multiplier=meta['count_multiplier'],diameter_mm=meta['diameter_mm'],repeat=meta['repeat'],
        dwell_bin=meta['dwell_bin'],amplitude=meta['amplitude'],**metrics.match(pred,truth_boxes,PERIOD),
        intersection=int((mask&truth).sum()),union=int((mask|truth).sum()),predicted_pixels=int(mask.sum()),
        truth_pixels=int(truth.sum()),predicted_regions=len(pred),predicted_boxes=pred,truth_boxes=truth_boxes,
        signed_region_evidence=regions)


def evaluate():
    f=verify_frozen();data=learn.Data(read(OUT/'datasets/test/index.json'),read(OLD/'normalization.json'))
    folder=OUT/'confirmation';folder.mkdir(exist_ok=True)
    signed=signed_scores(data,folder)
    for family,t in f['selection']['selected'].items():
        path=folder/f'{family}.json'
        if path.exists():continue
        rows=[result_row(meta,data.masks[i][16:144]>0,data.truth[i],prediction(signed[i],t['config'])) for i,meta in enumerate(data.rows)]
        write(path,dict(name=family,config=t['config'],**learn.result_summary(rows)))
        print('RESULT',family,json.dumps(learn.objective(rows)),flush=True)
    for name in ('original','previous'):
        path=folder/f'{name}.json'
        if path.exists():continue
        rows=[]
        for i,meta in enumerate(data.rows):
            if name=='original':
                op=f['original']['median']['selection']['best']
                row=learn.component_rows(np.abs(signed[i]),data.masks[i][16:144],data.truth[i],meta,op['threshold'],op['closing'],[op['min_area']])[0]
            else:
                op=f['previous']['base'];_,boxes=metrics.binary(np.abs(signed[i]),op['threshold'],op['min_area'],op['closing'])
                pred=previous.transform(boxes,f['previous']['correction'])
                row=result_row(meta,data.masks[i][16:144]>0,data.truth[i],[dict(box=b) for b in pred])
            rows.append(row)
        write(path,dict(name=name,unchanged=True,**learn.result_summary(rows)))
        print('RESULT',name,json.dumps(learn.objective(rows)),flush=True)
    path=folder/'cnn.json'
    if not path.exists():
        spec=f['original']['cnn'];model=base.model_for(spec)
        scores=learn.score_file(folder/'cnn.npy',data,lambda i:base.predict_image(data.image(i),model))
        rows=learn.evaluate_scores(data,scores,spec['selection']['best'])
        write(path,dict(name='cnn',unchanged_frozen_model=True,**learn.result_summary(rows)))
        print('RESULT cnn',json.dumps(learn.objective(rows)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','tune','freeze','background','generate','evaluate','all']);parser.add_argument('--index',type=int,default=0)
    a=parser.parse_args()
    if a.action=='all':
        prepare();tune();freeze();generate();evaluate()
    elif a.action=='background':background(a.index)
    else:globals()[a.action]()


if __name__=='__main__':
    main()
