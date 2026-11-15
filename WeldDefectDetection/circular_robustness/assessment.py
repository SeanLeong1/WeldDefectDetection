"""Fixed-coordinate robustness scoring; no test-driven parameter selection."""
import gc
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from circular_study import generation as g,training,median,metrics,evaluation
from . import campaign as c

def sample_periodic(array,row,col):
    if row.min()<-1e-5 or row.max()>array.shape[-2]-1+1e-5:
        raise ValueError('Requested localization error exceeds saved valid image halo')
    padded=np.pad(array,((0,0),(2,2)),mode='wrap')
    return map_coordinates(padded,[np.clip(row,0,array.shape[-2]-1),col%array.shape[-1]+2],order=1,mode='nearest')

def input_ring(raw,error):
    if all(error[k]==0 for k in ('dx','dy','dr')):return raw[:,16:112].copy()
    theta=2*np.pi*np.arange(c.geo.COLS)/c.geo.COLS
    radius=40+error['dr']+(np.arange(96)-47.5)*.05
    x=error['dx']+radius[:,None]*np.cos(theta);y=error['dy']+radius[:,None]*np.sin(theta)
    row=(np.hypot(x,y)-40)/.05+63.5;col=np.arctan2(y,x)*c.geo.COLS/(2*np.pi)
    return np.stack([sample_periodic(a,row,col) for a in raw])

def restore(score,error):
    if all(error[k]==0 for k in ('dx','dy','dr')):return score.copy(),np.ones_like(score,bool)
    theta=2*np.pi*np.arange(c.geo.COLS)/c.geo.COLS
    radius=40+(np.arange(64)-31.5)*.05
    x=radius[:,None]*np.cos(theta)-error['dx'];y=radius[:,None]*np.sin(theta)-error['dy']
    row=(np.hypot(x,y)-(40+error['dr']))/.05+31.5;col=np.arctan2(y,x)*c.geo.COLS/(2*np.pi)
    valid=(row>=0)&(row<=63)
    value=sample_periodic(score,np.clip(row,0,63),col)
    return np.where(valid,value,0).astype('f4'),valid

def normalized(raw,norm):return np.sqrt(np.clip((raw[:1]-norm['low'])/(norm['high']-norm['low']),0,1)).astype('f4')

def row_result(score,truth,instances,meta,spec,coverage):
    mask,boxes=metrics.binary(score,spec['threshold'],spec['min_area'],spec['closing'])
    _,union_boxes=metrics.regions(truth)
    individual=[]
    for instance in instances:
        _,bb=metrics.regions(instance)
        if len(bb)!=1:raise ValueError('Each injected instance must have one connected periodic label')
        individual.extend(bb)
    return dict(scene_id=meta['scene_id'],background_id=meta['background_id'],variant=meta['variant'],
        source_class=meta['source_class'],amplitude=meta['amplitude'],**metrics.match(boxes,union_boxes,score.shape[1]),
        instance=metrics.match(boxes,individual,score.shape[1]),injected_instances=len(individual),
        truth_regions=len(union_boxes),predicted_regions=len(boxes),intersection=int(((mask>0)&(truth>0)).sum()),
        union=int(((mask>0)|(truth>0)).sum()),truth_pixels=int(truth.sum()),
        truth_pixels_outside_inspection_band=int(((truth>0)&~coverage).sum()),
        coverage_fraction=float(coverage.mean()),predicted_boxes=boxes,truth_boxes=union_boxes,instance_boxes=individual)

def summarize(rows):
    result=metrics.summary(rows)
    tp,fp,fn=[sum(r['instance'][k] for r in rows) for k in ('tp','fp','fn')]
    result.update(instance_tp=tp,instance_fp=fp,instance_fn=fn,instance_f1=2*tp/max(2*tp+fp+fn,1),
        instance_recall=tp/max(tp+fn,1),truth_pixels_outside_inspection_band=sum(r['truth_pixels_outside_inspection_band'] for r in rows))
    return result

def run():
    torch.set_num_threads(4);frozen=c.prepare();norm=c.read(g.OUT/'normalization.json')
    rows=c.read(c.OUT/'index.json');folder=c.OUT/'predictions';folder.mkdir(exist_ok=True)
    model=training.load_model(frozen['cnn'])
    # Nominal evaluates every scene. Localization sweeps use paired clean/single
    # scenes; the combined error also tests every multi-pore layout.
    for meta in rows:
        path=folder/(meta['scene_id']+'.json')
        if path.exists():continue
        assert c.sha(meta['npz_path'])==meta['npz_sha256']
        with np.load(meta['npz_path']) as a:
            raw=a['raw'];truth=a['mask'][32:96];instances=a['instances'][:,32:96]
        conditions=c.PERTURBATIONS if meta['variant'] in ('clean','single_keyhole','single_intrinsic') else [c.PERTURBATIONS[0],c.PERTURBATIONS[-1]]
        outputs={}
        for error in conditions:
            x=normalized(input_ring(raw,error),norm)
            data=dict(x=x[None],mask=truth[None],meta=[meta])
            scores=dict(cnn=training.predict(data,model,frozen['cnn']['config'])[0],median=median.scores(data,frozen['median']['spec'])[0])
            for method,score in scores.items():
                score,coverage=restore(score,error)
                for point in ('best','low_fa'):
                    spec=frozen[method]['selection'][point]
                    outputs[f'{error["name"]}/{method}/{point}']=row_result(score,truth,instances,meta,spec,coverage)
        c.write(path,outputs);print('assessed',meta['scene_id'],len(conditions),flush=True)
        del raw,truth,instances;gc.collect()
    groups={}
    for meta in rows:
        for key,value in c.read(folder/(meta['scene_id']+'.json')).items():groups.setdefault(key,[]).append(value)
    results={}
    for key,values in groups.items():
        results[key]=dict(overall=summarize(values),per_variant={v:summarize([r for r in values if r['variant']==v]) for v in c.VARIANTS if any(r['variant']==v for r in values)})
        # Comparable aggregate across every error: clean + the two singles.
        single=[r for r in values if r['variant'] in ('clean','single_keyhole','single_intrinsic')]
        results[key]['single_pore_population']=summarize(single)
        if key.endswith('/best'):results[key]['single_pore_intervals']=metrics.bootstrap(single,repeats=2000)
    c.write(c.OUT/'results.json',dict(results=results,backgrounds=c.COUNT,scenes=len(rows),
        scope='Frozen synthetic robustness, union localization and stricter individual-pore localization. Overlapping bodies may form one union region; neither detector is trained for instance segmentation.',
        primary_operating_point='best fixed on v4 validation; low_fa is secondary and also frozen',physical_validation=False))
    print('ROBUSTNESS COMPLETE',flush=True)

if __name__=='__main__':run()
