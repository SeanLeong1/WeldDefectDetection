"""Fresh, frozen evaluation and paired acquisition-stratified uncertainty."""
from pathlib import Path
import gc, time
import numpy as np
import torch
from circular_study import metrics
from raised_crown import campaign as prior, learning as base, assessment as old, rendering as r
from . import campaign as c, learning as learn


def grouped_intervals(rows,repeats=2000,seed=10606001):
    groups={key:[x for x in rows if x['background_id']==key] for key in sorted({x['background_id'] for x in rows})}
    bins={b:[key for key,group in groups.items() if group[0]['dwell_bin']==b] for b in range(4)}
    rng=np.random.default_rng(seed); values=[]
    for _ in range(repeats):
        sample=[x for keys in bins.values() if keys for key in rng.choice(keys,len(keys),replace=True) for x in groups[key]]
        summary=metrics.summary(sample); values.append([summary['f1'],summary['recall'],summary['clean_ring_fa']])
    q=np.quantile(values,[.025,.975],axis=0)
    return {key:q[:,i].tolist() for i,key in enumerate(('f1','recall','clean_ring_fa'))}


def paired_difference(a,b,repeats=2000,seed=10606002):
    """A minus B, matching scenes and resampling independent backgrounds within bins."""
    assert {x['scene_id'] for x in a}=={x['scene_id'] for x in b}
    groups_a={key:[x for x in a if x['background_id']==key] for key in sorted({x['background_id'] for x in a})}
    groups_b={key:[x for x in b if x['background_id']==key] for key in groups_a}
    bins={k:[key for key,rows in groups_a.items() if rows[0]['dwell_bin']==k] for k in range(4)}
    rng=np.random.default_rng(seed); values=[]; fields=('f1','recall','clean_ring_fa')
    for _ in range(repeats):
        keys=[key for members in bins.values() if members for key in rng.choice(members,len(members),replace=True)]
        aa=metrics.summary([x for key in keys for x in groups_a[key]])
        bb=metrics.summary([x for key in keys for x in groups_b[key]])
        values.append([aa[f]-bb[f] for f in fields])
    q=np.quantile(values,[.025,.975],axis=0); aa=metrics.summary(a); bb=metrics.summary(b)
    return {f:dict(difference=aa[f]-bb[f],interval_95=q[:,i].tolist()) for i,f in enumerate(fields)}


def summarize(rows):
    hidden=[x for x in rows if x['source_class']=='hidden']
    return dict(metrics=metrics.summary(rows),intervals=grouped_intervals(rows),rows=rows,
        by_dwell_bin={str(b):metrics.summary([x for x in rows if x['dwell_bin']==b]) for b in range(4)},
        by_class={kind:metrics.summary([x for x in rows if x['source_class']==kind]) for kind in sorted({x['source_class'] for x in rows})},
        without_duplicate_hidden=metrics.summary([x for x in rows if x['source_class']!='hidden']),
        hidden_negative_rings=len(hidden),hidden_rings_flagged=sum(x['predicted_regions']>0 for x in hidden))


def methods(frozen):
    out={}
    for arm,spec in frozen['arms'].items():
        for candidate in spec['cnn_candidates']:
            out[f'{arm}_{candidate["seed"]}']=dict(kind='cnn',spec=candidate,norm=c.OUT/'normalization.json')
    out['median']=dict(kind='median',spec=frozen['median'],norm=c.OUT/'normalization.json')
    old_frozen=base.verify_freeze()
    for kind in ('cnn','median'):
        out['frozen_v5_'+kind]=dict(kind=kind,spec=old_frozen[kind],norm=prior.OUT/'normalization.json')
    return out


def evaluate_method(name):
    torch.set_num_threads(4); frozen=learn.verify_freeze(); c.verify()
    folder=c.OUT/'predictions'; folder.mkdir(exist_ok=True); path=folder/(name+'.json')
    descriptor=methods(frozen)[name]
    inputs=dict(frozen_sha256=c.sha(c.OUT/'frozen_models.json'),test_index_sha256=c.sha(c.OUT/'datasets/test/index.json'))
    if path.exists():
        result=c.read(path); assert result['inputs']==inputs; return result
    data=learn.load('test',c.read(descriptor['norm'])); spec=descriptor['spec']; started=time.monotonic()
    if descriptor['kind']=='cnn': scores=base.predict(data,base.model_for(spec))
    else: scores=np.stack([base.median_image(x[0],spec['spec']) for x in data['x']])
    rows=metrics.evaluate(data,scores,spec['selection']['best'],retain=True)
    for row,meta in zip(rows,data['meta']):
        row.update(dwell_bin=meta['dwell_bin'],dwell_time_us=meta['dwell_time_us'])
    result=summarize(rows); result.update(inputs=inputs,inference_seconds=time.monotonic()-started)
    c.write(path,result); print(name,result['metrics'],flush=True); return result


def stress(frozen,partition=None):
    folder=c.OUT/'stress_predictions'; folder.mkdir(exist_ok=True)
    models={arm:base.model_for(frozen['arms'][arm]['cnn']) for arm in c.ARMS}
    norm=c.read(c.OUT/'normalization.json'); records=[]
    for meta in c.read(c.OUT/'datasets/stress/index.json'):
        if partition is not None and meta['index']%partition[1]!=partition[0]: continue
        receipt=folder/(meta['scene_id']+'.json')
        if receipt.exists(): records.extend(c.read(receipt)); continue
        with np.load(meta['npz_path']) as a:
            native=a['native']; mask=a['mask'][r.SCORE]; instances=a['instances'][:,r.SCORE]; nominal=a['raw'][0]
        data=dict(mask=mask[None],meta=[meta]); scene=[]
        errors=[dict(name='nominal',dx=0.,dy=0.,dr=0.),old.estimated_error(native)]
        if meta['source_class'] in ('clean','pair_near'):
            errors.append(dict(name='combined',dx=.25,dy=-.25,dr=.25))
        for error in errors:
            image=nominal if error['name']=='nominal' else r.extract(native,error=error)
            x=base.normalize(image,norm)
            scores={arm:base.predict_image(x,model) for arm,model in models.items()}
            scores['median']=base.median_image(x,frozen['median']['spec'])
            for method,score in scores.items():
                mapped=score if error['name']=='nominal' else old.map_back(score,error)
                spec=frozen['median'] if method=='median' else frozen['arms'][method]['cnn']
                row=metrics.evaluate(data,mapped[None],spec['selection']['best'],retain=True)[0]
                truth=[box for m in instances for box in metrics.regions(m)[1]]
                scene.append(dict(method=method,error=error,dwell_bin=meta['dwell_bin'],dwell_time_us=meta['dwell_time_us'],
                    instance_metrics=metrics.match(row['predicted_boxes'],truth,5024),instance_count=len(truth),**row))
        c.write(receipt,scene); records.extend(scene); print('Stress',meta['scene_id'],flush=True)
    if partition is not None: return
    groups={}
    for method in (*c.ARMS,'median'):
        groups[method]={}
        for condition in ('nominal','image_estimated','combined'):
            rows=[x for x in records if x['method']==method and x['error']['name']==condition]
            n=sum(x['instance_count'] for x in rows); tp=sum(x['instance_metrics']['tp'] for x in rows)
            groups[method][condition]=dict(**summarize(rows),individual_recall=tp/max(n,1),individual_tp=tp,individual_count=n)
        matched=[x for x in records if x['method']==method and x['error']['name']=='nominal' and x['source_class'] in ('clean','pair_near')]
        groups[method]['nominal_combined_subset']=summarize(matched)
    c.write(c.OUT/'robustness_results.json',dict(groups=groups,rows=records,physical_validation=False,
        note='Combined offset only clean and close pairs: compare nominal_combined_subset, not all nominal scenes.'))


def run():
    import subprocess,sys
    torch.set_num_threads(4); c.verify(); frozen=learn.verify_freeze(); names=list(methods(frozen)); active=[]
    while names or active:
        while names and len(active)<3:
            name=names.pop(0); handle=(c.OUT/'logs'/f'eval_{name}.log').open('a',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,str(c.ROOT/'WeldDefectDetection/run_raised_acquisition_evaluation.py'),name],
                cwd=c.ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((proc,handle,name))
        for proc,handle,name in active[:]:
            if proc.poll() is not None:
                handle.close(); active.remove((proc,handle,name))
                if proc.returncode: raise RuntimeError('Evaluation failed: '+name)
                print('Evaluated',name,flush=True)
        if active: time.sleep(3)
    results={name:c.read(c.OUT/'predictions'/f'{name}.json') for name in methods(frozen)}
    selected={arm:f'{arm}_{frozen["arms"][arm]["cnn"]["seed"]}' for arm in c.ARMS}
    comparisons={}
    for label,a,b in [('hard_minus_coverage',selected['hard_negative'],selected['coverage']),
                      ('coverage_minus_frozen',selected['coverage'],'frozen_v5_cnn'),
                      ('hard_minus_frozen',selected['hard_negative'],'frozen_v5_cnn')]:
        comparisons[label]=paired_difference(results[a]['rows'],results[b]['rows'])
    c.write(c.OUT/'results.json',dict(results=results,selected=selected,selected_arm=frozen['selected_arm'],paired_differences=comparisons,
        confidence='2000 paired background-cluster bootstrap replicates, resampling within dwell bins',physical_validation=False))
    if not (c.OUT/'robustness_results.json').exists():
        processes=[]; handles=[]
        for partition in range(3):
            handle=(c.OUT/'logs'/f'stress_eval_{partition}.log').open('a',encoding='utf-8'); handles.append(handle)
            processes.append(subprocess.Popen([sys.executable,str(c.ROOT/'WeldDefectDetection/run_raised_acquisition_evaluation.py'),
                'stress',str(partition),'3'],cwd=c.ROOT,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
        codes=[proc.wait() for proc in processes]
        for handle in handles: handle.close()
        if any(codes): raise RuntimeError('Stress evaluation failed; see logs')
        stress(frozen)
