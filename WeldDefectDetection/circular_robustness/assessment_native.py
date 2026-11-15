"""Primary robustness scoring with direct Cartesian ring extraction."""
import gc
import numpy as np
import torch
from circular_study import generation as g,training,median,metrics
from . import campaign as c,assessment as a,native_inputs as native

def run():
    torch.set_num_threads(4);frozen=c.prepare();native.prepare();norm=c.read(g.OUT/'normalization.json')
    records=c.read(c.OUT/'index.json');folder=c.OUT/'predictions_native';folder.mkdir(exist_ok=True)
    model=training.load_model(frozen['cnn'])
    for meta in records:
        path=folder/(meta['scene_id']+'.json')
        if path.exists():continue
        receipt=c.read(c.OUT/'native_inputs'/(meta['scene_id']+'.json'))
        assert c.sha(receipt['path'])==receipt['sha256']
        with np.load(meta['npz_path']) as z:truth=z['mask'][32:96];instances=z['instances'][:,32:96]
        with np.load(receipt['path']) as z:source=z['topo']
        conditions=c.PERTURBATIONS if meta['variant'] in ('clean','single_keyhole','single_intrinsic') else [c.PERTURBATIONS[0],c.PERTURBATIONS[-1]]
        outputs={}
        for error in conditions:
            x=a.normalized(native.extract(source,error),norm)
            data=dict(x=x[None],mask=truth[None],meta=[meta])
            scores=dict(cnn=training.predict(data,model,frozen['cnn']['config'])[0],median=median.scores(data,frozen['median']['spec'])[0])
            for method,score in scores.items():
                score,coverage=a.restore(score,error)
                for point in ('best','low_fa'):
                    outputs[f'{error["name"]}/{method}/{point}']=a.row_result(score,truth,instances,meta,frozen[method]['selection'][point],coverage)
        c.write(path,outputs);print('assessed native',meta['scene_id'],len(conditions),flush=True)
        del source,truth,instances;gc.collect()
    groups={}
    for meta in records:
        for key,row in c.read(folder/(meta['scene_id']+'.json')).items():groups.setdefault(key,[]).append(row)
    results={}
    for key,values in groups.items():
        single=[r for r in values if r['variant'] in ('clean','single_keyhole','single_intrinsic')]
        results[key]=dict(overall=a.summarize(values),per_variant={v:a.summarize([r for r in values if r['variant']==v]) for v in c.VARIANTS if any(r['variant']==v for r in values)},single_pore_population=a.summarize(single))
        if key.endswith('/best'):results[key]['single_pore_intervals']=metrics.bootstrap(single,repeats=2000)
    c.write(c.OUT/'results.json',dict(results=results,backgrounds=c.COUNT,scenes=len(records),
        active_protocol='protocol_native_v2.json',input_extraction='Direct native Cartesian TOPO; no saved-annulus resampling',
        scope='Frozen synthetic robustness; primary union localization, secondary individual-pore localization. Overlap does not imply instance-segmentation ability.',
        primary_operating_point='best fixed on v4 validation; low_fa is secondary and frozen',physical_validation=False))
    print('NATIVE ROBUSTNESS COMPLETE',flush=True)

if __name__=='__main__':run()
