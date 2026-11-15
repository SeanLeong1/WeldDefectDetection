"""One-factor post-freeze acquisition diagnostic on an identified validation case."""
from dataclasses import replace
import numpy as np
import torch
from . import campaign as c,rendering as r,learning as learn
from circular_study import metrics


def run():
    path=c.OUT/'dwell_diagnostic.json'
    if path.exists():return c.read(path)
    torch.set_num_threads(4);frozen=learn.verify_freeze();c.verify_sources()
    selection=c.read(c.OUT/'selection_diagnostic.json');sid=selection['worst_clean_scene']
    meta=next(x for x in c.read(c.OUT/'datasets/val/index.json') if x['scene_id']==sid)
    cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config')
    sample,_=r.oldg.baseline.c.sampled(cfg,meta['seed'],meta['index']);sample=replace(sample,defects=[])
    engine=r.Engine(meta['seed'],meta['amplitude']);norm=c.read(c.OUT/'normalization.json');model=learn.model_for(frozen['cnn'])
    minimum=selection['features']['beam.dwell_time_us']['training_min'];records=[]
    for label,dwell in [('original',sample.beam.dwell_time_us),('training_minimum',minimum)]:
        changed=replace(sample,beam=replace(sample.beam,dwell_time_us=dwell));arrays,_=r.render(engine,changed)
        if label=='original':
            with np.load(meta['npz_path']) as a:assert np.array_equal(a['raw'],arrays['raw']),'Original validation renderer replay differed'
        data=dict(meta=[meta],mask=arrays['mask'][r.SCORE][None]);image=learn.normalize(arrays['raw'][0],norm)
        row=dict(condition=label,dwell_time_us=dwell)
        for method,score in [('cnn',learn.predict_image(image,model)),('median',learn.median_image(image,frozen['median']['spec']))]:
            value=metrics.evaluate(data,score[None],frozen[method]['selection']['best'])[0]
            row[method]=value
        row['noise_rms']=float(np.sqrt(np.mean((arrays['raw'][0]-arrays['quiet'])**2)))
        records.append(row);np.savez_compressed(c.OUT/f'dwell_{label}.npz',**arrays)
        print('Dwell diagnostic',label,dwell,'CNN false regions',row['cnn']['fp'],flush=True)
    result=dict(source_scene=sid,original_pixels_replayed_exactly=True,records=records,
        control='Same background, detector, geometry, seed, normalization, model and operating point; only dwell_time_us changed',
        limits='One validation-background diagnostic within an assumed raster acquisition model. Dwell changes shot/readout scaling and frame/drift timing together. It is not real SEM calibration or a new confirmation score.',
        selection_effect='None; no scene is replaced, no threshold or trained model changed')
    c.write(path,result);return result
