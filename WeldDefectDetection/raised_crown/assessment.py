"""Frozen confirmation, native-coordinate localisation errors and instance recall."""
from dataclasses import replace
from pathlib import Path
import gc,time
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from circular_study import metrics,training as oldtrain
from . import campaign as c,learning as learn,rendering as r

ERRORS=[dict(name='nominal',dx=0.,dy=0.,dr=0.)]
for axis in ('dx','dy','dr'):
    for sign in (-1,1):
        value=dict(dx=0.,dy=0.,dr=0.);value[axis]=sign*.5
        ERRORS.append(dict(name=f'{axis}_{sign*.5:+.2f}',**value))
ERRORS.append(dict(name='combined',dx=.25,dy=-.25,dr=.25))

def result(data,scores,choice):
    rows=metrics.evaluate(data,scores,choice,retain=True)
    hidden=[x for x in rows if x['source_class']=='hidden']
    return dict(metrics=metrics.summary(rows),intervals=metrics.bootstrap(rows),rows=rows,
        by_class={kind:metrics.summary([x for x in rows if x['source_class']==kind]) for kind in sorted({x['source_class'] for x in rows})},
        hidden_negative_rings=len(hidden),hidden_rings_flagged=sum(x['predicted_regions']>0 for x in hidden))

def map_back(score,error):
    theta=2*np.pi*np.arange(5024)/5024;radius=40+(np.arange(128)-63.5)*.05
    x=radius[:,None]*np.cos(theta)-error['dx'];y=radius[:,None]*np.sin(theta)-error['dy']
    rr=(np.hypot(x,y)-40-error['dr'])/.05+63.5
    rr=np.where((rr<0)&(rr>-1e-9),0,rr)
    rr=np.where((rr>127)&(rr<127+1e-9),127,rr)
    aa=(np.arctan2(y,x)%(2*np.pi))*5024/(2*np.pi)
    padded=np.pad(score,((0,0),(32,32)),mode='wrap')
    return map_coordinates(padded,[rr,aa+32],order=1,mode='constant',cval=0).astype('f4')

def estimated_error(native):
    # Fit centre and radius by maximising correlation of the median radial
    # signature across angular sectors. Only image pixels enter this estimator.
    # An image-only coarse circle grid tests a bounded prior around the nominal ring.
    best=None
    for dx in (-.5,0.,.5):
        for dy in (-.5,0.,.5):
            theta=2*np.pi*np.arange(0,5024,32)/5024
            radii=40+(np.arange(128)-63.5)*.05
            x=dx+radii[:,None]*np.cos(theta);y=dy+radii[:,None]*np.sin(theta)
            origin=(native.shape[0]-1)*.025
            band=map_coordinates(native,[(y+origin)/.05,(x+origin)/.05],order=1,mode='constant',cval=np.nan)
            signature=np.median(band,axis=1)
            residual=float(np.median(abs(band-signature[:,None])))
            if best is None or residual<best[0]:best=(residual,dx,dy,signature)
    # Symmetric profile centre from its bright radial support, no ground truth.
    signature=best[3];weights=np.maximum(signature-np.quantile(signature,.2),0)
    dr=float(np.clip(np.sum((np.arange(128)-63.5)*.05*weights)/max(weights.sum(),1e-9),-.5,.5))
    return dict(name='image_estimated',dx=best[1],dy=best[2],dr=dr,estimator='bounded image-only centre grid and intensity centroid radius')

def stress(frozen,model):
    folder=c.OUT/'stress_predictions';folder.mkdir(exist_ok=True)
    rows=c.read(c.OUT/'datasets/stress/index.json');norm=c.read(c.OUT/'normalization.json');records=[]
    for meta in rows:
        receipt=folder/(meta['scene_id']+'.json')
        if receipt.exists():records.extend(c.read(receipt));continue
        with np.load(meta['npz_path']) as a:
            native=a['native'];mask=a['mask'][r.SCORE];instances=a['instances'][:,r.SCORE]
            nominal=a['raw'][0]
        data=dict(mask=mask[None],meta=[meta]);scene=[]
        # All layouts receive nominal testing; localisation sweep applies to
        # clean and close-pair scenes, with each condition sharing native pixels.
        errors=(ERRORS if meta['source_class'] in ('clean','pair_near') else ERRORS[:1])+[estimated_error(native)]
        for error in errors:
            image=nominal if error['name']=='nominal' else r.extract(native,error=error)
            x=learn.normalize(image,norm)
            scores=dict(cnn=learn.predict_image(x,model),median=learn.median_image(x,frozen['median']['spec']))
            for method,score in scores.items():
                mapped=score if error['name']=='nominal' else map_back(score,error)
                choice=frozen[method]['selection']['best']
                row=metrics.evaluate(data,mapped[None],choice,retain=True)[0]
                truth=[box for m in instances for box in metrics.regions(m)[1]]
                individual=metrics.match(row['predicted_boxes'],truth,5024)
                scene.append(dict(method=method,error=error,instance_metrics=individual,instance_count=len(truth),**row))
        c.write(receipt,scene);records.extend(scene);print('stress',meta['scene_id'],flush=True)
    groups={}
    for method in ('cnn','median'):
        groups[method]={}
        for error in sorted({x['error']['name'] for x in records}):
            subset=[x for x in records if x['method']==method and x['error']['name']==error]
            tp=sum(x['instance_metrics']['tp'] for x in subset);n=sum(x['instance_count'] for x in subset)
            groups[method][error]=dict(metrics=metrics.summary(subset),intervals=metrics.bootstrap(subset),
                individual_recall=tp/max(n,1),individual_tp=tp,individual_count=n,
                by_class={k:metrics.summary([x for x in subset if x['source_class']==k]) for k in sorted({x['source_class'] for x in subset})})
    c.write(c.OUT/'robustness_results.json',dict(groups=groups,rows=records,
        note='Nominal and image-estimated conditions include all layouts; injected-error sweeps include clean and close pairs only. Compare matched populations.',
        mapping='Native Cartesian extraction; scores back to fixed true band; missing coverage scored zero',physical_validation=False))

def paired_diagnostic(frozen,model):
    folder=c.OUT/'paired_profile';folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():return
    cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config');norm=c.read(c.OUT/'normalization.json');rows=[]
    for index in range(3):
        seed=2350500000+index;sample,_=r.oldg.baseline.c.sampled(cfg,seed,index)
        for profile in ('raised','recessed'):
            engine=r.Engine(seed,(0,.05,.1)[index],profile)
            for kind in ('clean','open_keyhole','bulge'):
                arrays,metadata=r.render(engine,replace(sample,defects=c.plans(seed,index)[kind]))
                sid=f'paired-{index}-{profile}-{kind}';np.savez_compressed(folder/(sid+'.npz'),**arrays)
                meta=dict(scene_id=sid,background_id=f'paired-{index}',source_class=kind,amplitude=engine.amplitude)
                data=dict(mask=arrays['mask'][r.SCORE][None],meta=[meta]);image=learn.normalize(arrays['raw'][0],norm)
                for method,score in [('cnn',learn.predict_image(image,model)),('median',learn.median_image(image,frozen['median']['spec']))]:
                    row=metrics.evaluate(data,score[None],frozen[method]['selection']['best'])[0]
                    rows.append(dict(profile=profile,method=method,**row))
                c.write(folder/(sid+'.json'),metadata)
                print(sid,flush=True)
            del engine;gc.collect()
    c.write(folder/'results.json',dict(rows=rows,backgrounds=3,
        claim='Paired rendering diagnostic only, not adequately powered transfer confirmation',
        held_constant='B=0, uniform alloy, acquisition, pore law, sizes, positions and seeds'))

def evaluate():
    torch.set_num_threads(4);frozen=learn.verify_freeze();c.verify_sources()
    folder=c.OUT/'predictions';folder.mkdir(exist_ok=True);data=learn.load('test');results={}
    for spec in frozen['cnn_candidates']:
        name=f'cnn_{spec["seed"]}';path=folder/(name+'.json')
        if path.exists():results[name]=c.read(path);continue
        model=learn.model_for(spec);started=time.monotonic();scores=learn.predict(data,model)
        results[name]=result(data,scores,spec['selection']['best']);results[name]['inference_seconds']=time.monotonic()-started
        c.write(path,results[name]);print(name,results[name]['metrics'],flush=True)
    path=folder/'median.json'
    if path.exists():results['median']=c.read(path)
    else:
        scores=np.stack([learn.median_image(x[0],frozen['median']['spec']) for x in data['x']])
        results['median']=result(data,scores,frozen['median']['selection']['best']);c.write(path,results['median'])
    del data,scores;gc.collect()
    # Explicit radial tiling adapter preserves old weights/normalisation/thresholds.
    old=c.read(r.oldg.OUT/'frozen_models.json');oldnorm=c.read(r.oldg.OUT/'normalization.json')
    data=learn.load('test',oldnorm)
    for method in ('cnn','median'):
        name='frozen_v4_'+method;path=folder/(name+'.json')
        if path.exists():results[name]=c.read(path);continue
        if method=='cnn':
            model=oldtrain.load_model(old['cnn']);scores=learn.predict(data,model)
        else:scores=np.stack([learn.median_image(x[0],old['median']['spec'],legacy=True) for x in data['x']])
        results[name]=result(data,scores,old[method]['selection']['best'])
        results[name]['adapter']='Four radial 64x64 windows, central32 retained, no rescaling or refitting; not the historical central64 coverage'
        c.write(path,results[name]);print(name,results[name]['metrics'],flush=True)
    del data,scores;gc.collect()
    c.write(c.OUT/'results.json',dict(results=results,selected=f'cnn_{frozen["cnn"]["seed"]}',
        interpretation='Joint-distribution transfer vs retraining under the new synthetic renderer, not physical validation'))
    model=learn.model_for(frozen['cnn']);stress(frozen,model);paired_diagnostic(frozen,model)
