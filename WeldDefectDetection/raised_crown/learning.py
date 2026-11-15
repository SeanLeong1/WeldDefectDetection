"""Same 24-channel CNN, wider physical coverage, train-only calibration."""
import random,time,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from circular_study import metrics,median as historical_median
from src.algorithms.classical.cnn_detector import TinyDefectSegNet
from . import campaign as c,rendering as r

CENTRES=(32,64,96,128)

def normalize(raw,norm):
    return np.sqrt(np.clip((raw-norm['low'])/(norm['high']-norm['low']),0,1)).astype('f4')

def load(split,norm=None):
    norm=norm or c.read(c.OUT/'normalization.json');rows=c.read(c.OUT/f'datasets/{split}/index.json')
    images=[];masks=[]
    for row in rows:
        with np.load(row['npz_path']) as a:images.append(normalize(a['raw'][0],norm));masks.append(a['mask'])
    full=np.stack(masks)
    return dict(x=np.stack(images)[:,None],fullmask=full,mask=full[:,r.SCORE],meta=rows)

def tile(image,y,x):
    return image[...,y-32:y+32,(np.arange(64)+x-32)%image.shape[-1]]

def populations(data):
    positive=[];negative=[]
    for ring,mask in enumerate(data['fullmask']):
        for y in CENTRES:
            for x in range(16,mask.shape[1],32):
                (positive if tile(mask,y,x).any() else negative).append((ring,y,x))
    return np.array(positive),np.array(negative)

def batch(data,indices):
    return np.stack([tile(data['x'][ring],y,x) for ring,y,x in indices]),np.stack([tile(data['fullmask'][ring],y,x)[None] for ring,y,x in indices]).astype('f4')

def model_for(spec):
    model=TinyDefectSegNet(in_channels=1,base_channels=24)
    assert c.sha(spec['checkpoint'])==spec['checkpoint_sha256']
    model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);model.eval()
    return model

def predict_image(image,model):
    if image.shape!=(160,5024):raise ValueError('Requires raised-crown 160x5024 native annulus')
    stripes=[]
    with torch.inference_mode():
        for y in CENTRES:
            centres=list(range(16,5024,32));chunks=[]
            for j in range(0,len(centres),32):
                x=np.stack([tile(image,y,k)[None] for k in centres[j:j+32]])
                prob=torch.sigmoid(model(torch.from_numpy(x)))[:,0].numpy()
                chunks.extend(prob[:,16:48,16:48])
            stripes.append(np.concatenate(chunks,axis=1))
    return np.concatenate(stripes,axis=0)

def predict(data,model):
    return np.stack([predict_image(x[0],model) for x in data['x']])

def median_fit(data):
    records=[]
    for y in CENTRES:
        vals=[]
        for image,meta in zip(data['x'][:,0],data['meta']):
            if meta['source_class']!='clean':continue
            assert meta['split']=='train'
            for centre in range(0,5024,256):
                value,_,_=historical_median.residual(image[y-32:y+32],centre,'neighbours8');vals.append(value)
        vals=np.stack(vals);mu=np.median(vals,axis=0)
        scale=np.maximum(.003,1.4826*np.median(abs(vals-mu),axis=0))
        records.append(dict(centre=mu.tolist(),scale=scale.tolist(),windows=len(vals)))
    return dict(variant='neighbours8_radial_tiles',calibration=records,fit_split='train')

def median_image(image,spec,legacy=False):
    stripes=[]
    for i,y in enumerate(CENTRES):
        calibration=spec['calibration'][0 if legacy else i]
        mu=np.array(calibration['centre'],'f4');scale=np.array(calibration['scale'],'f4')
        out=[]
        for start in range(0,5024,32):
            residual,_,_=historical_median.residual(image[y-32:y+32],start+16,'neighbours8')
            out.append((abs(residual-mu)/scale)[16:48,16:48])
        stripes.append(np.concatenate(out,axis=1))
    return np.concatenate(stripes,axis=0)

def verify_freeze():
    p=c.read(c.OUT/'frozen_models.json')
    for path,h in p['files'].items():
        resolved=c.ROOT/path if p.get('portable') else path
        assert c.sha(resolved)==h,f'Changed frozen CV artifact: {resolved}'
    if p.get('portable'):
        for spec in [p['cnn'],*p['cnn_candidates']]:spec['checkpoint']=str(c.ROOT/spec['checkpoint'])
    return p

def train():
    torch.set_num_threads(4);c.verify_sources()
    if (c.OUT/'frozen_models.json').exists():return verify_freeze()
    assert not (c.OUT/'datasets/test/index.json').exists()
    folder=c.OUT/'training';folder.mkdir(exist_ok=True)
    normfile=c.OUT/'normalization.json'
    if not normfile.exists():
        values=[]
        for row in c.read(c.OUT/'datasets/train/index.json'):
            with np.load(row['npz_path']) as a:values.append(a['raw'][0,r.SCORE,::8].ravel())
        lo,hi=np.quantile(np.concatenate(values),[.01,.99]);pad=.08*(hi-lo)
        c.write(normfile,dict(low=max(0.,float(lo-pad)),high=float(hi+pad),source='train only',tone='sqrt'))
    protocol=dict(source_hashes={str(Path(__file__)):c.sha(__file__),str(Path(metrics.__file__)):c.sha(metrics.__file__),
        str(Path(historical_median.__file__)):c.sha(historical_median.__file__),
        str(Path(__file__).with_name('assessment.py')):c.sha(Path(__file__).with_name('assessment.py'))},
        data={str(c.OUT/f'datasets/{s}/index.json'):c.sha(c.OUT/f'datasets/{s}/index.json') for s in ('train','val')},
        seeds=c.SEEDS,epochs=20,examples_per_epoch=1200,architecture='unchanged TinyDefectSegNet base24',
        learning_rate=.001,weight_decay=.0001,sampler='50 percent positive tiles, 50 percent all negative tiles',
        selection='Final epoch 20, validation box F1; threshold/min_area/closing and seed selection before test',
        comparison_limit='Different surface geometry, defect law, B=0 and uniform composition versus v4; joint domain shift, not a geometry-only effect')
    pp=c.OUT/'cv_protocol.json'
    if pp.exists():assert c.read(pp)==protocol
    else:c.write(pp,protocol)
    data=load('train');val=load('val')
    medpath=folder/'median.json'
    if medpath.exists():med=c.read(medpath)
    else:
        spec=median_fit(data);scores=np.stack([median_image(x[0],spec) for x in val['x']])
        med=dict(spec=spec,selection=metrics.select(val,scores,[2,3,4,6,8,12,20,40,80]))
        c.write(medpath,med);print('Median validation',med['selection']['best']['metrics'],flush=True)
    positive,negative=populations(data);assert len(positive) and len(negative)
    px=np.mean([tile(data['fullmask'][a],b,d).mean() for a,b,d in positive])*.5
    weight=float(min(20,np.sqrt((1-px)/max(px,1e-8))))
    specs=[]
    for seed in c.SEEDS:
        report=folder/f'cnn_{seed}.json';checkpoint=folder/f'cnn_{seed}.pt'
        if report.exists():
            spec=c.read(report);assert c.sha(checkpoint)==spec['checkpoint_sha256'];specs.append(spec);continue
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(seed);model=TinyDefectSegNet(in_channels=1,base_channels=24)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20,eta_min=5e-5)
        history=[];start=0;started=time.monotonic()
        if checkpoint.exists():
            s=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(s['state_dict'])
            optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
            rng.bit_generator.state=s['rng'];torch.set_rng_state(s['torch_rng']);history=s['history'];start=s['epoch']
        for epoch in range(start,20):
            ids=np.concatenate([positive[rng.integers(len(positive),size=600)],negative[rng.integers(len(negative),size=600)]])
            rng.shuffle(ids);model.train();losses=[]
            for j in range(0,len(ids),32):
                x,y=batch(data,ids[j:j+32]);target=torch.from_numpy(y)
                optimizer.zero_grad(set_to_none=True);logits=model(torch.from_numpy(x));prob=torch.sigmoid(logits)
                loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(weight))+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(loss.item())
            scheduler.step();history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
            tmp=checkpoint.with_suffix('.tmp');torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history,epoch=epoch+1),tmp);tmp.replace(checkpoint)
            print('CNN',seed,'epoch',epoch+1,'loss',round(np.mean(losses),4),flush=True)
        model.eval();scores=predict(val,model);selection=metrics.select(val,scores,[.2,.35,.5,.65,.8,.9,.97,.995])
        spec=dict(seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),selection=selection,
            history=history,parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started)
        c.write(report,spec);specs.append(spec);print('CNN validation',seed,selection['best']['metrics'],flush=True)
        del model,scores;gc.collect()
    chosen=max(specs,key=lambda x:metrics.key(x['selection']['best']['metrics']))
    paths=[normfile,pp,c.OUT/'protocol.json',*folder.glob('*')]
    paths.extend(Path(p) for p in protocol['source_hashes'])
    for split in ('train','val'):
        paths.append(c.OUT/f'datasets/{split}/index.json')
        paths.extend(Path(x['npz_path']) for x in c.read(paths[-1]))
    c.write(c.OUT/'frozen_models.json',dict(cnn=chosen,cnn_candidates=specs,median=med,
        files={str(p):c.sha(p) for p in paths},frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        physical_validation=False,geometry_version='raised-crown-v1',input_shape=[160,5024]))
    print('FROZEN before confirmation generation',flush=True)
