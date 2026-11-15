"""Matched training arms with clean, training-only hard-negative mining."""
from pathlib import Path
import gc, random, time
import numpy as np
import torch
import torch.nn.functional as F
from raised_crown import learning as base, rendering as r
from circular_study import metrics
from src.algorithms.classical.cnn_detector import TinyDefectSegNet
from . import campaign as c


def load(split,norm=None):
    norm=norm or c.read(c.OUT/'normalization.json')
    rows=c.read(c.OUT/f'datasets/{split}/index.json'); images=[]; masks=[]
    for row in rows:
        with np.load(row['npz_path']) as a:
            images.append(base.normalize(a['raw'][0],norm)); masks.append(a['mask'])
    full=np.stack(masks)
    return dict(x=np.stack(images)[:,None],fullmask=full,mask=full[:,r.SCORE],meta=rows)


def prepare():
    c.verify(); folder=c.OUT/'training'; folder.mkdir(exist_ok=True)
    assert not (c.OUT/'datasets/test/index.json').exists(), 'Development must precede confirmation'
    paths=[Path(__file__),Path(base.__file__),Path(metrics.__file__),Path(base.historical_median.__file__),
        c.ROOT/'WeldDefectDetection/src/algorithms/classical/cnn_detector.py']
    protocol=dict(source_hashes={str(p):c.sha(p) for p in paths},
        data={str(c.OUT/f'datasets/{s}/index.json'):c.sha(c.OUT/f'datasets/{s}/index.json') for s in ('train','val')},
        arms=c.ARMS,seeds=c.SEEDS,epochs=20,patches_per_epoch=1200,
        selection='Final epoch 20, unchanged validation-only threshold grid and metrics')
    # JSON round-trip makes tuple/list comparison stable during resume.
    import json
    protocol=json.loads(json.dumps(protocol))
    path=c.OUT/'cv_protocol.json'
    if path.exists(): assert c.read(path)==protocol
    else: c.write(path,protocol)
    normfile=c.OUT/'normalization.json'
    if not normfile.exists():
        values=[]
        for row in c.read(c.OUT/'datasets/train/index.json'):
            with np.load(row['npz_path']) as a: values.append(a['raw'][0,r.SCORE,::8].ravel())
        lo,hi=np.quantile(np.concatenate(values),[.01,.99]); pad=.08*(hi-lo)
        c.write(normfile,dict(low=max(0.,float(lo-pad)),high=float(hi+pad),source='train only',tone='sqrt'))


def clean_candidates(data,seed):
    rng=np.random.default_rng(seed+991); candidates=[]
    for ring,meta in enumerate(data['meta']):
        if meta['source_class']!='clean': continue
        assert meta['split']=='train' and not data['fullmask'][ring].any()
        coords=np.array([(ring,y,x) for y in base.CENTRES for x in range(16,5024,32)])
        candidates.extend(coords[rng.choice(len(coords),168,replace=False)])
    return np.asarray(candidates)


def mine(data,model,candidates):
    model.eval(); values=[]
    with torch.inference_mode():
        for j in range(0,len(candidates),32):
            x,_=base.batch(data,candidates[j:j+32])
            p=torch.sigmoid(model(torch.from_numpy(x))).flatten(1)
            values.extend(p.topk(64,dim=1).values.mean(1).numpy().tolist())
    values=np.asarray(values); selected=[]
    for ring in np.unique(candidates[:,0]):
        indices=np.flatnonzero(candidates[:,0]==ring)
        selected.extend(indices[np.argsort(values[indices],kind='stable')[-16:]])
    selected=np.asarray(selected); hard=candidates[selected]
    assert all(data['meta'][int(i)]['split']=='train' and data['meta'][int(i)]['source_class']=='clean' for i in hard[:,0])
    return hard,dict(candidate_count=len(candidates),retained=len(hard),coordinates=hard.tolist(),
        scores=values[selected].tolist(),selection='16 highest-scoring clean tiles per training background')


def epoch_indices(positive,negative,hard,seed,epoch):
    # Separate streams preserve identical positive samples across matched arms.
    pos_rng=np.random.default_rng(seed+10000*epoch)
    neg_rng=np.random.default_rng(seed+10000*epoch+1)
    pos=positive[pos_rng.integers(len(positive),size=600)]
    neg=negative[neg_rng.integers(len(negative),size=600)]
    if hard is not None:
        hrng=np.random.default_rng(seed+10000*epoch+2)
        neg[300:]=hard[hrng.integers(len(hard),size=300)]
    ids=np.concatenate([pos,neg]); np.random.default_rng(seed+10000*epoch+3).shuffle(ids)
    return ids


def train_arm(arm):
    assert arm in c.ARMS; torch.set_num_threads(4); c.verify()
    assert not (c.OUT/'datasets/test/index.json').exists()
    protocol=c.read(c.OUT/'cv_protocol.json')
    for key in ('source_hashes','data'):
        for path,digest in protocol[key].items(): assert c.sha(path)==digest
    data=load('train'); val=load('val'); positive,negative=base.populations(data)
    px=np.mean([base.tile(data['fullmask'][a],b,d).mean() for a,b,d in positive])*.5
    weight=float(min(20,np.sqrt((1-px)/max(px,1e-8))))
    folder=c.OUT/'training'/arm; folder.mkdir(exist_ok=True); specs=[]
    for seed in c.SEEDS:
        report=folder/f'cnn_{seed}.json'; checkpoint=folder/f'cnn_{seed}.pt'
        if report.exists():
            spec=c.read(report); assert c.sha(checkpoint)==spec['checkpoint_sha256']; specs.append(spec); continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
        model=TinyDefectSegNet(in_channels=1,base_channels=24)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20,eta_min=5e-5)
        history=[]; mining=[]; hard=None; start=0; started=time.monotonic()
        candidates=clean_candidates(data,seed) if arm=='hard_negative' else None
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(state['state_dict']); optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']); torch.set_rng_state(state['torch_rng'])
            history=state['history']; mining=state['mining']; hard=state['hard']; start=state['epoch']
        for epoch in range(start,20):
            if arm=='hard_negative' and epoch in (5,10,15):
                hard,record=mine(data,model,candidates); mining.append(dict(after_epoch=epoch,**record))
            ids=epoch_indices(positive,negative,hard,seed,epoch); model.train(); losses=[]
            for j in range(0,len(ids),32):
                x,y=base.batch(data,ids[j:j+32]); target=torch.from_numpy(y)
                optimizer.zero_grad(set_to_none=True); logits=model(torch.from_numpy(x)); prob=torch.sigmoid(logits)
                loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(weight))+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); losses.append(loss.item())
            scheduler.step(); history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
            temp=checkpoint.with_suffix('.tmp')
            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                torch_rng=torch.get_rng_state(),history=history,mining=mining,hard=hard,epoch=epoch+1),temp)
            temp.replace(checkpoint); print(arm,seed,epoch+1,round(np.mean(losses),4),flush=True)
        model.eval(); scores=base.predict(val,model)
        selection=metrics.select(val,scores,[.2,.35,.5,.65,.8,.9,.97,.995])
        spec=dict(seed=seed,arm=arm,checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),selection=selection,
            history=history,mining=mining,parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started)
        c.write(report,spec); specs.append(spec); print('Validation',arm,seed,selection['best']['metrics'],flush=True)
        del model,scores; gc.collect()
    c.write(folder/'selection.json',dict(cnn=max(specs,key=lambda x:metrics.key(x['selection']['best']['metrics'])),cnn_candidates=specs))


def freeze():
    if (c.OUT/'frozen_models.json').exists(): return verify_freeze()
    torch.set_num_threads(4); assert not (c.OUT/'datasets/test/index.json').exists()
    data=load('train'); val=load('val'); spec=base.median_fit(data)
    scores=np.stack([base.median_image(x[0],spec) for x in val['x']])
    median=dict(spec=spec,selection=metrics.select(val,scores,[2,3,4,6,8,12,20,40,80]))
    arms={arm:c.read(c.OUT/f'training/{arm}/selection.json') for arm in c.ARMS}
    # Default is chosen prospectively on validation, not after confirmation.
    selected=max(c.ARMS,key=lambda arm:metrics.key(arms[arm]['cnn']['selection']['best']['metrics']))
    paths=[c.OUT/'protocol.json',c.OUT/'cv_protocol.json',c.OUT/'normalization.json']
    paths+=list((c.OUT/'training').rglob('*.json'))+list((c.OUT/'training').rglob('*.pt'))
    paths += [Path(x) for x in c.read(c.OUT/'cv_protocol.json')['source_hashes']]
    for split in ('train','val'):
        path=c.OUT/f'datasets/{split}/index.json'; paths.append(path)
        paths += [Path(x['npz_path']) for x in c.read(path)]
    evaluation=Path(__file__).with_name('assessment.py'); paths.append(evaluation)
    frozen=dict(arms=arms,selected_arm=selected,cnn=arms[selected]['cnn'],median=median,
        files={str(p):c.sha(p) for p in paths},frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        input_shape=[160,5024],physical_validation=False)
    c.write(c.OUT/'frozen_models.json',frozen); print('FROZEN',selected,flush=True); return frozen


def verify_freeze():
    p=c.read(c.OUT/'frozen_models.json')
    for path,digest in p['files'].items():
        resolved=c.ROOT/path if p.get('portable') else path
        assert c.sha(resolved)==digest,f'Frozen CV artifact changed: {path}'
    if p.get('portable'):
        for arm in p['arms'].values():
            for spec in [arm['cnn'],*arm['cnn_candidates']]: spec['checkpoint']=str(c.ROOT/spec['checkpoint'])
        p['cnn']['checkpoint']=str(c.ROOT/p['cnn']['checkpoint'])
    return p
