"""Controlled CPU CNN screening and seed replication; no test access."""
import gc,random,time,zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src.algorithms.classical.cnn_detector import TinyDefectSegNet
from box_comparison.core import sha
from . import generation as g,metrics,median

VARIANTS=['fixed_negatives','rotating_negatives','hard_negatives','augmentation','larger_context','focal_loss','capacity24','topo_tx_ty']


def configuration(variant):
    if variant not in VARIANTS:raise ValueError(variant)
    return dict(variant=variant,input_size=96 if variant=='larger_context' else 64,
        channels=3 if variant=='topo_tx_ty' else 1,base_channels=24 if variant=='capacity24' else 16,
        loss='weighted_focal_plus_dice' if variant=='focal_loss' else 'weighted_bce_plus_dice',
        sampler='fixed' if variant=='fixed_negatives' else 'rotating',epochs=20,examples_per_epoch=1200)


def extract(data,indices,size,channels,augment=None):
    xs=[];ys=[]
    for ring,centre in indices:
        if augment is not None:centre+=int(augment.integers(-16,17))
        image=data['x'][ring,:channels]
        if size==64:image=image[:,16:80]
        x=median.tile(image,centre,size).copy()
        y=median.tile(data['mask'][ring],centre,64)[None].astype('f4')
        if augment is not None:
            if augment.random()<.5:x=x[...,::-1].copy();y=y[...,::-1].copy()
            x=np.clip(x*augment.uniform(.90,1.10)+augment.uniform(-.02,.02),0,1)
        xs.append(x);ys.append(y)
    return np.stack(xs).astype('f4'),np.stack(ys)


def network(config):
    return TinyDefectSegNet(in_channels=config['channels'],base_channels=config['base_channels'])


def predict(data,model,config):
    model.eval();size=config['input_size'];channels=config['channels'];out=[]
    with torch.inference_mode():
        for ring in range(len(data['meta'])):
            centres=list(range(16,data['x'].shape[-1],32));chunks=[]
            for j in range(0,len(centres),32):
                x,_=extract(data,[(ring,c) for c in centres[j:j+32]],size,channels)
                score=torch.sigmoid(model(torch.from_numpy(x)))[:,0].numpy()
                if size==96:score=score[:,16:80,16:80]
                chunks.extend(score[:,:,16:48])
            out.append(np.concatenate(chunks,axis=1))
    return np.stack(out)


def populations(data):
    positive=[];negative=[];fixed=[];rng=np.random.default_rng(100484001)
    for ring,mask in enumerate(data['mask']):
        neg=[]
        for centre in range(16,mask.shape[1],32):
            pair=(ring,centre)
            if median.tile(mask,centre).any():positive.append(pair)
            else:negative.append(pair);neg.append(pair)
        fixed.extend([neg[i] for i in rng.choice(len(neg),8,replace=False)])
    return np.array(positive),np.array(negative),np.array(fixed)


def mine(data,negative,model,config,rng):
    ids=rng.choice(len(negative),min(720,len(negative)),replace=False);candidates=negative[ids];scores=[]
    model.eval()
    with torch.inference_mode():
        for j in range(0,len(candidates),32):
            x,_=extract(data,candidates[j:j+32],config['input_size'],config['channels'])
            s=torch.sigmoid(model(torch.from_numpy(x)))[:,0]
            scores.extend(s.flatten(1).topk(16,dim=1).values.mean(1).numpy())
    return candidates[np.argsort(scores)[-max(1,len(candidates)//3):]]


def fit(train,val,variant,seed):
    folder=g.OUT/'training';folder.mkdir(exist_ok=True)
    path=folder/f'cnn_{variant}_{seed}.json';checkpoint=folder/f'cnn_{variant}_{seed}.pt'
    if path.exists():
        result=g.read(path);assert sha(checkpoint)==result['checkpoint_sha256'];return result
    config=configuration(variant);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True);rng=np.random.default_rng(seed)
    model=network(config);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20,eta_min=5e-5)
    positive,negative,fixed=populations(train)
    px=np.mean([median.tile(train['mask'][r],c).mean() for r,c in positive])*.5
    weight=float(min(20,np.sqrt((1-px)/max(px,1e-8))))
    history=[];hard=None;start_epoch=0;started=time.monotonic()
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);history=state['history'];start_epoch=state['epoch'];hard=state['hard']
    for epoch in range(start_epoch,20):
        if variant=='hard_negatives' and epoch in (5,10,15):hard=mine(train,negative,model,config,rng)
        neg=fixed if variant=='fixed_negatives' else negative
        selected=np.concatenate([positive[rng.integers(len(positive),size=600)],neg[rng.integers(len(neg),size=600)]])
        if hard is not None:selected[600:900]=hard[rng.integers(len(hard),size=300)]
        rng.shuffle(selected);model.train();losses=[]
        for j in range(0,len(selected),32):
            x,y=extract(train,selected[j:j+32],config['input_size'],config['channels'],rng if variant=='augmentation' else None)
            target=torch.from_numpy(y);optimizer.zero_grad(set_to_none=True)
            logits=model(torch.from_numpy(x))
            if config['input_size']==96:logits=logits[:,:,16:80,16:80]
            prob=torch.sigmoid(logits)
            bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(weight),reduction='none')
            if variant=='focal_loss':bce=bce*(1-(prob*target+(1-prob)*(1-target)))**2
            dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
            loss=bce.mean()+dice;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(loss.item())
        scheduler.step();history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        temporary=checkpoint.with_suffix('.tmp')
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history,epoch=epoch+1,hard=hard,config=config),temporary)
        temporary.replace(checkpoint)
        print(f'cnn {variant} seed={seed} epoch={epoch+1} loss={np.mean(losses):.4f}',flush=True)
    score=predict(val,model,config)
    choice=metrics.select(val,score,[.2,.35,.5,.65,.8,.9,.97,.995])
    result=dict(method='cnn',variant=variant,seed=seed,config=config,checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha(checkpoint),selection=choice,history=history,seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()),positive_windows=len(positive),negative_windows=len(negative),
        fixed_negative_windows=len(fixed),positive_weight=weight,checkpoint_rule='Final epoch 20, no epoch selection')
    g.write(path,result);print('selected',variant,seed,choice['best']['metrics'],flush=True)
    del model,score;gc.collect();return result


def load_model(spec):
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    model=network(spec['config']);state=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict']);model.eval();return model


def run():
    g.baseline.c.verify_sources(g.OUT)
    assert not (g.OUT/'datasets/test/index.json').exists(),'Test already opened: use a new version for training changes'
    if not (g.OUT/'normalization.json').exists():g.normalize()
    sources=list(Path(__file__).parent.glob('*.py'))
    hashes={str(p.resolve()):sha(p) for p in sources if p.name in ('training.py','metrics.py','median.py','generation.py','geometry.py','evaluation.py','diagnostics.py','quiet.py')}
    input_files={}
    for split in ('train','val'):
        index=g.OUT/f'datasets/{split}/index.json';input_files[str(index.resolve())]=sha(index)
        for row in g.read(index):
            assert sha(row['npz_path'])==row['npz_sha256'],row['scene_id']
            input_files[row['npz_path']]=row['npz_sha256']
    freeze=g.OUT/'cv_protocol_v2.json'
    protocol=dict(schema='cv-primary-topo/v2',source_hashes=hashes,input_files=input_files,selection='Final epoch 20; seed1 TOPO-only variant screening; baseline and TOPO-only winner replicated; auxiliary TOPO+Tx+Ty reported separately; all validation choices frozen before test',
        configs=[configuration(v) for v in VARIANTS],inference='Stride32, central32 retained from 64-output tiles; periodic full-ring components and circular box IoU >= .5')
    if freeze.exists():assert g.read(freeze)==protocol,'CV sources changed after screening began'
    else:g.write(freeze,protocol)
    train=g.load('train');val=g.load('val')
    medians=median.run(train,val)
    cnns=[]
    for v in VARIANTS[:-1]:cnns.append(fit(train,val,v,g.SEEDS[0]))
    del train,val;gc.collect()
    train=g.load('train',3);val=g.load('val',3)
    cnns.append(fit(train,val,'topo_tx_ty',g.SEEDS[0]))
    winner=max([c for c in cnns if c['config']['channels']==1],key=lambda c:metrics.key(c['selection']['best']['metrics']))['variant']
    # Baseline and validation-selected variant receive two additional independent seeds.
    if winner!='topo_tx_ty':
        train['x']=train['x'][:,:1].copy();val['x']=val['x'][:,:1].copy();gc.collect()
    for v in dict.fromkeys(['fixed_negatives',winner]):
        for seed in g.SEEDS[1:]:cnns.append(fit(train,val,v,seed))
    chosen_median=max(medians,key=lambda c:metrics.key(c['selection']['best']['metrics']))
    chosen_cnn=max([c for c in cnns if c['variant']==winner],key=lambda c:metrics.key(c['selection']['best']['metrics']))
    payload=dict(cnn=chosen_cnn,median=chosen_median,cnn_variant=winner,cnn_candidates=cnns,median_candidates=medians,
        auxiliary_cnn=next(c for c in cnns if c['config']['channels']==3),
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files={})
    paths=[g.OUT/'normalization.json',g.OUT/'protocol.json',freeze,*list((g.OUT/'training').glob('*'))]
    payload['files']={str(p.resolve()):sha(p) for p in paths if p.is_file()}
    payload['files'].update(hashes)
    payload['files'].update(input_files)
    g.write(g.OUT/'frozen_models.json',payload)
    with zipfile.ZipFile(g.OUT/'cv_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,str(p.relative_to(g.ROOT)))
    print('FROZEN: CNN '+winner+', median '+chosen_median['variant'],flush=True)
