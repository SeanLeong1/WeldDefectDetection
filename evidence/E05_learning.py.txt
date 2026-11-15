"""Grouped development, bounded selection and fresh confirmation."""
from collections import OrderedDict
from pathlib import Path
import gc
import itertools
import random
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from circular_study import metrics
from raised_crown import learning as base
from raised_acquisition import learning as historical
from src.algorithms.classical.cnn_detector import TinyDefectSegNet
from . import campaign as c

cv2.setNumThreads(1)
torch.set_num_threads(4)
OFFSETS=(-256,-192,-128,-64,64,128,192,256)


def selected_rows(split,arm=None):
    rows=c.read(c.OUT/f'datasets/{split}/index.json')
    if arm:
        count,levels=c.ARMS[arm]
        rows=[r for r in rows if r['count_multiplier'] in levels and (split!='train' or r['index']<count)]
    return rows


class Data:
    """Raw arrays stay on disk; geometry masks share storage across acquisitions."""
    def __init__(self,rows,norm):
        self.rows=rows;self.norm=norm;self.cache=OrderedDict();geometry={}
        self.masks=[];self.truth=[]
        for row in rows:
            path=row['geometry_path']
            if path not in geometry:
                with np.load(path) as a: m=a['mask'].astype('u1')
                geometry[path]=(m,metrics.regions(m[16:144])[1])
            m,t=geometry[path];self.masks.append(m);self.truth.append(t)

    def raw(self,i):
        if i in self.cache:self.cache.move_to_end(i);return self.cache[i]
        x=np.load(self.rows[i]['raw_path'],mmap_mode='r');self.cache[i]=x
        if len(self.cache)>16:self.cache.popitem(last=False)
        return x

    def image(self,i,sigma=0.):
        raw=self.raw(i)
        if sigma:raw=gaussian_filter(raw,sigma,mode=('reflect','wrap'))
        return base.normalize(raw,self.norm)

    def batch(self,ids):
        x=np.stack([base.normalize(base.tile(self.raw(int(i)),int(y),int(a)),self.norm)[None] for i,y,a in ids])
        y=np.stack([base.tile(self.masks[int(i)],int(r),int(a))[None] for i,r,a in ids]).astype('f4')
        return x,y

    def populations(self):
        cache={};positive=[];negative=[]
        for i,row in enumerate(self.rows):
            path=row['geometry_path']
            if path not in cache:
                pos=[];neg=[];m=self.masks[i]
                for y in base.CENTRES:
                    for x in range(16,5024,32):
                        (pos if base.tile(m,y,x).any() else neg).append((y,x))
                cache[path]=(pos,neg)
            pos,neg=cache[path]
            positive.extend((i,y,x) for y,x in pos);negative.extend((i,y,x) for y,x in neg)
        return np.asarray(positive),np.asarray(negative)


def prepare():
    c.verify()
    if (c.OUT/'frozen_models.json').exists():
        verify_freeze()
        return
    assert not (c.OUT/'datasets/test/index.json').exists()
    folder=c.OUT/'development';folder.mkdir(exist_ok=True)
    sources=[Path(__file__),Path(metrics.__file__),Path(base.__file__),Path(historical.__file__),
             c.ROOT/'WeldDefectDetection/src/algorithms/classical/cnn_detector.py']
    protocol=dict(source_hashes={str(p):c.sha(p) for p in sources},
        indexes={s:c.sha(c.OUT/f'datasets/{s}/index.json') for s in ('train','val')},
        source_protocol_sha256=c.sha(c.OUT/'protocol.json'))
    target=c.OUT/'development_protocol.json'
    if target.exists():assert c.read(target)==protocol,'Development code/data changed'
    else:c.write(target,protocol)
    path=c.OUT/'normalization.json'
    if not path.exists():
        values=[]
        # Same transform in all arms; all its samples belong to every nested
        # training geometry set. Normalization exposure to all levels is declared.
        for row in selected_rows('train'):
            if row['index']<24:
                values.append(np.load(row['raw_path'],mmap_mode='r')[16:144:8,::8].ravel())
        lo,hi=np.quantile(np.concatenate(values),[.01,.99]);pad=.08*(hi-lo)
        c.write(path,dict(low=max(0.,float(lo-pad)),high=float(hi+pad),tone='sqrt',
            source='24 nested training backgrounds, all count levels; no validation/test pixels'))


def verify_development():
    p=c.read(c.OUT/'development_protocol.json')
    for path,digest in p['source_hashes'].items():assert c.sha(path)==digest,path
    for s,digest in p['indexes'].items():assert c.sha(c.OUT/f'datasets/{s}/index.json')==digest
    assert c.sha(c.OUT/'protocol.json')==p['source_protocol_sha256']


def component_rows(score,truth,boxes,row,threshold,closing,areas):
    """Reuse labels across area cutoffs; exact historical circular metric."""
    mask=(score>=threshold).astype('u1')
    if closing>1:
        mask=cv2.morphologyEx(np.pad(mask,((0,0),(closing,closing)),mode='wrap'),
            cv2.MORPH_CLOSE,np.ones((closing,closing),'u1'))[:,closing:-closing]
    out=[]
    # Seam components require the existing periodic-union algorithm.
    seam=bool(mask[:,0].any() and mask[:,-1].any())
    if not seam:
        n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
        overlap=np.bincount(lab[truth>0],minlength=n)
    for area in areas:
        if seam:
            kept,pred=metrics.regions(mask,area)
            inter=int(((kept>0)&(truth>0)).sum());pixels=int(kept.sum())
        else:
            ids=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=area);ids=ids[ids!=0]
            pred=[[int(stats[k,0]),int(stats[k,1]),int(stats[k,0]+stats[k,2]),int(stats[k,1]+stats[k,3])] for k in ids]
            inter=int(overlap[ids].sum());pixels=int(stats[ids,cv2.CC_STAT_AREA].sum())
        truth_pixels=int(truth.sum())
        out.append(dict(scene_id=row['scene_id'],background_id=row['background_id'],source_class=row['source_class'],
            count_multiplier=row['count_multiplier'],diameter_mm=row['diameter_mm'],repeat=row['repeat'],
            dwell_bin=row['dwell_bin'],amplitude=row['amplitude'],**metrics.match(pred,boxes,score.shape[1]),
            intersection=inter,union=pixels+truth_pixels-inter,predicted_pixels=pixels,
            truth_pixels=truth_pixels,predicted_regions=len(pred),predicted_boxes=pred,truth_boxes=boxes))
    return out


def objective(rows):
    levels=sorted({r['count_multiplier'] for r in rows})
    by={str(m):metrics.summary([r for r in rows if r['count_multiplier']==m]) for m in levels}
    pooled=metrics.summary(rows)
    return dict(macro_f1=float(np.mean([x['f1'] for x in by.values()])),metrics=pooled,by_count=by)


def selection_key(trial):return trial['macro_f1'],trial['metrics']['pixel_iou'],-trial['metrics']['fp']


def select(data,scores,thresholds):
    assert all(r['split']=='val' for r in data.rows)
    trials=[]
    for threshold in thresholds:
        for closing in c.CLOSINGS:
            rows={a:[] for a in c.AREAS}
            for i,meta in enumerate(data.rows):
                choices=component_rows(scores[i],data.masks[i][16:144],data.truth[i],meta,threshold,closing,c.AREAS)
                for area,value in zip(c.AREAS,choices):rows[area].append(value)
            for area in c.AREAS:
                trials.append(dict(threshold=float(threshold),closing=closing,min_area=area,**objective(rows[area])))
    return dict(best=max(trials,key=selection_key),trials=trials)


def evaluate_scores(data,scores,op):
    return [component_rows(scores[i],data.masks[i][16:144],data.truth[i],row,
                           op['threshold'],op['closing'],[op['min_area']])[0] for i,row in enumerate(data.rows)]


def median_residual(image):
    return image-np.median(np.stack([np.roll(image,-k,axis=1) for k in OFFSETS]),axis=0)


def median_fit(data,sigma):
    samples=[median_residual(data.image(i,sigma))[:,::8] for i,r in enumerate(data.rows) if r['source_class']=='clean']
    a=np.concatenate(samples,axis=1);centre=np.median(a,axis=1,keepdims=True)
    scale=np.maximum(.003,1.4826*np.median(abs(a-centre),axis=1,keepdims=True))
    return dict(sigma=sigma,centre=centre.tolist(),scale=scale.tolist(),calibration_split='train')


def median_score(image,spec):
    return (abs(median_residual(image)-np.asarray(spec['centre'],'f4'))/np.asarray(spec['scale'],'f4'))[16:144]


def score_file(path,data,predict):
    done=path.with_suffix('.done.json')
    identity=dict(scene_ids=[r['scene_id'] for r in data.rows],shape=[len(data.rows),128,5024])
    if done.exists():
        assert c.read(done)==identity
        return np.load(path,mmap_mode='r')
    path.parent.mkdir(parents=True,exist_ok=True)
    arr=np.lib.format.open_memmap(path,mode='w+',dtype='f4',shape=tuple(identity['shape']))
    for i in range(len(data.rows)):
        score=predict(i);assert score.shape==(128,5024) and np.isfinite(score).all()
        arr[i]=score
        if (i+1)%48==0:print('Scored',path.stem,i+1,'/',len(data.rows),flush=True)
    arr.flush();del arr;c.write(done,identity)
    return np.load(path,mmap_mode='r')


def tune_median(arm,train,val):
    folder=c.OUT/'development'/arm;folder.mkdir(exist_ok=True)
    target=folder/'median.json'
    if target.exists():return c.read(target)
    candidates=[]
    for sigma in c.SIGMAS:
        report=folder/f'median_sigma{sigma:g}.json'
        if report.exists():candidates.append(c.read(report));continue
        start=time.monotonic();spec=median_fit(train,sigma)
        scores=score_file(folder/f'median_sigma{sigma:g}_val.npy',val,lambda i:median_score(val.image(i,sigma),spec))
        selection=select(val,scores,c.MED_THRESHOLDS)
        record=dict(spec=spec,selection=selection,seconds=time.monotonic()-start)
        c.write(report,record);candidates.append(record)
        print('Median selected',arm,sigma,selection['best']['macro_f1'],flush=True)
        del scores;gc.collect()
    chosen=max(candidates,key=lambda x:selection_key(x['selection']['best']))
    c.write(target,dict(**chosen,arm=arm,sigmas_tested=list(c.SIGMAS)))
    return c.read(target)


def clean_candidates(data,seed):
    rng=np.random.default_rng(seed+991);out=[]
    groups={r['background_id'] for r in data.rows if r['source_class']=='clean'}
    coords=[(y,x) for y in base.CENTRES for x in range(16,5024,32)]
    for group in sorted(groups):
        rings=[i for i,r in enumerate(data.rows) if r['background_id']==group and r['source_class']=='clean']
        pool=np.asarray([(i,y,x) for i in rings for y,x in coords])
        out.extend(pool[rng.choice(len(pool),168,replace=False)])
    return np.asarray(out)


def mine(data,model,candidates):
    model.eval();values=[]
    with torch.inference_mode():
        for j in range(0,len(candidates),32):
            x,_=data.batch(candidates[j:j+32])
            p=torch.sigmoid(model(torch.from_numpy(x))).flatten(1)
            values.extend(p.topk(64,dim=1).values.mean(1).numpy().tolist())
    values=np.asarray(values);groups=np.asarray([data.rows[int(i)]['background_id'] for i in candidates[:,0]])
    chosen=[]
    for group in sorted(set(groups)):
        ids=np.flatnonzero(groups==group);chosen.extend(ids[np.argsort(values[ids],kind='stable')[-16:]])
    chosen=np.asarray(chosen)
    return candidates[chosen],dict(candidate_count=len(candidates),retained=len(chosen),coordinates=candidates[chosen].tolist())


def train_cnn(arm,seed,train,val):
    folder=c.OUT/'development'/arm;folder.mkdir(exist_ok=True)
    target=folder/f'cnn_{seed}.json'
    if target.exists():
        result=c.read(target);assert c.sha(result['checkpoint'])==result['checkpoint_sha256'];return result
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    model=TinyDefectSegNet(in_channels=1,base_channels=24)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20,eta_min=5e-5)
    positive,negative=train.populations()
    px=np.mean([base.tile(train.masks[int(i)],int(y),int(x)).mean() for i,y,x in positive])*.5
    weight=float(min(20,np.sqrt((1-px)/max(px,1e-8))))
    candidates=clean_candidates(train,seed);hard=None;history=[];mining=[];start=0;train_seconds=0.
    resume=folder/f'cnn_{seed}_resume.pt'
    if resume.exists():
        s=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(s['state_dict']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        torch.set_rng_state(s['torch_rng']);hard=s['hard'];history=s['history'];mining=s['mining'];start=s['epoch'];train_seconds=s['train_seconds']
    for epoch in range(start,20):
        t=time.monotonic()
        if epoch in (5,10,15):
            hard,record=mine(train,model,candidates);mining.append(dict(after_epoch=epoch,**record))
        ids=historical.epoch_indices(positive,negative,hard,seed,epoch)
        model.train();losses=[]
        for j in range(0,len(ids),32):
            x,y=train.batch(ids[j:j+32]);target_y=torch.from_numpy(y)
            optimizer.zero_grad(set_to_none=True);logits=model(torch.from_numpy(x));prob=torch.sigmoid(logits)
            loss=F.binary_cross_entropy_with_logits(logits,target_y,pos_weight=torch.tensor(weight))+1-(2*(prob*target_y).sum()+1)/(prob.sum()+target_y.sum()+1)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(loss.item())
        scheduler.step();train_seconds+=time.monotonic()-t
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        state=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            torch_rng=torch.get_rng_state(),hard=hard,history=history,mining=mining,epoch=epoch+1,train_seconds=train_seconds)
        temporary=resume.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(resume)
        if epoch+1 in (10,20):torch.save(dict(state_dict=model.state_dict(),epoch=epoch+1),folder/f'cnn_{seed}_e{epoch+1}.pt')
        print('Epoch',arm,seed,epoch+1,round(history[-1]['loss'],4),flush=True)
    selections=[]
    for epoch in (10,20):
        record=folder/f'cnn_{seed}_e{epoch}.json';checkpoint=folder/f'cnn_{seed}_e{epoch}.pt'
        if record.exists():selections.append(c.read(record));continue
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);model.eval()
        started=time.monotonic()
        scores=score_file(folder/f'cnn_{seed}_e{epoch}_val.npy',val,lambda i:base.predict_image(val.image(i),model))
        selection=select(val,scores,c.CNN_THRESHOLDS)
        spec=dict(arm=arm,seed=seed,epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),
            selection=selection,train_seconds=train_seconds,validation_seconds=time.monotonic()-started,
            parameters=sum(p.numel() for p in model.parameters()),history=history,mining=mining)
        c.write(record,spec);selections.append(spec);del scores;gc.collect()
    best=max(selections,key=lambda x:selection_key(x['selection']['best']))
    c.write(target,best)
    return best


def develop(arm=None):
    prepare();verify_development()
    for name in ([arm] if arm else c.ARMS):
        final=c.OUT/'development'/name/'selection.json'
        if final.exists():continue
        norm=c.read(c.OUT/'normalization.json')
        train=Data(selected_rows('train',name),norm);val=Data(selected_rows('val',name),norm)
        median=tune_median(name,train,val)
        candidates=[train_cnn(name,s,train,val) for s in c.SEEDS]
        chosen=max(candidates,key=lambda x:selection_key(x['selection']['best']))
        c.write(final,dict(arm=name,median=median,cnn=chosen,cnn_candidates=candidates))
        del train,val;gc.collect()


def freeze():
    if (c.OUT/'frozen_models.json').exists():return verify_freeze()
    verify_development();assert not (c.OUT/'datasets/test/index.json').exists()
    arms={a:c.read(c.OUT/'development'/a/'selection.json') for a in c.ARMS}
    paths=[c.OUT/'protocol.json',c.OUT/'development_protocol.json',c.OUT/'normalization.json']
    paths+=list((c.OUT/'development').rglob('*.json'))+list((c.OUT/'development').rglob('*.pt'))
    paths += [Path(p) for p in c.read(c.OUT/'development_protocol.json')['source_hashes']]
    for split in ('train','val'):
        path=c.OUT/f'datasets/{split}/index.json';paths.append(path)
        for row in c.read(path):paths.extend([Path(row['raw_path']),Path(row['geometry_path'])])
    frozen=dict(arms=arms,files={str(p):c.sha(p) for p in set(paths)},
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),physical_validation=False)
    c.write(c.OUT/'frozen_models.json',frozen);return frozen


def verify_freeze(full=False):
    p=c.read(c.OUT/'frozen_models.json')
    # Workers verify model/config/source identities. The final audit verifies
    # every large scene once, avoiding rereading all training pixels per worker.
    for path,digest in p['files'].items():
        if full or Path(path).suffix not in ('.npy','.npz'):
            assert c.sha(path)==digest,path
    return p


def result_summary(rows):
    return dict(**objective(rows),intervals=metrics.bootstrap(rows),
        by_count_intervals={str(m):metrics.bootstrap([r for r in rows if r['count_multiplier']==m]) for m in c.MULTIPLIERS},
        by_size={label:metrics.summary([r for r in rows if r['source_class']=='clean' or lo<=r['diameter_mm']<hi])
                 for label,lo,hi in [('small',.15,.30),('medium',.30,.60),('large',.60,1.01)]},
        rows=rows)


def evaluate(only_arm=None):
    frozen=verify_freeze();norm=c.read(c.OUT/'normalization.json')
    data=Data(selected_rows('test'),norm);folder=c.OUT/'confirmation';folder.mkdir(exist_ok=True)
    for arm,selection in frozen['arms'].items():
        if only_arm and arm!=only_arm:continue
        for seed_spec in selection['cnn_candidates']:
            name=f'{arm}_cnn_{seed_spec["seed"]}';target=folder/f'{name}.json'
            if target.exists():continue
            model=base.model_for(seed_spec)
            t=time.monotonic()
            scores=score_file(folder/f'{name}.npy',data,lambda i:base.predict_image(data.image(i),model))
            prediction_seconds=time.monotonic()-t
            rows=evaluate_scores(data,scores,seed_spec['selection']['best'])
            c.write(target,dict(name=name,arm=arm,kind='cnn',seed=seed_spec['seed'],
                selected=seed_spec['seed']==selection['cnn']['seed'],prediction_seconds=prediction_seconds,
                operating_point=seed_spec['selection']['best'],**result_summary(rows)))
            del model,scores;gc.collect()
        name=f'{arm}_median';target=folder/f'{name}.json'
        if not target.exists():
            spec=selection['median']['spec'];t=time.monotonic()
            scores=score_file(folder/f'{name}.npy',data,lambda i:median_score(data.image(i,spec['sigma']),spec))
            seconds=time.monotonic()-t;rows=evaluate_scores(data,scores,selection['median']['selection']['best'])
            c.write(target,dict(name=name,arm=arm,kind='median',prediction_seconds=seconds,
                operating_point=selection['median']['selection']['best'],spec=spec,**result_summary(rows)))
            del scores;gc.collect()
    if only_arm:return
    # Evaluate old models unchanged on exactly the same new confirmation scenes.
    old=c.read(c.old.OUT/'frozen_models.json');oldnorm=c.read(c.old.OUT/'normalization.json')
    olddata=Data(data.rows,oldnorm)
    for kind in ('cnn','median'):
        target=folder/f'retained_{kind}.json'
        if target.exists():continue
        spec=old[kind];model=base.model_for(spec) if kind=='cnn' else None;t=time.monotonic()
        scores=score_file(folder/f'retained_{kind}.npy',olddata,
            lambda i:base.predict_image(olddata.image(i),model) if kind=='cnn' else base.median_image(olddata.image(i),spec['spec']))
        seconds=time.monotonic()-t;rows=evaluate_scores(olddata,scores,spec['selection']['best'])
        c.write(target,dict(name=f'retained_{kind}',kind=kind,arm='retained',prediction_seconds=seconds,**result_summary(rows)))
    c.write(c.OUT/'results.json',{p.stem:c.read(p) for p in folder.glob('*.json') if not p.name.endswith('.done.json')})


def check_metrics():
    rng=np.random.default_rng(1027)
    for seam in (False,True):
        score=np.zeros((128,5024),'f4');score[40:48,20:30]=.8
        if seam:score[60:66,:4]=.8;score[60:66,-4:]=.8
        score[80,100]=.9;truth=(score>.7).astype('u1');boxes=metrics.regions(truth)[1]
        row=dict(scene_id='check',background_id='check',source_class='open_intrinsic',count_multiplier=1,
                 diameter_mm=.5,repeat=0,dwell_bin=0,amplitude=0.)
        for close in c.CLOSINGS:
            got=component_rows(score,truth,boxes,row,.5,close,c.AREAS)
            for area,g in zip(c.AREAS,got):
                m,pred=metrics.binary(score,.5,area,close);expected=metrics.match(pred,boxes,5024)
                assert all(g[k]==v for k,v in expected.items())
                assert g['predicted_pixels']==int(m.sum())
                assert g['intersection']==int(((m>0)&(truth>0)).sum())
    image=rng.random((160,5024),dtype=np.float32)
    assert np.allclose(np.roll(median_residual(image),31,1),median_residual(np.roll(image,31,1)))
    changed=image.copy();changed[60,0]+=100
    before=image[60,0]-median_residual(image)[60,0]
    after=changed[60,0]-median_residual(changed)[60,0]
    assert abs(before-after)<1e-5
    c.write(c.OUT/'method_checks.json',dict(status='passed',circular_metric_parity=True,angular_wrap=True,target_excluded=True))
