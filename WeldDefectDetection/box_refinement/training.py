import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from src.algorithms.classical.cnn_detector import TinyDefectSegNet
from src.training.train_cnn import WeightedBCEDiceLoss
from box_comparison.training import inputs,predict_network,scores_for as previous_scores
from box_comparison.core import sha,write,selection_key
from .methods import residual_score,select,augment_pair


def fit_simple(val,folder,previous):
    selected={}
    for arm in ('thresholding','median'):
        # Retain the previous frozen configuration as a genuine fallback.
        best={**previous[arm],'implementation':'previous'}
        trials=[]
        for sigma in (1.,2.):
            for standardized in (False,True):
                feature=dict(sigma=sigma,standardized=standardized)
                scores=np.stack([residual_score(t,r,arm,**feature) for t,r in zip(val['target'],val['reference'])])
                thresholds=(1.5,2.5,4.,6.) if standardized else (.005,.01,.02,.04,.08)
                chosen,records=select(val,scores,thresholds)
                for record in records:record['feature']=feature
                trials.extend(records)
                if selection_key(chosen['metrics'])>selection_key(best['metrics']):
                    best=dict(arm=arm,implementation='refined',**chosen)
        write(Path(folder)/f'{arm}_selection.json',dict(selected=best,trials=trials))
        selected[arm]=best
        print(f'{arm}: validation box F1={best["metrics"]["box_f1"]:.4f}; {best["implementation"]}',flush=True)
    return selected


def fit_network(train,val,folder,arm,seed,epochs=40):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report=folder/f'{arm}_{seed}.json';checkpoint=folder/f'{arm}_{seed}.pt'
    if report.exists():
        result=json.loads(report.read_text())
        if sha(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Completed checkpoint changed.')
        return result
    if checkpoint.exists():raise ValueError('Partial training is preserved; use a fresh run directory after inspection.')
    torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True)
    gen=torch.Generator().manual_seed(seed+1);augmentation=torch.Generator().manual_seed(seed+2)
    arrays=[torch.from_numpy(train[k][:,None].astype(np.float32)) for k in ('target','reference','mask')]
    positive=train['mask'].reshape(len(train['mask']),-1).any(axis=1)
    weights=np.where(positive,.5/max(positive.sum(),1),.5/max((~positive).sum(),1))
    sampler=WeightedRandomSampler(torch.from_numpy(weights),num_samples=len(positive),replacement=True,generator=gen)
    loader=DataLoader(TensorDataset(*arrays),batch_size=32,sampler=sampler,num_workers=0)
    model=TinyDefectSegNet(in_channels=3 if arm=='cnn_median' else 1)
    # Estimate class weighting for the expected 50/50 sampled population.
    fraction=.5*float(train['mask'][positive].mean()) if positive.any() else 0.
    positive_weight=min(20.,float(np.sqrt((1-fraction)/max(fraction,1e-8))))
    loss_fn=WeightedBCEDiceLoss(positive_weight,dice_weight=.5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00005)
    xval=inputs(val,arm=='cnn_median');history=[];best=None;started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for target,reference,mask in loader:
            target,reference,mask=augment_pair(target,reference,mask,augmentation)
            x=torch.cat([target,reference,torch.abs(target-reference)],1) if arm=='cnn_median' else target
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(x),mask)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        entry=dict(epoch=epoch,loss=float(np.mean(losses)),learning_rate=scheduler.get_last_lr()[0])
        if epoch%5==0:
            chosen,_=select(val,predict_network(model,xval),(.2,.4,.6,.8))
            entry.update(chosen)
            if best is None or selection_key(chosen['metrics'])>selection_key(best['metrics']):
                best=dict(epoch=epoch,**chosen)
                torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=seed,epoch=epoch,in_channels=3 if arm=='cnn_median' else 1),checkpoint)
            print(f'{arm} seed={seed} epoch={epoch}/{epochs} validation box F1={chosen["metrics"]["box_f1"]:.4f}',flush=True)
        history.append(entry)
    result=dict(arm=arm,seed=seed,selected=best,history=history,positive_weight=positive_weight,
        positive_training_targets=int(positive.sum()),training_targets=len(positive),
        elapsed_seconds=time.monotonic()-started,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint))
    write(report,result);return result


def scores_for(data,spec):
    if spec['implementation']=='previous':return previous_scores(data,spec)
    if spec['arm'] in ('thresholding','median'):
        return np.stack([residual_score(t,r,spec['arm'],**spec['feature']) for t,r in zip(data['target'],data['reference'])])
    return previous_scores(data,spec)
