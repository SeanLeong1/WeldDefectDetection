"""Supplementary training-only memorization diagnostic, balanced across pore types."""
from pathlib import Path
import sys,time
import numpy as np
import torch
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from pore_campaign.campaign import DEFAULT,check
from box_refinement.campaign import require
from box_development.data import load
from box_comparison.core import subset,write,sha
from box_development.methods import make_network,predict
from src.training.train_cnn import WeightedBCEDiceLoss
if __name__=='__main__':
    check(DEFAULT);torch.set_num_threads(2)
    plan=DEFAULT/'balanced_diagnostic_plan.json'
    if plan.exists():raise RuntimeError('Preserve existing diagnostic attempt.')
    write(plan,dict(selection='First four positive training windows of each pore type and first eight clean training windows. No validation/confirmation inputs.',
                    steps=300,seed=921820002,role='Supplementary diagnostic only; never a candidate or input to threshold/model selection.',source=str(Path(__file__).resolve()),source_sha256=sha(__file__)))
    while not (DEFAULT/'receipts/development.json').exists():time.sleep(2)
    require(DEFAULT,'development')
    data=subset(load(DEFAULT/'datasets/development','development'),'train')
    ids=[]
    for kind in ('keyhole_pore','intrinsic_pore'):
        group=[i for i,m in enumerate(data['meta']) if m['source_class']==kind and data['mask'][i].any()][:4]
        if len(group)!=4:raise RuntimeError('Insufficient training pores for diagnostic.')
        ids.extend(group)
    ids.extend([i for i in range(len(data['meta'])) if not data['mask'][i].any()][:8])
    torch.manual_seed(921820002);model=make_network(1)
    x=torch.from_numpy(data['target'][ids,None]);y=torch.from_numpy(data['mask'][ids,None].astype(np.float32))
    lossfn=WeightedBCEDiceLoss(10.,dice_weight=.5);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    history=[]
    for step in range(300):
        model.train();opt.zero_grad();loss=lossfn(model(x),y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite diagnostic loss.')
        loss.backward();opt.step()
        if step in (0,49,99,199,299):history.append(dict(step=step+1,loss=float(loss.detach())))
    pred=predict(model,x.numpy())[:,0]>=.5;truth=y.numpy()[:,0]>0
    scores={}
    for kind,indices in [('keyhole_pore',range(4)),('intrinsic_pore',range(4,8)),('clean',range(8,16)),('overall',range(16))]:
        pp,tt=pred[list(indices)],truth[list(indices)]
        tp=int((pp&tt).sum());fp=int((pp&~tt).sum());fn=int((~pp&tt).sum())
        scores[kind]=dict(pixel_tp=tp,pixel_fp=fp,pixel_fn=fn,pixel_f1=2*tp/max(2*tp+fp+fn,1))
    write(DEFAULT/'balanced_overfit_diagnostic.json',dict(window_ids=[data['meta'][i]['window_id'] for i in ids],history=history,metrics=scores,
        limitation='Memorization on sixteen training targets, not generalization or physical validity. No model settings changed.'))
    print(scores)
