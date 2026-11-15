"""Precompute only the preregistered 1x fits while independent new data render.

No final campaign method is selected/frozen here. The main stage verifies the
completed fit records after its full expansion prerequisite passes.
"""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
import torch
from box_development.campaign import DEFAULT,DEV,SEEDS,check,require
from box_development.data import load
from box_development.training import fit
from box_comparison.core import subset,write

if __name__=='__main__':
    root=DEFAULT;check(root);require(root,'audit');torch.set_num_threads(4)
    data=load(DEV,'original');train,val=subset(data,'train'),subset(data,'val')
    reports=[]
    for arm in ('cnn','cnn_median'):
        for seed in SEEDS:
            reports.append(fit(train,val,root/'training',arm,seed,1,arm,epochs=20))
    write(root/'precomputation_1x.json',dict(status='ok',scope='Original audited train/validation only; no final model selection.',
        checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in reports}))
