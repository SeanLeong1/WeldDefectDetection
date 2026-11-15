"""Precompute the fixed 2x target-only CNN fits from audited source batches."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
import torch
from box_development.campaign import DEFAULT,DEV,SEEDS,check,require
from box_development.data import load,concatenate
from box_development.training import fit
from box_comparison.core import subset,write

if __name__=='__main__':
    root=DEFAULT;check(root);require(root,'audit');require(root,'train_1');torch.set_num_threads(4)
    data=load(DEV,'original');val=subset(data,'val')
    train=concatenate([subset(data,'train'),load(root/'datasets/train_1','addition_1',True)])
    reports=[fit(train,val,root/'training','cnn',seed,2,'cnn',epochs=20) for seed in SEEDS]
    write(root/'precomputation_2x_cnn.json',dict(status='ok',scope='Original train plus first audited expansion batch; original validation; no final selection.',
        checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in reports}))
