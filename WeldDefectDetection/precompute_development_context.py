"""Compute preregistered independent 4x context fits after expansion audit.

This optional worker performs no model selection or confirmation evaluation.
The main stage reuses hash-verified completed fit records. If the main process
reaches an unfinished checkpoint, its conservative partial-run guard stops it;
resume the main command after this worker completes rather than deleting files.
"""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
import torch
from box_development.campaign import DEFAULT,DEV,SEEDS,check,require
from box_development.data import load,training_sets
from box_development.training import fit
from box_development.methods import fit_scale
from box_comparison.core import subset,write

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['cnn_median','context_scaled'],required=True)
    args=parser.parse_args();root=DEFAULT;check(root);require(root,'audit');require(root,'expand')
    torch.set_num_threads(4)
    train=training_sets(root,DEV)[4];val=subset(load(DEV,'original'),'val')
    scale=fit_scale(train,'median')[0].tolist() if args.variant=='context_scaled' else None
    reports=[fit(train,val,root/'training','cnn_median',seed,4,args.variant,epochs=20,scale=scale) for seed in SEEDS]
    write(root/f'precomputation_{args.variant}.json',dict(status='ok',scope='Audited 4x training and original validation only; no final selection.',
        checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in reports}))
