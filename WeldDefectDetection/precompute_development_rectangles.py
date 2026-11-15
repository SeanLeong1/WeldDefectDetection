"""Run only the preregistered rectangle experiment once its evidence gate passes.

One completed 4x mask-only fit with three extent errors is sufficient to prove
the existential gate. Later seeds cannot revoke that condition. Final campaign
selection still waits for all planned fits and audits in develop_boxes.py.
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
from box_development.audit import diagnose
from box_comparison.core import subset,write
from box_comparison.campaign import read

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['cnn','cnn_median'],required=True)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args();root=DEFAULT;check(root);require(root,'audit');require(root,'expand');torch.set_num_threads(4)
    val=subset(load(DEV,'original'),'val');counts={}
    for seed in SEEDS:
        path=root/'training'/f'{args.arm}_{args.arm}_4x_{seed}.json'
        if path.exists():
            spec=read(path)
            report=diagnose(val,{args.arm:spec},root/f'precompute_extent_{args.arm}_{seed}.json')
            counts[str(seed)]=report[args.arm]['counts']['inaccurate_extent']
    gate=dict(arm=args.arm,counts=counts,passed=bool(counts and max(counts.values())>=3))
    write(root/f'precompute_rectangle_gate_{args.arm}.json',gate);print(gate,flush=True)
    if gate['passed'] and not args.probe:
        train=training_sets(root,DEV)[4]
        reports=[fit(train,val,root/'training',args.arm,seed,4,args.arm,epochs=20,rectangle=True) for seed in SEEDS]
        write(root/f'precomputation_rectangle_{args.arm}.json',dict(status='ok',gate=gate,
            checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in reports}))
