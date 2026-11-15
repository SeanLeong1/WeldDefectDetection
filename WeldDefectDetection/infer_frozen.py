"""Run a frozen static or spatial model on raw TOPO input without labels."""
from pathlib import Path
import argparse
import numpy as np
from src.algorithms.frozen_inference import infer_frozen

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['static','spatial'])
    for name in ('input','checkpoint','normalization','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Inference output already exists; use a new path.')
    result=infer_frozen(args.kind,args.input,args.checkpoint,args.normalization)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**result)
    print(args.output.resolve())

if __name__=='__main__':
    main()
