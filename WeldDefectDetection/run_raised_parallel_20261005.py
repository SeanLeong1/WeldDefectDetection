from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['train','val','test','stress']);parser.add_argument('--worker',type=int);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    from raised_crown.parallel_generation import worker,run
    if args.worker is None:run(args.split,args.workers)
    else:worker(args.split,args.worker,args.workers)
