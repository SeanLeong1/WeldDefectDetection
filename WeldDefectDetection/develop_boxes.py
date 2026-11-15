"""Execute the five-step TOPO detector development and confirmation campaign."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from box_development.campaign import DEFAULT,STAGES,run
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all',*STAGES,'train_1','train_2','train_3'],default='all')
    args=parser.parse_args();run(args.root,args.stage)
