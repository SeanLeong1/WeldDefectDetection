"""Run the four-arm TOPO box campaign without modifying historical artifacts."""
import argparse
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/'DataSimulation'))
sys.path.insert(0,str(REPO/'WeldDefectDetection'))

from box_comparison.campaign import DEFAULT, STAGES, run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all',*STAGES],default='all')
    args=parser.parse_args()
    run(args.root,args.stage)
