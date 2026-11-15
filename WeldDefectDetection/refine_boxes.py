"""Bounded second development and fresh confirmation of TOPO box methods."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from box_refinement.campaign import DEFAULT,STAGES,run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=['all',*STAGES],default='all');a=p.parse_args();run(a.root,a.stage)
