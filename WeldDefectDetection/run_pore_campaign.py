"""Run the versioned surface-pore comparison and independent confirmation."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from pore_campaign.campaign import DEFAULT,STAGES,run
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all',*STAGES],default='all')
    args=parser.parse_args();run(args.root,args.stage)
