"""Label-free inference using a frozen third-campaign TOPO detector."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from box_development.campaign import DEFAULT,ARMS
from box_development.inference import infer
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--approach',choices=ARMS,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();infer(args.root,args.input,args.output,args.approach)
