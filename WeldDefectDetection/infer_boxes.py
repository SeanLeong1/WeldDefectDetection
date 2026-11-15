import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'WeldDefectDetection'),str(REPO/'DataSimulation')]
from box_comparison.campaign import ARMS, DEFAULT
from box_comparison.inference import infer
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Label-free bounding boxes from raw TOPO crops.')
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--approach',choices=ARMS,required=True)
    a=p.parse_args();infer(a.root,a.input,a.output,a.approach)
