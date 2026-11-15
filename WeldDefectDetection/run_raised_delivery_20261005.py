"""Ordered, resumable delivery with disjoint-background generation workers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
if __name__=='__main__':
    from raised_crown import campaign as c
    from raised_crown.parallel_generation import run
    from raised_crown.learning import train
    from raised_crown.execution import evaluate
    from raised_crown.reporting import report
    c.prepare();c.pilot()
    run('train',4);run('val',4)
    train()
    run('test',4);run('stress',4)
    evaluate();c.audit();report()
    print('ORDERED RESEARCH DELIVERY COMPLETE',flush=True)
