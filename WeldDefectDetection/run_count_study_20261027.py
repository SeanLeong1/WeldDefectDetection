"""Versioned count-noise study. Use the repository .venv Python."""
from pathlib import Path
import argparse
import sys
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'WeldDefectDetection'),str(ROOT/'DataSimulation')]
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from count_study import campaign as c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','check','background','generate','develop','freeze','evaluate','report','audit','all'])
    p.add_argument('--split',choices=['train','val','test'],default='train')
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--arm',choices=list(c.ARMS))
    p.add_argument('--campaign')
    a=p.parse_args()
    if a.action=='background' and a.campaign!=c.OUT.name:
        raise RuntimeError('Background worker requires an explicit current campaign identity; retired controllers cannot dispatch new workers.')
    if a.action=='prepare':c.prepare()
    elif a.action=='check':c.noise_check()
    elif a.action=='background':c.background(a.split,a.index)
    elif a.action=='generate':c.generate(a.split,a.workers)
    elif a.action=='audit':c.audit()
    else:
        from count_study import learning as learn, reporting
        if a.action=='develop':learn.develop(a.arm)
        elif a.action=='freeze':learn.freeze()
        elif a.action=='evaluate':learn.evaluate(a.arm)
        elif a.action=='report':reporting.report()
        else:
            c.prepare();c.noise_check();learn.check_metrics()
            c.generate('train',a.workers);c.generate('val',a.workers)
            learn.develop();learn.freeze();c.generate('test',a.workers)
            learn.evaluate();c.audit();reporting.report()


if __name__=='__main__':main()
