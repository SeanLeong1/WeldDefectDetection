"""Resume the prospectively specified circular acquisition study."""
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]


def main():
    from raised_acquisition import campaign as c, learning as learn
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','background','development','train','arm','confirmation','evaluate','audit','report','all'])
    p.add_argument('--split',choices=c.COUNTS); p.add_argument('--index',type=int); p.add_argument('--arm',choices=c.ARMS)
    args=p.parse_args(); stage=args.stage
    if stage in ('prepare','all'): c.prepare()
    if stage=='background': c.generate_background(args.split,args.index)
    if stage in ('development','all'): c.generate('train'); c.generate('val')
    if stage=='arm': learn.train_arm(args.arm)
    if stage in ('train','all'):
        if (c.OUT/'frozen_models.json').exists(): learn.verify_freeze()
        else:
            from raised_acquisition.preflight import run as preflight
            if not (c.OUT/'development_preflight.json').exists(): preflight()
            learn.prepare(); processes=[]; handles=[]
            for arm in c.ARMS:
                handle=(c.OUT/'logs'/f'train_{arm}.log').open('a',encoding='utf-8'); handles.append(handle)
                processes.append(subprocess.Popen([sys.executable,__file__,'arm','--arm',arm],cwd=ROOT,
                    stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
            codes=[proc.wait() for proc in processes]
            for handle in handles: handle.close()
            if any(codes): raise RuntimeError('Training failed; see logs')
            learn.freeze()
    if stage in ('confirmation','all'): c.generate('test',workers=6); c.generate('stress',workers=6)
    if stage in ('evaluate','all'):
        from raised_acquisition.assessment import run
        run()
    if stage in ('audit','all'): c.audit()
    if stage in ('report','all'):
        from raised_acquisition.reporting import report
        report()


if __name__=='__main__': main()
