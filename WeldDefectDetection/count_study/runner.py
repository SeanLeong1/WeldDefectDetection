"""Durable stage orchestration with bounded subprocess concurrency."""
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
from . import campaign as c


def run_jobs(stage,jobs,workers):
    logs=c.OUT/'logs';logs.mkdir(exist_ok=True)
    pending=list(jobs);active=[];completed=[]
    while pending or active:
        while pending and len(active)<workers:
            name,args=pending.pop(0)
            log=logs/f'{name}.log';handle=log.open('a',encoding='utf-8')
            process=subprocess.Popen([sys.executable,str(c.ROOT/'WeldDefectDetection/run_count_study_20261027.py'),*args],
                cwd=c.ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((name,process,handle,log))
        status=dict(state='running',stage=stage,pid=os.getpid(),
            active=[dict(name=n,pid=p.pid,log=str(log)) for n,p,h,log in active],
            pending=[n for n,a in pending],completed=completed,
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        c.write(c.OUT/'execution_status.json',status)
        for name,process,handle,log in active[:]:
            if process.poll() is not None:
                handle.close();active.remove((name,process,handle,log))
                if process.returncode:
                    # Let other independent jobs finish; preserve their artifacts.
                    for _,other,fh,_ in active:other.wait();fh.close()
                    raise RuntimeError(f'{stage}/{name} exited {process.returncode}; inspect {log}')
                completed.append(name);print('Completed',stage,name,flush=True)
        if active:time.sleep(5)


def main():
    try:
        from . import learning as learn, reporting
        c.prepare();c.noise_check();learn.check_metrics()
        if '--adopt-training-generation' in sys.argv:
            run_jobs('development_data',[('generate_val',['generate','--split','val','--workers','2'])],1)
            while not (c.OUT/'datasets/train/index.json').exists():
                c.write(c.OUT/'execution_status.json',dict(state='running',stage='waiting_for_existing_training_generation',
                    pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                time.sleep(10)
        else:
            run_jobs('development_data',[('generate_train',['generate','--split','train','--workers','4']),
                                         ('generate_val',['generate','--split','val','--workers','4'])],1)
        learn.prepare()
        # Independent arms run concurrently. Each process uses four Torch threads.
        run_jobs('training_and_validation',[(f'develop_{a}',['develop','--arm',a]) for a in c.ARMS],2)
        learn.freeze()
        run_jobs('fresh_confirmation_data',[('generate_test',['generate','--split','test','--workers','4'])],1)
        run_jobs('confirmation_evaluation',[(f'evaluate_{a}',['evaluate','--arm',a]) for a in c.ARMS],2)
        # This reuses completed arm results, then runs retained controls and writes the aggregate.
        learn.evaluate()
        c.audit();reporting.report()
        c.write(c.OUT/'execution_status.json',dict(state='complete',stage='report_and_audit',pid=os.getpid(),
            report=str(c.ROOT/'docs/COUNT_DIVERSITY_RESULTS_20261027.md'),
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    except BaseException as exc:
        c.write(c.OUT/'execution_status.json',dict(state='failed',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc(),
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        raise


if __name__=='__main__':main()
