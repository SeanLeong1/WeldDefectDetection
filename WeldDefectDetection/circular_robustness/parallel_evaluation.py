"""Process scheduling only: execute frozen evaluation on disjoint backgrounds.

The frozen native evaluator is unchanged. A worker sees a partition of the scene
index and suppresses its partial aggregate. After workers finish, the original
evaluator reads every saved scene and writes the only promoted aggregate.
"""
from pathlib import Path
import argparse,subprocess,sys,time
from . import campaign as c,assessment_native


def worker(number,shards):
    original_read=c.read;original_write=c.write;target=(c.OUT/'index.json').resolve()
    def read(path):
        value=original_read(path)
        return [r for r in value if r['index']%shards==number] if Path(path).resolve()==target else value
    def write(path,value):
        if Path(path).resolve()==(c.OUT/'results.json').resolve():return
        original_write(path,value)
    c.read=read;c.write=write
    assessment_native.run()


def run(shards=2):
    assert shards==2,'This delivery uses two CPU workers, each retaining four Torch threads.'
    c.prepare();assert (c.OUT/'native_inputs/index.json').exists()
    folder=c.OUT/'execution';folder.mkdir(exist_ok=True)
    record=dict(schema='circular-runtime-scheduling/v1',workers=shards,torch_threads_each=4,
        source_sha256=c.sha(__file__),active_protocol_sha256=c.sha(c.OUT/'protocol_native_v2.json'),
        scope='Scheduling amendment only. Frozen code, data, thresholds and per-scene computations unchanged. Disjoint index modulo partitions; partial aggregates suppressed. Original evaluator aggregates all scene files after success.',
        completed_scene_files_on_entry=len(list((c.OUT/'predictions_native').glob('*.json'))))
    path=folder/'parallel_protocol.json'
    if not path.exists():c.write(path,record)
    else:
        previous=c.read(path)
        assert previous['source_sha256']==record['source_sha256'] and previous['active_protocol_sha256']==record['active_protocol_sha256']
    processes=[];logs=[]
    try:
        for i in range(shards):
            log=(folder/f'worker_{i}.log').open('a',encoding='utf-8');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-u','-m','circular_robustness.parallel_evaluation','--worker',str(i),'--shards',str(shards)],
                cwd=c.ROOT/'WeldDefectDetection',stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
        previous=-1
        while any(p.poll() is None for p in processes):
            completed=len(list((c.OUT/'predictions_native').glob('*.json')))
            if completed!=previous:print('scored scenes',completed,'/ 168',flush=True);previous=completed
            for p in processes:
                if p.poll() not in (None,0):raise RuntimeError('Worker failed; see execution/worker logs.')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        assert len(list((c.OUT/'predictions_native').glob('*.json')))==168
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait()
        for log in logs:log.close()
    assessment_native.run()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args()
    if args.worker is None:run(args.shards)
    else:worker(args.worker,args.shards)
