"""Disjoint-background process scheduling; unchanged frozen scene generator."""
from pathlib import Path
import builtins,subprocess,sys,time
from . import campaign as c

def worker(split,index,count):
    # Each process has its own module namespace and RNGs. The frozen generator's
    # only module-level range call schedules backgrounds. Scientific seeds and
    # global background indices remain unchanged.
    c.range=lambda total:builtins.range(index,total,count)
    original_write=c.write
    target=c.OUT/f'datasets/{split}/index.json'
    def partition_write(path,payload):
        path=Path(path)
        if path==target:path=path.with_name(f'index.part{index}.json')
        original_write(path,payload)
    c.write=partition_write
    c.generate(split)

def run(split,count=2):
    c.verify_sources();logs=c.OUT/'execution_logs';logs.mkdir(exist_ok=True)
    receipt=logs/f'{split}_scheduler.json'
    record=dict(source=str(Path(__file__)),sha256=c.sha(__file__),workers=count,
        partitions=[list(range(i,c.COUNTS[split],count)) for i in range(count)],
        policy='Disjoint background scheduling only; unchanged frozen scene renderer and seeds')
    if receipt.exists():assert c.read(receipt)==record
    else:c.write(receipt,record)
    procs=[];handles=[]
    for index in range(count):
        handle=(logs/f'{split}_{index}.log').open('a',encoding='utf-8');handles.append(handle)
        command=[sys.executable,str(c.ROOT/'WeldDefectDetection/run_raised_parallel_20261005.py'),split,'--worker',str(index),'--workers',str(count)]
        procs.append(subprocess.Popen(command,cwd=c.ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
    while any(p.poll() is None for p in procs):
        time.sleep(10)
        done=len(list((c.OUT/f'datasets/{split}').glob(f'{split}-*.json')))
        print(split,'completed scene receipts',done,flush=True)
    for handle in handles:handle.close()
    if any(p.returncode for p in procs):raise RuntimeError('Generation worker failed; inspect execution_logs, resume preserves completed scenes')
    rows=[]
    for index in range(count):rows.extend(c.read(c.OUT/f'datasets/{split}/index.part{index}.json'))
    ids=[x['scene_id'] for x in rows];assert len(ids)==len(set(ids))
    assert {x['index'] for x in rows}==set(range(c.COUNTS[split]))
    for row in rows:assert c.sha(row['npz_path'])==row['npz_sha256']
    c.write(c.OUT/f'datasets/{split}/index.json',sorted(rows,key=lambda x:x['scene_id']))
