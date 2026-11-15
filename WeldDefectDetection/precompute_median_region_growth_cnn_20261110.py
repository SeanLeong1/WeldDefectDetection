"""Overlap unchanged frozen CNN inference with fresh-background generation."""
import argparse,os,subprocess,sys,time
import numpy as np
import median_region_growth_20261110 as s


def worker(i):
    f=s.verify_frozen();spec=f['original']['cnn'];model=s.base.model_for(spec)
    norm=s.read(s.OLD/'normalization.json');folder=s.OUT/'cnn_prefetch';folder.mkdir(exist_ok=True)
    rows=s.read(s.OUT/f'datasets/test/{i:03d}/index.json');target=folder/f'{i:03d}.npy';receipt=folder/f'{i:03d}.json'
    identity=dict(scene_ids=[r['scene_id'] for r in rows],raw_hashes=[r['raw_sha256'] for r in rows],checkpoint_sha256=spec['checkpoint_sha256'])
    if receipt.exists():assert s.read(receipt)==identity;return
    data=s.learn.Data(rows,norm)
    values=np.stack([s.base.predict_image(data.image(j),model) for j in range(len(rows))])
    np.save(target,values);s.write(receipt,identity)


def main():
    f=s.verify_frozen();spec=f['original']['cnn'];folder=s.OUT/'cnn_prefetch';folder.mkdir(exist_ok=True)
    if (folder/'execution.json').exists() and not (folder/'execution_serial.json').exists():
        s.write(folder/'execution_serial.json',s.read(folder/'execution.json'))
    s.write(folder/'execution.json',dict(started_utc=s.utc(),worker_sha256=s.sha(__file__),
        checkpoint_sha256=spec['checkpoint_sha256'],workers=3,torch_threads_per_worker=4,
        purpose='Unchanged frozen inference, scheduled alongside generation; no fitting or selection.'))
    complete={i for i in range(s.FRESH_COUNT) if (folder/f'{i:03d}.json').exists()};active=[]
    try:
        while len(complete)<s.FRESH_COUNT:
            for i in range(s.FRESH_COUNT):
                if len(active)>=3:break
                if i in complete or any(k==i for _,_,k in active) or not (s.OUT/f'datasets/test/{i:03d}/index.json').exists():continue
                fh=(folder/f'{i:03d}.log').open('a',encoding='utf8')
                proc=subprocess.Popen([sys.executable,'-B',__file__,'--worker',str(i)],stdout=fh,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((proc,fh,i))
            for proc,fh,i in active[:]:
                if proc.poll() is not None:
                    fh.close();active.remove((proc,fh,i))
                    if proc.returncode:raise RuntimeError(f'CNN worker {i} failed')
                    complete.add(i);print('CNN',len(complete),'/',s.FRESH_COUNT,flush=True)
            if len(complete)<s.FRESH_COUNT:time.sleep(2)
    finally:
        for proc,fh,_ in active:
            if proc.poll() is None:proc.terminate()
            fh.close()
    index=s.OUT/'datasets/test/index.json'
    while not index.exists():time.sleep(1)
    rows=s.read(index);identity=dict(scene_ids=[r['scene_id'] for r in rows],shape=[len(rows),128,5024])
    temporary=folder/'complete.npy';arr=np.lib.format.open_memmap(temporary,mode='w+',dtype='f4',shape=tuple(identity['shape']))
    for i in range(s.FRESH_COUNT):
        receipt=s.read(folder/f'{i:03d}.json');assert receipt['scene_ids']==identity['scene_ids'][i*18:(i+1)*18]
        arr[i*18:(i+1)*18]=np.load(folder/f'{i:03d}.npy',mmap_mode='r')
    arr.flush();del arr
    dest=s.OUT/'confirmation/cnn.npy';dest.parent.mkdir(exist_ok=True)
    # Publish only before the main evaluator finishes its first family.
    # Later, it owns the path; retain the separate blocks instead of racing.
    if not dest.exists() and not (dest.parent/'seed_only.json').exists():
        try:os.rename(temporary,dest)
        except FileExistsError:pass
        else:
            s.write(dest.with_suffix('.done.json'),identity)
            s.write(folder/'publication.json',dict(published_utc=s.utc(),score_sha256=s.sha(dest),frozen_inference_unchanged=True))
            print('CNN scores published before first median family completed',flush=True)
            return
    print('Separate CNN blocks retained; evaluator owns main path',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.worker is None:main()
    else:worker(args.worker)
