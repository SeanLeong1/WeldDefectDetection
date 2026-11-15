"""Overlap frozen CNN inference with fresh rendering; never change selection.

Publishes a complete score file only while the main run has not yet completed
its first median evaluation. Otherwise it leaves separate reusable block files.
"""
import time, os
import numpy as np
import median_localisation_20261110 as s

def main():
    frozen=s.verify_frozen();spec=frozen['original']['cnn'];model=s.base.model_for(spec)
    norm=s.read(s.OLD/'normalization.json');folder=s.OUT/'cnn_prefetch';folder.mkdir(exist_ok=True)
    done=set()
    s.write(folder/'execution.json',dict(started_utc=s.utc(),worker_sha256=s.sha(__file__),
        checkpoint_sha256=spec['checkpoint_sha256'],purpose='Same frozen inference, overlapped with rendering. No selection or model update.'))
    while len(done)<s.FRESH_COUNT:
        progressed=False
        for i in range(s.FRESH_COUNT):
            index=s.OUT/f'datasets/test/{i:03d}/index.json';target=folder/f'{i:03d}.npy';receipt=folder/f'{i:03d}.json'
            if i in done or not index.exists():continue
            rows=s.read(index);identity=dict(scene_ids=[r['scene_id'] for r in rows],raw_hashes=[r['raw_sha256'] for r in rows],checkpoint_sha256=spec['checkpoint_sha256'])
            if receipt.exists():assert s.read(receipt)==identity
            else:
                data=s.learn.Data(rows,norm)
                values=np.stack([s.base.predict_image(data.image(j),model) for j in range(len(rows))])
                np.save(target,values);s.write(receipt,identity)
            done.add(i);progressed=True;print('CNN PREFETCH',i,len(done),'/',s.FRESH_COUNT,flush=True)
        if not progressed:time.sleep(2)
    index=s.OUT/'datasets/test/index.json'
    while not index.exists():time.sleep(1)
    rows=s.read(index);identity=dict(scene_ids=[r['scene_id'] for r in rows],shape=[len(rows),128,5024])
    temp=folder/'complete.npy';arr=np.lib.format.open_memmap(temp,mode='w+',dtype='f4',shape=tuple(identity['shape']))
    for i in range(s.FRESH_COUNT):
        block=np.load(folder/f'{i:03d}.npy',mmap_mode='r');receipt=s.read(folder/f'{i:03d}.json')
        assert receipt['scene_ids']==identity['scene_ids'][i*18:(i+1)*18]
        arr[i*18:(i+1)*18]=block
    arr.flush();del arr
    dest=s.OUT/'confirmation/cnn.npy';dest.parent.mkdir(exist_ok=True)
    # The controller evaluates two median sigma files before reaching CNN.
    # Restrict publication to the first median stage to avoid a concurrent writer.
    if not dest.exists() and not (dest.parent/'original.json').exists():
        # Windows rename is atomic and refuses an existing destination. Hard
        # links are unavailable in some OneDrive-managed project directories.
        try:os.rename(temp,dest)
        except FileExistsError:pass
        else:
            s.write(dest.with_suffix('.done.json'),identity)
            s.write(folder/'publication.json',dict(published_utc=s.utc(),score_sha256=s.sha(dest),
                checkpoint_sha256=spec['checkpoint_sha256'],frozen_inference_unchanged=True))
            print('CNN complete scores published before median evaluation finished',flush=True)
            return
    print('Separate complete CNN scores retained; main evaluator owns its score path',flush=True)

if __name__=='__main__':main()
