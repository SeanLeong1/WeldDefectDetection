"""Evaluate frozen methods as independent new backgrounds finish generation."""
import time
import numpy as np
import median_exploration_20261115 as ex

def main():
    ex.verify(True)
    spec=ex.read(ex.OUT/'protocol.json')['cnn'];model=ex.base.model_for(spec)
    (ex.OUT/'confirmation_scenes').mkdir(exist_ok=True)
    folder=ex.OUT/'cnn_scene_results';folder.mkdir(exist_ok=True)
    pending=set(range(ex.N_FRESH));started=time.monotonic()
    while pending:
        ready=[i for i in sorted(pending) if (ex.OUT/f'datasets/test/{i:03d}/index.json').exists()]
        if not ready:
            if time.monotonic()-started>10800:raise TimeoutError('Waiting for generation for three hours')
            time.sleep(3);continue
        for i in ready:
            for r in ex.read(ex.OUT/f'datasets/test/{i:03d}/index.json'):
                ex.confirm_worker(r)
                p=folder/f"{r['scene_id']}.json"
                if p.exists():continue
                raw=np.load(r['raw_path']);score=ex.base.predict_image(ex.base.normalize(raw,ex.NORM),model)
                op=spec['selection']['best'];_,pred=ex.metrics.binary(score,op['threshold'],op['min_area'],op['closing'])
                result=ex.result_row({k:r[k] for k in ('scene_id','background_id','source_class','count_multiplier','diameter_mm','repeat')},ex.truth_of(r),pred)
                ex.write(p,result)
            pending.remove(i)
            print('FROZEN EVALUATION COMPLETE',i,'remaining',len(pending),flush=True)
    print('ALL FROZEN EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
