"""Descriptive warmed CPU inference benchmark; no labels used in scoring."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'DataSimulation'))
from variability_campaign.campaign import read,write,sha,ARMS
from box_development.methods import make_network,predict,simple_scores
from box_refinement.methods import predicted_regions


def measure(root):
    torch.set_num_threads(4)
    frozen=read(root/'frozen_models.json');norm=read(root/'normalization.json')
    r=read(root/'datasets/development/index.json')[0]
    with np.load(r['npz_path']) as a:
        raw=a['raw'];raw=np.sqrt(np.clip((raw-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1)).astype(np.float32)
    frames=np.stack([raw[:,k*64:(k+1)*64] for k in range(9)])
    target=frames[4:5];context=frames[[2,3,5,6]][None]
    model=make_network(1)
    spec=frozen['models']['cnn'];assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    model.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)['state_dict']);model.eval()
    rows=[]
    for batch in (1,32):
        data=dict(target=np.repeat(target,batch,axis=0),context=np.repeat(context,batch,axis=0))
        for arm in ARMS:
            s=frozen['models'][arm]
            def once():
                scores=predict(model,data['target'][:,None])[:,0] if arm=='cnn' else simple_scores(data,s)
                return [predicted_regions(v,**s['parameters'])[1] for v in scores]
            once();times=[]
            for _ in range(20):
                start=time.perf_counter();once();times.append(time.perf_counter()-start)
            rows.append(dict(arm=arm,batch_size=batch,repetitions=20,median_batch_ms=float(np.median(times)*1000),
                median_ms_per_crop=float(np.median(times)*1000/batch),p10_batch_ms=float(np.quantile(times,.1)*1000),p90_batch_ms=float(np.quantile(times,.9)*1000)))
    result=dict(device='CPU',torch_threads=4,cuda_available=torch.cuda.is_available(),
                model_parameters=sum(p.numel() for p in model.parameters()),input_shape=[64,64],rows=rows,
                scope='Warmed score calculation and connected-component boxes; excludes file IO, model loading and shared raw normalization.',
                caveat='Concurrent rendering and repeated inputs; descriptive local timings, not real-time deployment validation.',
                script_sha256=sha(__file__))
    write(root/'inference_runtime.json',result)
    print(result)


if __name__=='__main__':
    measure(REPO/'DataSimulation/data/output/variability_campaign_20260926_v1')
