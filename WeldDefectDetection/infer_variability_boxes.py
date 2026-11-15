"""Label-free inference for the frozen paired-variability experiment.

Input NPZ: target_raw (64,64), context_raw (4,64,64), pixel_size_mm scalar.
CNN only needs target_raw; context is required by the median method. Output is
half-open xyxy pixel boxes and the frozen decision settings.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from box_comparison.core import sha,write
from box_development.methods import scores_for
from box_refinement.methods import predicted_regions


def infer(campaign, source, arm='cnn'):
    campaign=Path(campaign)
    frozen=json.loads((campaign/'frozen_models.json').read_text(encoding='utf-8'))
    if sha(campaign/'normalization.json')!=frozen['normalization_sha256']:
        raise ValueError('Normalization changed after model freezing.')
    norm=json.loads((campaign/'normalization.json').read_text(encoding='utf-8'))
    with np.load(source,allow_pickle=False) as a:
        target=np.asarray(a['target_raw'],dtype=np.float32)
        spacing=float(np.asarray(a['pixel_size_mm']).item())
        context=np.asarray(a['context_raw'],dtype=np.float32) if 'context_raw' in a else np.zeros((4,64,64),np.float32)
        if arm=='median' and 'context_raw' not in a:raise ValueError('Median requires four neighbouring crops.')
    if target.shape!=(64,64) or context.shape!=(4,64,64) or not np.isfinite(target).all() or not np.isfinite(context).all():
        raise ValueError('Expected finite 64x64 target and four 64x64 context arrays.')
    if not np.isclose(spacing,.05,atol=1e-9):raise ValueError('Model supports 0.05 mm pixels only.')
    def normalize(raw):
        return np.sqrt(np.clip((raw-norm['raw_low'])/(norm['raw_high']-norm['raw_low']),0,1)).astype(np.float32)
    x=normalize(target)[None];c=normalize(context)[None]
    data=dict(target=x,context=c,reference=np.median(c,axis=1))
    spec=frozen['models'][arm]
    score=scores_for(data,spec)[0,0]
    _,boxes=predicted_regions(score,**spec['parameters'])
    return dict(approach=arm,boxes_xyxy=boxes,coordinate_policy='half-open pixel indices',
                parameters=spec['parameters'],pixel_size_mm=spacing,physical_validation=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=Path(__file__).resolve().parents[1]/'DataSimulation/data/output/variability_campaign_20260926_v1')
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--approach',choices=['cnn','median','thresholding'],default='cnn')
    args=parser.parse_args();torch.set_num_threads(4)
    write(args.output,infer(args.campaign,args.input,args.approach))
