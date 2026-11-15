"""Run a frozen experimental policy on a raw, nominally registered TOPO array.

Example: python apply_median_scale_templates_20261115.py image.npy --count 64
Input shape is (160,5024). Output boxes are [left,top,right,bottom], exclusive
right/bottom, in scored rows (raw rows 16:144) and raw coordinates.
"""
import argparse,json
from pathlib import Path
import numpy as np
import median_scale_templates_20261115 as st

def infer(raw,count,policy='selected'):
    frozen=st.verify(True)
    if count not in (1,4,64):raise ValueError('Count must be 1, 4 or 64')
    if np.shape(raw)!=(160,5024):raise ValueError('Expected raw registered TOPO shape (160,5024)')
    if not np.isfinite(raw).all():raise ValueError('Input contains non-finite values')
    p=frozen['policies'][policy];op=p[str(count)]['op'] if policy=='per_count' else p['op']
    x=st.ex.normalized(raw);family=op['family']
    if family=='baseline':
        d=st.residual(x,'long');score=st.standardize(d,.003,st.ex.CAL_OLD)[16:144]
        boxes=st.ex.reconstruct(st.ex.components(score,16,3),st.ex.ORIGINAL_OP)
    elif family=='previous':
        old=st.read(st.ex.OUT/'frozen.json')['policies']['classical']['op']
        d=st.residual(x,'short');score=st.standardize(d,.003)[16:144]
        boxes=st.ex.reconstruct(st.ex.components(score,old['threshold'],old['closing']),old)
    elif family=='components':
        if op['score']=='baseline':d=st.residual(x,'long');score=st.standardize(d,.003,st.ex.CAL_OLD)[16:144]
        else:
            ref,mode,floor=op['score'].split('_');d=st.residual(x,ref)
            cal=st.read(st.OUT/'training_calibration.json')['count_calibration'][str(count)] if mode=='count' else None
            score=st.standardize(d,float(floor),cal)[16:144]
        boxes=st.boxes_from_components(st.ex.components(score,op['threshold'],3),op)
    else:
        d=st.residual(x,op['reference'])
        if family=='multiscale':candidates=st.multiscale(d,op['reference'])
        elif family=='templates':
            with np.load(st.OUT/'templates.npz') as z:bank=z['bank']
            candidates=st.template_candidates(d,bank,st.read(st.OUT/'training_calibration.json')['templates'])
        else:raise ValueError(f'Unknown family {family}')
        boxes=st.nms_peaks(candidates,op)
    return dict(policy=policy,operating_point=op,count_multiplier=count,scored_boxes=boxes,
        raw_boxes=[[l,t+16,r,b+16] for l,t,r,b in boxes],angular_width=5024,periodic=True,
        scope='Frozen synthetic experiment; no measured-image calibration or automatic ring localisation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw');p.add_argument('--count',type=int,choices=(1,4,64),required=True)
    p.add_argument('--policy',default='selected');p.add_argument('--output');a=p.parse_args()
    result=infer(np.load(a.raw),a.count,a.policy)
    if a.output:st.write(Path(a.output),result)
    else:print(json.dumps(result,indent=2))
