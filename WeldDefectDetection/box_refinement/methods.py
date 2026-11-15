from __future__ import annotations
import cv2
import numpy as np
from box_comparison.core import boxes_from_mask, match_boxes, summarize, selection_key
from src.utils.image_preprocessing import filter_small_components


def residual_score(target, reference, arm, sigma=1., standardized=True):
    """Remove the continuous horizontal weld/shading profile without labels."""
    target=np.asarray(target,dtype=np.float32)
    residual=target if arm=='thresholding' else target-np.asarray(reference,dtype=np.float32)
    residual=residual-np.median(residual,axis=1,keepdims=True)
    smooth=cv2.GaussianBlur(residual,(0,0),sigmaX=sigma,sigmaY=sigma)
    if standardized:
        # Per-crop robust scale is label-free. The fixed floor prevents nearly
        # constant backgrounds from magnifying quantization/noise arbitrarily.
        scale=max(.003,float(1.4826*np.median(np.abs(smooth-np.median(smooth)))))
        smooth=smooth/scale
    return np.abs(smooth).astype(np.float32)


def predicted_regions(score,threshold,min_area,closing=1,low_ratio=1.,padding=0):
    score=np.asarray(score)
    if score.ndim!=2 or not np.isfinite(score).all():raise ValueError('Expected a finite 2D score image.')
    if threshold<=0 or not 0<low_ratio<=1 or min_area<1 or padding<0 or closing not in (1,3):
        raise ValueError('Invalid frozen component settings.')
    high=score>=threshold
    weak=(score>=threshold*low_ratio).astype(np.uint8)
    if closing>1:weak=cv2.morphologyEx(weak,cv2.MORPH_CLOSE,np.ones((closing,closing),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(weak,connectivity=8)
    keep=np.zeros(n,np.uint8)
    supported=np.unique(labels[high & (weak>0)])
    keep[supported]=1;keep[0]=0
    keep[1:] *= (stats[1:,cv2.CC_STAT_AREA]>=min_area)
    mask=keep[labels]
    h,w=mask.shape
    boxes=[[max(0,x0-padding),max(0,y0-padding),min(w,x1+padding),min(h,y1+padding)]
           for x0,y0,x1,y1 in boxes_from_mask(mask)]
    return mask,boxes


def evaluate(data,scores,parameters):
    if len(scores)!=len(data['meta']):raise ValueError('Scores and targets differ.')
    rows=[]
    for meta,truth,score in zip(data['meta'],data['mask'],scores):
        mask,boxes=predicted_regions(score,**parameters)
        pred=mask>0;truth=truth>0
        rows.append(dict(**meta,boxes_xyxy=boxes,**match_boxes(boxes,meta['truth_boxes']),
            pixel_tp=int(np.sum(truth&pred)),pixel_fp=int(np.sum(~truth&pred)),pixel_fn=int(np.sum(truth&~pred)),
            target_positive=bool(truth.any()),predicted_positive=bool(boxes)))
    return rows


def select(data,scores,thresholds,*,min_areas=(1,5,15),closings=(1,3),low_ratios=(.5,1.),paddings=(0,2,4)):
    if not data['meta'] or any(m['split']!='val' for m in data['meta']):
        raise ValueError('Selection requires nonempty development validation data.')
    best=None;trials=[]
    for threshold in thresholds:
        for area in min_areas:
            for closing in closings:
                for ratio in low_ratios:
                    p=dict(threshold=float(threshold),min_area=area,closing=closing,low_ratio=ratio,padding=0)
                    base=evaluate(data,scores,p)
                    for padding in paddings:
                        rows=[]
                        for row,score in zip(base,scores):
                            h,w=score.shape
                            boxes=[[max(0,x0-padding),max(0,y0-padding),min(w,x1+padding),min(h,y1+padding)]
                                   for x0,y0,x1,y1 in row['boxes_xyxy']]
                            rows.append({**row,'boxes_xyxy':boxes,**match_boxes(boxes,row['truth_boxes'])})
                        metrics=summarize(rows)
                        trial=dict(parameters={**p,'padding':padding},metrics=metrics)
                        trials.append(trial)
                        if best is None or selection_key(metrics)>selection_key(best['metrics']):best=trial
    return best,trials


def augment_pair(target,reference,mask,generator):
    """Common transformations keep target, reference and truth aligned."""
    import torch
    batch=len(target)
    flip=torch.rand(batch,generator=generator)<.5
    target=target.clone();reference=reference.clone();mask=mask.clone()
    target[flip]=target[flip].flip(-1);reference[flip]=reference[flip].flip(-1);mask[flip]=mask[flip].flip(-1)
    gain=.9+.2*torch.rand((batch,1,1,1),generator=generator)
    offset=-.02+.04*torch.rand((batch,1,1,1),generator=generator)
    return (target*gain+offset).clamp(0,1),(reference*gain+offset).clamp(0,1),mask
