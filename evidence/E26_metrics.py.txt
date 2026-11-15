"""Periodic connected regions and one-to-one circular bounding-box metrics."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def regions(mask,min_area=1):
    mask=np.asarray(mask,dtype='u1');h,w=mask.shape
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    parent=np.arange(n)
    def root(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    for y in range(h):
        if lab[y,0]:
            for yy in range(max(0,y-1),min(h,y+2)):
                if lab[yy,-1]:parent[root(lab[yy,-1])]=root(lab[y,0])
    mapping=np.array([root(i) for i in range(n)])
    lab=mapping[lab];areas=np.bincount(lab.ravel(),minlength=n)
    keep=(areas>=min_area);keep[0]=False
    boxes=[]
    seam=set(lab[:,0][lab[:,0]>0]) & set(lab[:,-1][lab[:,-1]>0])
    for k in np.flatnonzero(keep):
        if k not in seam:
            x,y,ww,hh,_=stats[k]
            boxes.append([int(x),int(y),int(x+ww),int(y+hh)])
            continue
        ys,xs=np.where(lab==k);columns=np.unique(xs)
        gaps=np.diff(np.r_[columns,columns[0]+w]);j=int(np.argmax(gaps))
        start=int(columns[(j+1)%len(columns)])
        length=int(w-gaps[j]+1)
        boxes.append([start,int(ys.min()),start+length,int(ys.max()+1)])
    return keep[lab].astype('u1'),boxes


def binary(score,threshold,area,closing):
    m=(score>=threshold).astype('u1')
    if closing>1:
        pad=closing
        padded=np.pad(m,((0,0),(pad,pad)),mode='wrap')
        m=cv2.morphologyEx(padded,cv2.MORPH_CLOSE,np.ones((closing,closing),'u1'))[:,pad:-pad]
    return regions(m,area)


def circular_iou(a,b,period):
    vertical=max(0,min(a[3],b[3])-max(a[1],b[1]))
    horizontal=sum(max(0,min(a[2],b[2]+s)-max(a[0],b[0]+s)) for s in (-period,0,period))
    intersection=vertical*horizontal
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def match(pred,truth,period):
    tp=0;total=0.
    if pred and truth:
        ious=np.array([[circular_iou(p,t,period) for t in truth] for p in pred])
        ok=ious>=.5;ii,jj=linear_sum_assignment(-(ok*(min(len(pred),len(truth))+1+ious)))
        pairs=[(i,j) for i,j in zip(ii,jj) if ok[i,j]]
        tp=len(pairs);total=float(sum(ious[i,j] for i,j in pairs))
    return dict(tp=tp,fp=len(pred)-tp,fn=len(truth)-tp,matched_iou_sum=total)


def evaluate(data,scores,spec,retain=False):
    if scores.shape!=data['mask'].shape or len(data['meta'])!=len(scores):
        raise ValueError('One complete score annulus and mask are required for every scene')
    rows=[]
    if 'truth' not in data:data['truth']=[regions(m)[1] for m in data['mask']]
    for score,truth,boxes,meta in zip(scores,data['mask'],data['truth'],data['meta']):
        m,pred=binary(score,spec['threshold'],spec['min_area'],spec['closing'])
        row=dict(scene_id=meta['scene_id'],background_id=meta['background_id'],source_class=meta['source_class'],
            amplitude=meta['amplitude'],**match(pred,boxes,score.shape[1]),
            intersection=int(((m>0)&(truth>0)).sum()),union=int(((m>0)|(truth>0)).sum()),
            predicted_pixels=int(m.sum()),truth_pixels=int(truth.sum()),predicted_regions=len(pred))
        if retain:row.update(predicted_boxes=pred,truth_boxes=boxes)
        rows.append(row)
    return rows


def summary(rows):
    tp,fp,fn=[sum(r[k] for r in rows) for k in ('tp','fp','fn')]
    clean=[r for r in rows if r['source_class']=='clean']
    union=sum(r['union'] for r in rows)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        f1=2*tp/max(2*tp+fp+fn,1),pixel_iou=sum(r['intersection'] for r in rows)/max(union,1),
        clean_rings=len(clean),clean_rings_flagged=sum(r['predicted_regions']>0 for r in clean),
        clean_ring_fa=sum(r['predicted_regions']>0 for r in clean)/max(len(clean),1),
        clean_false_regions=sum(r['fp'] for r in clean),scenes=len(rows))


def key(metrics):
    return metrics['f1'],metrics['pixel_iou'],-metrics['fp']


def select(data,scores,thresholds,fast=False):
    if not data['meta'] or any(r['split']!='val' for r in data['meta']):
        raise ValueError('Operating-point selection requires validation data only')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite validation score')
    trials=[]
    for t in thresholds:
        for area in (3,15):
            for close in ((1,) if fast else (1,3)):
                s=dict(threshold=float(t),min_area=area,closing=close)
                trials.append(dict(**s,metrics=summary(evaluate(data,scores,s))))
    best=max(trials,key=lambda t:key(t['metrics']))
    feasible=[t for t in trials if t['metrics']['clean_ring_fa']<=.10]
    low=max(feasible,key=lambda t:(t['metrics']['recall'],t['metrics']['precision'],t['metrics']['pixel_iou'])) if feasible else None
    return dict(best=best,low_fa=low,trials=trials)


def bootstrap(rows,repeats=2000,seed=100483001):
    groups={k:[r for r in rows if r['background_id']==k] for k in sorted({r['background_id'] for r in rows})}
    keys=list(groups);rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        sample=[r for k in rng.choice(keys,len(keys),replace=True) for r in groups[k]]
        m=summary(sample);values.append([m['f1'],m['recall'],m['clean_ring_fa']])
    ci=np.quantile(values,[.025,.975],axis=0)
    return {k:ci[:,i].tolist() for i,k in enumerate(('f1','recall','clean_ring_fa'))}
