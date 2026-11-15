"""Neighbouring-median ablations, with train-only robust calibration."""
import cv2
import numpy as np
from box_development.methods import aligned_reference
from . import metrics
from .generation import OUT,write,read,load

VARIANTS=['legacy_scale','centred','aligned','conditional','neighbours8','context_adaptive']


def tile(image,centre,size=64):
    return image[..., (np.arange(size)+centre-size//2)%image.shape[-1]]


def align(target,neighbour):
    def moments(image):
        p=np.median(image,axis=1);p=np.maximum(p-p.min(),0)+1e-5
        y=np.arange(len(p));c=np.sum(y*p)/sum(p)
        return c,np.sqrt(np.sum((y-c)**2*p)/sum(p))
    tc,tw=moments(target);nc,nw=moments(neighbour)
    scale=np.clip(nw/max(tw,1e-6),.9,1.1);shift=np.clip(nc-tc,-4,4)
    yy=np.arange(64,dtype='f4')[:,None];xx=np.arange(64,dtype='f4')[None,:]
    y=np.broadcast_to((yy-tc)*scale+tc+shift,(64,64)).astype('f4')
    return cv2.remap(neighbour,np.broadcast_to(xx,(64,64)),y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)


def residual(image,centre,variant):
    target=tile(image,centre)
    offsets=(-256,-192,-128,-64,64,128,192,256) if variant=='neighbours8' else (-128,-64,64,128)
    context=np.stack([tile(image,centre+k) for k in offsets])
    if variant=='aligned':context=np.stack([align(target,n) for n in context])
    ref=aligned_reference(target,context)
    value=cv2.GaussianBlur(target-ref,(0,0),1.)
    dispersion=float(np.median(abs(context-np.median(context,axis=0))))
    return value,float(np.quantile(target,.95)),dispersion


def fit(data,variant):
    if any(r['split']!='train' for r in data['meta']):raise ValueError('Train only')
    values=[];brightness=[];disp=[]
    for image,meta in zip(data['x'][:,0,16:80],data['meta']):
        if meta['source_class']!='clean':continue
        for centre in range(0,image.shape[1],256):
            r,b,d=residual(image,centre,variant);values.append(r);brightness.append(b);disp.append(d)
    values=np.stack(values);brightness=np.array(brightness)
    edges=np.quantile(brightness,[.25,.5,.75])
    labels=np.digitize(brightness,edges) if variant=='conditional' else np.zeros(len(values),int)
    calibration=[]
    for k in range(4 if variant=='conditional' else 1):
        v=values[labels==k];centre=np.median(v,axis=0)
        scale=np.maximum(.003,cv2.GaussianBlur(1.4826*np.median(abs(v-centre),axis=0),(0,0),2.))
        calibration.append(dict(centre=centre.tolist(),scale=scale.tolist(),windows=len(v)))
    return dict(variant=variant,calibration=calibration,brightness_edges=edges.tolist(),
        context_dispersion=float(np.median(disp)),fit_split='train',fit_windows=len(values))


def scores(data,spec):
    output=[];variant=spec['variant']
    centres=[np.array(c['centre'],dtype='f4') for c in spec['calibration']]
    scales=[np.array(c['scale'],dtype='f4') for c in spec['calibration']]
    for image in data['x'][:,0,16:80]:
        result=np.empty_like(image)
        for start in range(0,image.shape[1],32):
            # Each target contributes its central 32 columns, with periodic halo.
            centre=start+16;r,b,d=residual(image,centre,variant)
            k=int(np.digitize(b,spec['brightness_edges'])) if variant=='conditional' else 0
            scale=scales[k]
            if variant=='context_adaptive':scale=scale*np.clip(d/max(spec['context_dispersion'],.003),.75,2.)
            score=abs(r-(0 if variant=='legacy_scale' else centres[k]))/scale
            result[:,start:start+32]=score[:,16:48]
        output.append(result)
    return np.stack(output)


def run(train,val):
    results=[]
    for variant in VARIANTS:
        path=OUT/f'training/median_{variant}.json'
        if path.exists():results.append(read(path));continue
        spec=fit(train,variant);s=scores(val,spec)
        choice=metrics.select(val,s,[2,3,4,6,8,12,20,40,80])
        result=dict(method='median',variant=variant,spec=spec,selection=choice)
        write(path,result);results.append(result)
        print('median',variant,choice['best']['metrics'],flush=True)
    return results
