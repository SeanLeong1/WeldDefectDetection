"""Post-freeze diagnosis, never an operating-point selection stage."""
import numpy as np
import torch
from circular_study import metrics
from . import campaign as c,learning as learn,rendering as r,geometry as geo

def run():
    torch.set_num_threads(4);frozen=learn.verify_freeze();model=learn.model_for(frozen['cnn'])
    norm=c.read(c.OUT/'normalization.json');rows=[]
    for meta in c.read(c.OUT/'datasets/stress/index.json'):
        if meta['source_class']!='small_open':continue
        with np.load(meta['npz_path']) as a:image=learn.normalize(a['raw'][0],norm);mask=a['mask'][r.SCORE]
        data=dict(meta=[meta],mask=mask[None]);score=learn.predict_image(image,model)
        chosen=frozen['cnn']['selection']['best']
        for area in sorted({1,3,15,chosen['min_area']}):
            op=dict(chosen,min_area=area)
            row=metrics.evaluate(data,score[None],op,retain=True)[0]
            rows.append(dict(min_area=area,threshold=chosen['threshold'],score_max_in_truth=float(score[mask>0].max()) if mask.any() else None,**row))
    sampling=[]
    for diameter in (.15,.225,.3):
        for pitch in (.05,.025,.0125):
            for phase in (0.,.5):
                axis=np.arange(-1,1+pitch/2,pitch)+phase*pitch;u,arc=np.meshgrid(axis,axis,indexing='ij')
                p=dict(kind='keyhole_pore',effect='open',diameter_mm=diameter,depth_mm=.25,angle_rad=0.,radial_offset_mm=0.)
                _,mask,_=geo.surface_effect(u,arc/40,p,pitch**2*(1+u/40))
                area=float(np.sum(mask*pitch**2*(1+u/40)))
                expected=float(np.pi*(.5*diameter)*(.8*diameter)*(1-.35**2))
                sampling.append(dict(diameter_mm=diameter,pitch_mm=pitch,phase_pixels=phase,pixels=int(mask.sum()),
                    footprint_area_mm2=area,analytic_area_mm2=expected,relative_area_error=(area-expected)/expected))
    groups={str(a):metrics.summary([x for x in rows if x['min_area']==a]) for a in sorted({x['min_area'] for x in rows})}
    c.write(c.OUT/'small_pore_diagnostic.json',dict(rows=rows,component_area_comparison=groups,
        geometric_sampling=sampling,selection_effect='None; frozen threshold and production component rule unchanged',
        limit='Sampling sweep evaluates geometric footprints, not a full high-resolution detector/noise rerender',
        purpose='Separate component-area filtering, score response and footprint undersampling'))
    print('Small-pore diagnostic:',groups,flush=True)
