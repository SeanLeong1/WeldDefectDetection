"""Isolated deterministic small-pore detector sampling, not CV retuning."""
from unittest.mock import patch
import numpy as np
from scipy.ndimage import gaussian_filter
from . import campaign as c,geometry as g,rendering as r
from .shadow import VisibilityQuadrature,q


def run():
    cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config');seed=c.BASES['pilot']
    sample,_=r.oldg.baseline.c.sampled(cfg,seed,0);detector=sample.detector
    pore=dict(kind='keyhole_pore',effect='open',diameter_mm=.225,depth_mm=.25,angle_rad=0.,radial_offset_mm=0.)
    def height(x,y,defect):
        u=np.hypot(x,y)-40;theta=np.arctan2(y,x);h=g.height(u,theta,seed,0.)
        if defect:h+=g.surface_effect(u,theta,pore,1.)[0]
        return h
    class Direct(VisibilityQuadrature):
        def __init__(self,x,y,z,defect):self.x=x;self.y=y;self.z=z;self.defect=defect;self.maps={}
        def at(self,az):
            key=float(az)
            if key not in self.maps:
                horizon=np.full(self.x.shape,-np.inf)
                reach=3.1*detector.outer_radius_mm/detector.height_mm+.05
                for s in np.arange(.025,reach+.0125,.025):
                    other=height(self.x+s*np.cos(az),self.y+s*np.sin(az),self.defect)
                    horizon=np.maximum(horizon,(other-self.z)/s)
                self.maps[key]=horizon
            return self.maps[key]
    engine=r.Engine(seed,0.)
    sigma_mm=engine._interaction_blur_px(sample.beam.focal_spot_mm,sample.beam.e0_kev,.05)*.05
    saved={};records=[]
    for pitch in (.05,.025,.0125):
        axis=np.linspace(-1,1,round(2/pitch)+1);yy,xx=np.meshgrid(axis,axis+40,indexing='ij')
        images=[]
        for defect in (False,True):
            h=height(xx,yy,defect);normals=r.old.se.ScanEngine._normals_from_heightmap(h,pitch);out=[]
            for start in range(0,h.size,2048):
                ids=slice(start,start+2048);v=Direct(xx.ravel()[ids],yy.ravel()[ids],h.ravel()[ids],defect)
                with patch.object(q,'_integrated_quadrant_directional_term',v.integrated):
                    value=q.compute_quadrant_signals(normals.reshape(-1,3)[ids][None],detector,sample.material,
                        np.full((1,len(v.x)),sample.material.z),b_coeff=0.)[0]
                out.append(value)
            quadrants=np.concatenate(out).reshape(*h.shape,4)
            blurred=gaussian_filter(quadrants,(sigma_mm/pitch,sigma_mm/pitch,0),mode='reflect')
            topo=r.oldg.baseline.form_channels(blurred).topo;images.append(topo)
        delta=images[1]-images[0];central=abs(axis)<=.5
        saved[str(pitch)]=delta;records.append(dict(pitch_mm=pitch,peak_absolute_signal=float(abs(delta[np.ix_(central,central)]).max()),
            rms_signal=float(np.sqrt(np.mean(delta[np.ix_(central,central)]**2)))))
        print('Small-pore detector sampling',pitch,flush=True)
    reference=saved['0.0125']
    for row in records:
        step=round(row['pitch_mm']/.0125);aligned=reference[::step,::step];delta=saved[str(row['pitch_mm'])]
        n=delta.shape[0];lo=round(.5/row['pitch_mm']);core=(slice(lo,n-lo),)*2
        row['relative_rms_difference_to_12_5um']=float(np.sqrt(np.mean((delta[core]-aligned[core])**2))/max(np.sqrt(np.mean(aligned[core]**2)),1e-12))
    c.write(c.OUT/'small_pore_signal_sampling.json',dict(records=records,pore=pore,physical_blur_sigma_mm=sigma_mm,
        ray_step_mm=.025,angular_model='Existing Lambertian quadrant integrator with direct analytic nonlocal visibility',
        controls='Same physical 2x2 mm field, crown, pore, detector, unit yield modifiers and fixed physical Gaussian blur; compare central 1x1 mm',
        limits='Isolated deterministic detector-signal diagnostic, not a full noisy raster_v2 rerender, not calibrated SEM resolution, and not a re-evaluation of trained CV at another pitch',
        selection_effect='None; production pitch, models, threshold and component rule remain frozen'))
    np.savez_compressed(c.OUT/'small_pore_signal_sampling.npz',**{'delta_'+k.replace('.','p'):v for k,v in saved.items()})
