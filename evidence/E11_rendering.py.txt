"""New geometry through the preserved native Cartesian detector/acquisition code."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from scipy.ndimage import map_coordinates
from circular_study import generation as oldg
from bse_renderer.models import VoxelVolumeState
from . import geometry as geo
from .shadow import VisibilityQuadrature

ROWS=160
SCORE=slice(16,144)  # 6.4 mm scored width, 8 mm extraction, same .05 mm pitch
COLS=5024
PAD=6.
old=oldg.old

def extract(array,rows=ROWS,order=1,error=None,pitch=.05):
    error=error or dict(dx=0.,dy=0.,dr=0.)
    radius=40+error['dr']+(np.arange(rows)-(rows-1)/2)*pitch
    theta=2*np.pi*np.arange(COLS)/COLS
    x=error['dx']+radius[:,None]*np.cos(theta);y=error['dy']+radius[:,None]*np.sin(theta)
    out=map_coordinates(array,[(y+(array.shape[0]-1)*.025)/.05,(x+(array.shape[1]-1)*.025)/.05],order=order,mode='constant',cval=np.nan)
    if not np.isfinite(out).all():raise ValueError('Extraction outside valid native support')
    return out

class Engine(oldg.Engine):
    def __init__(self,seed,amplitude,profile='raised'):
        # Explicit supported fallback, not an extrapolated or bypassed B fit.
        old.se.ScanEngine.__init__(self,image_size=old.N,pixel_size_mm=.05,
            render_model_mode='formula_only',emission_b_mode='lambertian',
            render_view_mode='top_surface',weld_layout=old.WeldLayoutRuntime(mode='circular_weld',circle_radius_mm=40.),
            acquisition_model='raster_v2',acquisition_profile=oldg.baseline.profile_for(seed))
        self.sample_seed=seed;self.amplitude=amplitude;self.quadrature_cache={}
        self.profile=profile;self.geometry_cache=None

    def volume(self,**kw):
        if self.geometry_cache is None:
            n=kw['image_size'];p=.05;axis=(np.arange(n)-(n-1)/2)*p
            yy,xx=np.meshgrid(axis,axis,indexing='ij')
            self.u=np.hypot(xx,yy)-40;self.theta=np.arctan2(yy,xx)
            self.geometry_cache=geo.height(self.u,self.theta,self.sample_seed,self.amplitude,self.profile)
            self.axis=axis
        h=self.geometry_cache.copy();shape=(1,*h.shape);m=kw['material']
        return VoxelVolumeState(material_z=np.full(shape,m.z,'f4'),yield_modifier=np.ones(shape,'f4'),
            void_fraction=np.zeros(shape,'f4'),relief_mm=h[None],defect_mask=np.zeros(shape,bool),
            fusion_mask=(np.abs(self.u)<=1.4)[None],x_mm=self.axis,y_mm=self.axis,z_mm=np.array([0.]),
            spacing_mm=(.05,.05,.05),origin_mm=(float(self.axis[0]),float(self.axis[0]),0.),
            metadata=dict(pixel_size_mm=.05,nominal_pixel_size_mm=.05,surface_plane_index=0,
                geometry=geo.specification(),representation='surface arrays plus analytic solid/void geometry',
                weld_layout=dict(mode='circular_weld',circle_radius_mm=40.),depth_extent_mm=7.))

    def inject(self,volume,plan,rng,**kw):
        self.pore_records=[];self.visible_native=[]
        for i,item in enumerate(plan):
            p=item['params'];delta,mask,record=geo.surface_effect(self.u,self.theta,p,.05**2)
            # Openings form their geometric union using the lowest floor. Bulges
            # add prescribed displaced volumes; interaction is explicitly empirical.
            if p['effect']=='open':volume.relief_mm[0]=np.minimum(volume.relief_mm[0],self.geometry_cache+delta)
            else:volume.relief_mm[0]+=delta
            self.pore_records.append(dict(instance_id=i,**p,**record))
            if mask.any():
                self.visible_native.append(mask)
                volume.defect_mask[0]|=mask
                volume.defect_channels.append(dict(type=item['type'],params=p,category_id=i+1,mask=mask[None].astype('u1')))

    def _build_geometry_response_maps(self,section_relief_mm,void_fraction_map,defects):
        # Keep response independent of defect labels. Surface geometry alone
        # carries the new defect signal; no hidden occupancy-based darkening.
        shape=section_relief_mm.shape
        return {k:np.ones(shape,'f4') for k in ('openness_scale_map','depth_scale_map','yield_scale_map','blur_scale_map','topo_gain_map')},[]

    def _apply_adaptive_quadrant_blur(self,quadrants,*,sigma_px,blur_scale_map):
        reach=np.sqrt(2)*(int(4*sigma_px+.5)+2)*.05
        if 4.8+reach>=PAD:raise ValueError('Insufficient native annular halo')
        return old.se.ScanEngine._apply_adaptive_quadrant_blur(self,quadrants,sigma_px=sigma_px,blur_scale_map=blur_scale_map)

def render(engine,sample,full=False,retain_native=False,uncached=False):
    def quadrature(**kw):
        engine.visibility=VisibilityQuadrature(engine.last_relief,sample.detector,
            radial_step=.05 if full else .1,angular_stride=2 if full else 4,ray_step=.05 if full else .1,
            base=None if full or uncached else getattr(engine,'base_visibility',None))
        out=engine.visibility.compute(**kw)
        if not sample.defects and not full and not uncached:engine.base_visibility=engine.visibility
        return out
    with patch.object(old.se,'build_voxel_volume',engine.volume),patch.object(old.se,'apply_defect_bodies',engine.inject),patch.object(oldg.baseline,'PAD_MM',PAD),patch.object(old.se,'compute_quadrant_signals',quadrature):
        result=engine.run(sample)
    gain=float(np.interp(sample.beam.focusing_current_norm,[0,1],[1.10,.92]))
    quiet=oldg.baseline.form_channels(engine.acquisition_quiet_quadrants).topo*gain
    raw=np.stack([extract(getattr(result.channels,k)) for k in ('topo','tx','ty')]).astype('f4')
    mask=extract(result.defect_mask,order=0).astype('u1')
    # Current acquisition profiles have zero motion. Fail if this changes rather
    # than accidentally saving unwarped instance masks.
    assert engine.acquisition_profile.velocity_x_mm_s==engine.acquisition_profile.velocity_y_mm_s==0
    instances=np.stack([extract(m.astype('f4'),order=0) for m in engine.visible_native]).astype('u1') if engine.visible_native else np.zeros((0,ROWS,COLS),'u1')
    assert np.array_equal(mask,instances.any(0))
    meta=dict(geometry=geo.specification(),profile=engine.profile,amplitude=engine.amplitude,pore_records=engine.pore_records,
        acquisition=result.metadata.get('acquisition',result.metadata.get('acquisition_model',{})),
        renderer_metadata=result.metadata,signal='local-surface empirical BSE angular response; no electron tracking',
        material_contrast='uniform 304SS; no artificial fusion atomic-number offset',
        visibility_limit='Nonlocal ray horizons; parallel detector directions and constant local working-distance approximation',
        visibility_resolution=dict(radial_mm=.05 if full else .1,angular_stride=2 if full else 4,ray_step_mm=.05 if full else .1),
        physical_validation=False)
    payload=dict(raw=raw,quiet=extract(quiet).astype('f4'),mask=mask,instances=instances,
                 height=extract(engine.last_relief).astype('f4'))
    meta['renderer_metadata']['legacy_geometry_controls_unused']=True
    meta['renderer_metadata']['weld_geometry_mode']='raised_crown' if engine.profile=='raised' else 'paired_recessed_control'
    meta['renderer_metadata']['ebw_reinforcement_mm']=3. if engine.profile=='raised' else 0.
    meta['renderer_metadata']['ebw_width_mm']=2.8
    meta['renderer_metadata']['fusion_penetration_depth_mm']=7.
    if retain_native:
        axis=(np.arange(old.N)-(old.N-1)/2)*.05
        valid=abs(np.hypot(axis[:,None],axis[None,:])-40)<4.9
        payload['native']=np.where(valid,result.channels.topo,np.nan).astype('f4')
    return payload,meta
