"""Resolution sensitivity of the smallest native pore bodies at a fixed surface.

Component diagnostic only: fixed 0.05 mm PSF, unit detector gains, no raster
noise. Absolute material/yield response and the native two-plane pore occupancy
are retained. The diagnostic does not select or alter model training data.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from box_comparison.core import sha
from . import generation as g,geometry as geo


def component(sample,pitch,angle,kind):
    n=round(6.4/pitch);axis=(np.arange(n)-(n-1)/2)*pitch
    x=40*np.cos(angle)+axis;y=40*np.sin(angle)+axis
    yy,xx=np.meshgrid(y,x,indexing='ij');u=np.hypot(xx,yy)-40
    height=geo.surface(u,np.arctan2(yy,xx),seed=2230410000,depth=.14,crown=.012,width=1.1,haz=1.,focus=.35,speed=800.,power=18.,amplitude=.1)
    shape=(2,n,n);fusion=abs(u)<=.55
    volume=g.old.VoxelVolumeState(material_z=np.repeat(np.where(fusion,sample.material.z+.2,sample.material.z)[None],2,axis=0).astype('f4'),
        yield_modifier=np.ones(shape,'f4'),void_fraction=np.zeros(shape,'f4'),relief_mm=np.repeat(height[None],2,axis=0).astype('f4'),
        defect_mask=np.zeros(shape,bool),fusion_mask=np.repeat(fusion[None],2,axis=0),x_mm=x,y_mm=y,z_mm=np.array([-.08,0.]),
        spacing_mm=(pitch,pitch,.08),origin_mm=(float(x[0]),float(y[0]),-.08),metadata=dict(surface_plane_index=1,pixel_size_mm=pitch))
    if kind!='clean':
        geo.inject(volume,[dict(type=kind,params=dict(diameter_mm=.5,depth_mm=.25,angle_rad=angle,radial_offset_mm=0.))],np.random.default_rng(2230410100))
    section,_=g.old.se.ScanEngine._top_surface_section(volume,min_visible_area_px=1)
    normals=g.old.se.ScanEngine._normals_from_heightmap(section.relief_mm_map,pitch)
    localyield=np.clip(section.yield_modifier_map*(1-.22*section.void_fraction_map),.15,1.15)
    q=g.old.chunked_quadrants(normals=normals,detector=sample.detector,material=sample.material,
        material_z_map=section.material_z_map,yield_modifier_map=localyield,
        openness_map=np.clip(.88*section.void_fraction_map,0,1),b_coeff=.920694853227)
    q=gaussian_filter(q,(.05/pitch,.05/pitch,0));factor=round(.05/pitch)
    q=q.reshape(128,factor,128,factor,4).mean(axis=(1,3))
    return g.baseline.form_channels(q).topo[32:96,32:96],float(section.defect_mask.sum()*pitch*pitch)


def run():
    cfg=g.old.load_config_bundle(g.ROOT/'DataSimulation/config');sample,_=g.baseline.c.sampled(cfg,2230400100,0)
    rows=[];saved={}
    for angle in (0.,np.pi/4,np.pi/2):
        values={}
        for pitch in (.05,.025,.0125):
            values[pitch]={kind:component(sample,pitch,angle,kind) for kind in g.baseline.c.KINDS}
        for kind in g.baseline.c.KINDS[1:]:
            reference=values[.0125][kind][0]-values[.0125]['clean'][0]
            for pitch in (.05,.025,.0125):
                delta=values[pitch][kind][0]-values[pitch]['clean'][0]
                rows.append(dict(kind=kind,angle_deg=float(np.degrees(angle)),pitch_mm=pitch,
                    relative_delta_l2=float(np.linalg.norm(delta-reference)/max(np.linalg.norm(reference),1e-12)),
                    geometric_area_mm2=values[pitch][kind][1]))
                saved[f'{kind}_a{round(np.degrees(angle))}_p{pitch}']=delta
        print('pore convergence',np.degrees(angle),rows[-6:],flush=True)
    folder=g.OUT/'diagnostics';np.savez_compressed(folder/'pore_convergence_arrays.npz',**saved)
    g.write(folder/'pore_convergence.json',dict(rows=rows,diameter_mm=.5,depth_mm=.25,
        source_sha256=sha(__file__),geometry_sha256=sha(geo.__file__),
        definition='Relative L2 of paired defect-minus-clean quiet TOPO versus 0.0125 mm. Common 0.05 mm Gaussian PSF and area-integrated output. Native pore body and response constants, fixed surface planes, unit gains, no acquisition noise.',
        scope='Smallest pore parameters at three orientations; not a complete end-to-end convergence or physical validation claim'))


if __name__=='__main__':run()
