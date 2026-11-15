"""Periodic procedural geometry in physical coordinates, not a weld-process solver.

Texture is sampled once on a fixed physical cylinder, independently of the output
grid. Finite transition widths are explicit assumptions, not measured parameters.
"""
from functools import lru_cache
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.interpolate import CubicSpline

RADIUS = 40.
PITCH = .05
COLS = 5024
ARC_PITCH = 2*np.pi*RADIUS/COLS


def smooth_gate(x, low, high, transition):
    def step(t):
        t = np.clip(t, 0, 1)
        return t*t*t*(10+t*(-15+6*t))
    return step((x-low)/transition)*step((high-x)/transition)


@lru_cache(maxsize=4)
def fields(seed, correlation_mm):
    rng = np.random.default_rng(seed)
    line = gaussian_filter1d(rng.normal(size=COLS), max(2., correlation_mm/ARC_PITCH), mode='wrap')
    line = (line-line.mean())/max(line.std(), 1e-9)
    texture = gaussian_filter(rng.normal(size=(161,COLS)), (1.2,1.5), mode=('reflect','wrap'))
    texture /= texture.std()
    knots = np.linspace(0,2*np.pi,41)
    variable=[]
    for _ in range(2):
        v=rng.uniform(-1,1,41);v[-1]=v[0]
        f=CubicSpline(knots,v,bc_type='periodic')
        variable.append(f(np.linspace(0,2*np.pi,COLS,endpoint=False)))
    variable=np.array(variable);variable/=np.max(abs(variable),axis=1,keepdims=True)
    return line.astype('f4'),texture.astype('f4'),variable.astype('f4')


def surface(radial_offset, angle, *, seed, depth, crown, width, haz, focus,
            speed, power, offset=0., amplitude=0., mode='recessed_seam', smooth=True):
    """Evaluate the same continuous cylindrical surrogate at arbitrary coordinates."""
    u,theta=np.broadcast_arrays(np.asarray(radial_offset),np.asarray(angle))
    q=(theta%(2*np.pi))*COLS/(2*np.pi)
    wavelength=(speed/60)/(50+30*(power/10))
    line,texture,var=fields(int(seed),float(max(.10,.9*wavelength)))
    def periodic(values):
        return map_coordinates(values,[q],order=3,mode='grid-wrap',prefilter=True,output=np.float64)
    u=(u-offset)/(1+amplitude*periodic(var[0]))
    d=depth*(1+amplitude*periodic(var[1]))
    w=max(.15,.5*width);n=abs(u)/w
    key=np.exp(-.5*(u/max(.04,.15*w))**2)
    if mode=='recessed_seam':
        trace=-d*np.exp(-.5*(u/(.65*w))**2)*key
        cap=crown*np.maximum(1-n*n,0)
        gate=smooth_gate(n,.90,1.05,.04) if smooth else ((n>=.9)&(n<=1.05))
        boundary=-.05*np.maximum(d,.03)*gate
        rough=.025*np.maximum(d,.03)*(1.35-.45*focus)*periodic(line)*np.exp(-1.5*n*n)
        noiseamp=.005
    elif mode=='closed_fusion':
        c=np.clip(max(crown,.2*depth),.012,.12)
        cap=c*np.maximum(1-(n/1.18)**2,0)**1.2
        trace=-.09*d*np.exp(-.5*(u/(.22*w))**2)
        boundary=.025*c*np.exp(-.5*((n-1.02)/.11)**2)
        rough=.45*.025*np.maximum(d,.03)*(1.35-.45*focus)*periodic(line)*np.exp(-1.5*n*n)
        noiseamp=.002
    else:
        raise ValueError(mode)
    # Explicit periodic angular extension; radial clamp avoids wrapping opposite toes.
    tex=np.pad(texture,((0,0),(16,16)),mode='wrap')
    local=map_coordinates(tex,[np.clip(u/.05+80,0,160),q+16],order=3,mode='nearest',output=np.float64)
    h=abs(u)/max(w+.05,(width+haz)/2)
    gate=smooth_gate(h,.95,1.15,.05) if smooth else ((h>=.95)&(h<=1.15))
    return trace+cap+boundary+rough+noiseamp*local*gate+.006*d*local*np.exp(-.7*n*n)


def profile(*, width_px, height_px, pixel_size_mm, layout, seed, crown_height_mm,
            seam_depth_mm, bead_width_mm, haz_width_mm, beam_focus_quality,
            travel_speed_mm_min, beam_power_kw, asymmetry_offset_mm=0.,
            weld_geometry_mode='recessed_seam', amplitude=0.):
    x=(np.arange(width_px)-(width_px-1)/2)*pixel_size_mm
    y=(np.arange(height_px)-(height_px-1)/2)*pixel_size_mm
    return surface(np.hypot(y[:,None],x[None,:])-layout.circle_radius_mm,
        np.arctan2(y[:,None],x[None,:]),seed=seed,depth=seam_depth_mm,
        crown=crown_height_mm,width=bead_width_mm,haz=haz_width_mm,focus=beam_focus_quality,
        speed=travel_speed_mm_min,power=beam_power_kw,offset=asymmetry_offset_mm,
        amplitude=amplitude,mode=weld_geometry_mode)


def inject(volume, defect_plan, rng, **kwargs):
    """Native pore response constants, with controlled local radial/tangent coordinates."""
    from bse_renderer.geometry import voxel_volume as vv
    yy,xx=np.meshgrid(volume.y_mm,volume.x_mm,indexing='ij')
    for item in defect_plan:
        kind=item['type'];p=dict(item['params']);diam=p['diameter_mm'];depth=p['depth_mm']
        angle=float(p['angle_rad']);off=float(p['radial_offset_mm'])
        cx=(RADIUS+off)*np.cos(angle);cy=(RADIUS+off)*np.sin(angle)
        radial=(xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
        tangent=-(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
        z=vv._surface_z_reference_mm(volume)
        if kind=='keyhole_pore':
            main=vv._ellipse_profile_mm(radial,tangent,cx_mm=0.,cy_mm=0.,rx_mm=.38*diam,ry_mm=.78*diam,theta_rad=float(rng.normal(0,np.deg2rad(8))),power=1.)
            tail=vv._segment_profile_mm(radial,tangent,x0_mm=0.,y0_mm=-.12*diam,x1_mm=float(rng.normal(0,.1*diam)),y1_mm=.95*diam,width0_mm=.18*diam,width1_mm=.08*diam,power=1.1)
            xy=np.clip(.88*main+.45*tail,0,1)
            zp=vv._z_profile(volume.z_mm,center_mm=z,half_extent_mm=max(4*volume.spacing_mm[2],rng.uniform(.45,1.10),.60),power=1.2)
            response=dict(yield_scale=.76,void_fraction=.80,relief_delta_mm=-.045*max(depth,.08))
        elif kind=='intrinsic_pore':
            xy=vv._ellipse_profile_mm(radial,tangent,cx_mm=0.,cy_mm=0.,rx_mm=.42*diam,ry_mm=.36*diam,theta_rad=float(rng.uniform(-np.pi/2,np.pi/2)),power=1.4)
            zp=vv._z_profile(volume.z_mm,center_mm=z+float(rng.normal(0,.08*diam)),half_extent_mm=max(.24,.45*diam),power=1.5)
            response=dict(yield_scale=.86,void_fraction=.52,relief_delta_mm=-.012*max(depth,.05),z_scale=.985)
        else:
            raise ValueError('This experiment supports the two existing pore classes only.')
        mask=vv._apply_body_to_volume(volume,zp[:,None,None]*xy[None],**response)
        volume.defect_channels.append(dict(type=kind,params=p,category_id=vv.CATEGORY_IDS[kind],mask=mask.astype('u1')))
