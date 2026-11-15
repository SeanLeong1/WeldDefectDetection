"""Solid fusion geometry and explicitly assumed pore/surface coupling, in mm."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import PchipInterpolator

@dataclass(frozen=True)
class Geometry:
    version: str = 'raised-crown-v1'
    radius_mm: float = 40.
    crown_height_mm: float = 3.
    crown_width_mm: float = 2.8
    penetration_depth_mm: float = 7.
    fusion_surface_width_mm: float = 2.8
    fusion_width_at_5mm: float = 1.2
    pixel_size_mm: float = .05
    electron_depth_cutoff_mm: None = None

SPEC=Geometry()

def height(u,theta,seed=0,amplitude=0.,profile='raised'):
    """One nominal peak; independently seeded smooth periodic macro-variation."""
    rng=np.random.default_rng(seed)
    phases=rng.uniform(0,2*np.pi,4)
    width=SPEC.crown_width_mm*(1+amplitude*np.sin(3*theta+phases[0]))
    high=SPEC.crown_height_mm*(1+amplitude*np.sin(5*theta+phases[1]))
    q=u/(width/2)
    envelope=np.maximum(1-q*q,0)**3
    crown=high*envelope
    if profile=='recessed':crown=-.18*envelope
    elif profile!='raised':raise ValueError(profile)
    # Small bounded texture, never a new macroscopic central trench.
    texture=.004*np.sin(71*theta+phases[2])*np.sin(9*u+phases[3])*envelope
    return (crown+texture).astype('f4')

def fusion_half_width(depth):
    depth=np.asarray(depth)
    out=PchipInterpolator([0,5,7],[1.4,.6,0])(np.clip(depth,0,7))
    return np.where((depth>=0)&(depth<=7),out,0.)

def fusion(u,z,surface):
    """Solid fused material; this mask never labels a defect or a void."""
    return ((z<=0)&(z>=-7)&(np.abs(u)<=fusion_half_width(-z))) | ((z>0)&(z<=surface))

def pore_field(u,theta,p):
    arc=SPEC.radius_mm*((theta-p['angle_rad']+np.pi)%(2*np.pi)-np.pi)
    local=u-p['radial_offset_mm'];diam=p['diameter_mm']
    rx=.5*diam;ry=(.8 if p['kind']=='keyhole_pore' else .5)*diam
    q=(local/rx)**2+(arc/ry)**2
    return q,rx,ry

def surface_effect(u,theta,p,area_weights):
    """No SEM depth rule. Enclosed pore and surface deformation are independent.

    Open bowl is the lower cap of an ellipsoid whose centre is above the
    healthy surface in surface-following coordinates. Bulge is a declared
    volume-preserving counterfactual, not a thermo-mechanical prediction.
    """
    q,rx,ry=pore_field(u,theta,p);rz=p['depth_mm']
    # arc = R*theta, so the Cartesian area Jacobian is (R+u)/R.
    # Symmetry about the pore's radial centre integrates u to its offset.
    volume=4*np.pi*rx*ry*rz/3*(1+p['radial_offset_mm']/SPEC.radius_mm)
    effect=p['effect'];support=q<1
    if effect=='hidden':delta=np.zeros_like(q);mask=np.zeros_like(support)
    elif effect=='open':
        cap=np.sqrt(np.maximum(1-q,0))
        delta=-rz*np.maximum(cap-.35,0)
        mask=cap>.35
    elif effect=='bulge':
        kernel=np.maximum(1-q,0)**3
        denominator=float(np.sum(kernel*area_weights))
        if denominator<=0:raise ValueError('Pore footprint not resolved')
        delta=kernel*(p.get('coupling_fraction',1.)*volume/denominator)
        mask=support
    else:raise ValueError(effect)
    return delta.astype('f4'),mask,dict(pore_volume_mm3=float(volume),
        integrated_surface_change_mm3=float(np.sum(delta*area_weights)),
        coupling='assumed_displacement_fraction' if effect=='bulge' else 'surface_cap' if effect=='open' else 'none',
        physical_validation=False)

def pore_contains(u,theta,z,healthy_surface,p):
    q,_,_=pore_field(u,theta,p)
    rz=p['depth_mm']
    centre=healthy_surface+(.35*rz if p['effect']=='open' else -rz-p.get('roof_mm',.15))
    return q+((z-centre)/rz)**2<=1

def specification():
    return dict(**asdict(SPEC),datum='original surface z=0',crown_width_status='assumed equal to fusion surface width',
        crown_height_status='user target, not measured calibration',plate_thickness_required=False,
        pore_bulge_status='controlled displacement hypothesis; not inferred from pore presence',
        depth_policy='No researched numerical electron depth implemented')
