"""Controlled surface-angle response, separate from the primary CV experiment.

Rotating a uniform surface normal relative to a fixed beam and detector is a
local orientation test. It does not implement independent beam deflection or
global image projection of a rigid tilted specimen.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'DataSimulation'))
from bse_renderer.config_loader import load_config_bundle
from bse_renderer.models import DetectorRuntime
from bse_renderer.renderer.quadrant_detector import compute_quadrant_signals
from bse_renderer.renderer.image_formation import form_channels
from box_comparison.core import write,sha


def measure(root):
    cfg=load_config_bundle(REPO/'DataSimulation/config');material=cfg.material.presets['304SS']
    detector=DetectorRuntime(height_mm=12.,radius_mm=12.,inner_radius_mm=4.,outer_radius_mm=16.,
        quadrants_deg={'A':0.,'B':90.,'C':180.,'D':270.},sensitivity={k:1. for k in 'ABCD'},psi0_rad=np.pi/4)
    b=.9206948532270666;rows=[]
    for tilt in [-10.,-5.,-2.,-1.,0.,1.,2.,5.,10.]:
        a=np.radians(tilt);n=np.array([np.sin(a),0.,np.cos(a)]).reshape(1,1,3)
        q=compute_quadrant_signals(n,detector,material,np.full((1,1),material.z),b_coeff=b)
        channels=form_channels(q)
        rows.append(dict(surface_rotation_deg=tilt,local_incidence_deg=abs(tilt),quadrants=q[0,0].tolist(),
             topo=float(channels.topo[0,0]),sum_signal=float(channels.compo[0,0]),
             normalized_tx=float(channels.tx[0,0]/channels.compo[0,0]),
             normalized_ty=float(channels.ty[0,0]/channels.compo[0,0]),
             normalized_topo=float(channels.topo[0,0]/channels.compo[0,0])))
    zero=next(r for r in rows if r['surface_rotation_deg']==0)
    assert np.ptp(zero['quadrants'])<1e-7
    for angle in [1.,2.,5.,10.]:
        left=next(r for r in rows if r['surface_rotation_deg']==-angle)
        right=next(r for r in rows if r['surface_rotation_deg']==angle)
        np.testing.assert_allclose(left['normalized_topo'],right['normalized_topo'],atol=1e-7)
    result=dict(kind='synthetic local surface-orientation diagnostic',B=b,beam_direction=[0,0,-1],
        detector=dict(height_mm=12.,inner_radius_mm=4.,outer_radius_mm=16.,equal_sensitivity=True),
        rows=rows,independent_beam_deflection=False,projection_or_occlusion_model=False,
        source_script_sha256=sha(__file__),physical_validation=False)
    write(root/'beam_angle_response.json',result)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    angles=[r['surface_rotation_deg'] for r in rows]
    for i,name in enumerate('ABCD'):
        axes[0].plot(angles,[r['quadrants'][i]/r['sum_signal'] for r in rows],'o-',label=name)
    axes[0].set(xlabel='Surface rotation (degrees)',ylabel='Quadrant fraction of total signal');axes[0].legend()
    axes[1].plot(angles,[r['normalized_topo'] for r in rows],'o-',color='#1764a2')
    axes[1].set(xlabel='Surface rotation (degrees)',ylabel='TOPO / total signal')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Fixed beam and detector; uniform surface normal rotated; no defects or noise')
    fig.savefig(root/'beam_angle_response.png',dpi=180);plt.close(fig)
    print('Verified flat-surface symmetry and opposite-tilt response.')


if __name__=='__main__':
    measure(REPO/'DataSimulation/data/output/variability_campaign_20260926_v1')
