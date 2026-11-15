"""Meaningful numerical gates; physical limitations remain explicitly separate."""
from dataclasses import replace
import gc
import numpy as np
from scipy.ndimage import map_coordinates
from . import geometry as geo,rendering as r

def geometry_checks():
    np.testing.assert_allclose(2*geo.fusion_half_width([0,5,7]),[2.8,1.2,0])
    u=np.linspace(-2,2,8001);h=geo.height(u,np.zeros_like(u),1,0)
    assert abs(float(h.max())-3)<.005
    np.testing.assert_allclose(geo.height(u,0,1,.1),geo.height(u,2*np.pi,1,.1),atol=1e-6)
    # Numerically integrate a prescribed displaced volume on a fine local grid.
    axis=np.linspace(-1.5,1.5,601);x,y=np.meshgrid(axis,axis)
    p=dict(kind='intrinsic_pore',effect='bulge',diameter_mm=.7,depth_mm=.25,angle_rad=0.,radial_offset_mm=0.,coupling_fraction=1.)
    delta,mask,m=geo.surface_effect(x,y/40,p,(axis[1]-axis[0])**2)
    assert abs(m['integrated_surface_change_mm3']/m['pore_volume_mm3']-1)<1e-6
    shifted=dict(p,radial_offset_mm=1.05)
    uu=x+1.05;weights=(axis[1]-axis[0])**2*(1+uu/40)
    _,_,vm=geo.surface_effect(uu,y/40,shifted,weights)
    qq,_,_=geo.pore_field(uu,y/40,shifted)
    integrated_void=float(np.sum(2*shifted['depth_mm']*np.sqrt(np.maximum(1-qq,0))*weights))
    assert abs(integrated_void/vm['pore_volume_mm3']-1)<.001
    hidden,hm,_=geo.surface_effect(x,y/40,dict(p,effect='hidden'),.005**2)
    assert not hidden.any() and not hm.any()
    assert geo.pore_contains(0.,0.,2.6,3.,dict(p,effect='hidden',roof_mm=.15))
    assert geo.fusion(0.,-6.,3.) and not geo.fusion(2.,-5.,0.)
    # The nominal bump's derivative converges under physical resampling.
    fine=np.gradient(geo.height(u,np.zeros_like(u),1,0),u)
    coarse_u=np.arange(-2,2.001,.025);coarse=np.gradient(geo.height(coarse_u,0,1,0),.025)
    derivative_error=float(np.max(abs(np.interp(coarse_u,u,fine)-coarse)))
    assert derivative_error<.04
    return dict(anchors=True,periodic=True,hidden_surface_unchanged=True,volume_balance=True,
        max_slope=float(max(abs(fine))),max_inclination_deg=float(np.degrees(np.arctan(max(abs(fine))))),
        derivative_error_25um=derivative_error,no_electron_depth_cutoff=geo.SPEC.electron_depth_cutoff_mm is None)

def shadow_diagnostic(engine,sample):
    """Sparse geometric ray audit, not a calibrated detector correction.

    Tests the local-frame parallel directions assumed by the inherited detector.
    Positive local dot products can still be blocked by distant surface.
    """
    u,theta=np.meshgrid(np.linspace(-2.5,2.5,41),np.linspace(0,2*np.pi,24,endpoint=False))
    x=(40+u.ravel())*np.cos(theta.ravel());y=(40+u.ravel())*np.sin(theta.ravel())
    native=engine.last_relief;origin=(native.shape[0]-1)*.025
    z=map_coordinates(native,[(y+origin)/.05,(x+origin)/.05],order=1)
    normals=r.old.se.ScanEngine._normals_from_heightmap(native,.05)
    ns=np.stack([map_coordinates(normals[...,j],[(y+origin)/.05,(x+origin)/.05],order=1) for j in range(3)],1)
    blocked=0;outward=0
    detector=sample.detector
    for radius in (detector.inner_radius_mm,(detector.inner_radius_mm+detector.outer_radius_mm)/2,detector.outer_radius_mm):
        phi=np.arctan2(radius,detector.height_mm)
        for az in np.linspace(0,2*np.pi,16,endpoint=False):
            direction=np.array([np.sin(phi)*np.cos(az),np.sin(phi)*np.sin(az),np.cos(phi)])
            positive=ns@direction>0;hit=np.zeros(len(z),bool)
            for distance in np.arange(.1,12.01,.1):
                xx=x+distance*np.cos(az);yy=y+distance*np.sin(az)
                surf=map_coordinates(native,[(yy+origin)/.05,(xx+origin)/.05],order=1,mode='constant',cval=0)
                hit|=surf>z+distance/np.tan(phi)+.005
            outward+=int(positive.sum());blocked+=int((hit&positive).sum())
    return dict(sampled_outward_rays=outward,blocked_rays=blocked,fraction=blocked/max(outward,1),
        interpretation='Unweighted sampled-direction fraction, not detector signal error. No occlusion correction applied.',
        consequence='Synthetic local-response benchmark only; cannot claim validated steep-crown image realism.')

def run_pilot():
    from . import campaign as c
    c.verify_sources();folder=c.OUT/'pilot';folder.mkdir(exist_ok=True)
    if (folder/'gates.json').exists():return c.read(folder/'gates.json')
    checks=geometry_checks();cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config')
    sample,_=r.oldg.baseline.c.sampled(cfg,c.BASES['pilot'],0)
    engine=r.Engine(sample.seed,0.);samples={}
    for kind in ('clean','open_keyhole','open_intrinsic','bulge','hidden','pair_overlap'):
        payload,meta=r.render(engine,replace(sample,defects=c.plans(sample.seed,0)[kind]),retain_native=kind=='clean')
        np.savez_compressed(folder/(kind+'.npz'),**payload);c.write(folder/(kind+'.json'),meta)
        samples[kind]=dict(raw=payload['raw'].copy(),quiet=payload['quiet'].copy(),mask=payload['mask'].copy())
        print('pilot',kind,flush=True)
        del payload;gc.collect()
    assert np.array_equal(samples['clean']['raw'],samples['hidden']['raw'])
    direct,_=r.render(engine,replace(sample,defects=c.plans(sample.seed,0)['open_keyhole']),uncached=True)
    cache_error=float(np.max(abs(direct['quiet']-samples['open_keyhole']['quiet'])))
    assert cache_error<1e-6,cache_error
    # Refine the physical horizon grid and ray step independently of CV scores.
    previous=c.OUT.with_name('raised_crown_20261005_v3')/'pilot/gates.json'
    if previous.exists():
        old=c.read(previous);assert old['numerical_gate']=='passed'
        quiet_error=old['visibility_refinement_max_error'];convergence_rms=old['visibility_refinement_relative_rms']
        with np.load(previous.parent/'clean.npz') as a:np.testing.assert_array_equal(a['quiet'],samples['clean']['quiet'])
        refinement_provenance=dict(path=str(previous),sha256=c.sha(previous),clean_image_exact_match=True)
    else:
        full,_=r.render(engine,replace(sample,defects=[]),full=True)
        quiet_error=float(np.max(abs(full['quiet']-samples['clean']['quiet'])))
        convergence_rms=float(np.sqrt(np.mean((full['quiet']-samples['clean']['quiet'])**2))/max(np.sqrt(np.mean(full['quiet']**2)),1e-9))
        refinement_provenance=dict(recomputed=True)
    assert convergence_rms<.03,convergence_rms
    shadow=shadow_diagnostic(engine,sample)
    c.write(folder/'gates.json',dict(numerical_gate='passed',geometry=checks,visibility_refinement_max_error=quiet_error,visibility_refinement_relative_rms=convergence_rms,
        cache_noise_free_max_error=cache_error,visibility_refinement_provenance=refinement_provenance,
        hidden_clean_exact_match=True,nonlocal_occlusion=shadow,
        full_grid_noise_note='Compare noise-free signals when quadrature changes; Poisson realisations are not bitwise equivalent',
        physics_gate='Nonlocal geometric visibility included; uncalibrated angular/material response, parallel detector geometry and constant local working distance remain assumptions',
        emission_decision='Existing assumed B rejected above 60 degrees; supported explicit Lambertian fallback used for this new study',
        full_plate_thickness_needed=False,researched_depth_implemented=False))
    print('PILOT GATES PASSED (numerical), physics limitations recorded',flush=True)
