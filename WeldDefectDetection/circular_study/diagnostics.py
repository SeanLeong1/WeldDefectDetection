"""Component convergence, assumptions, and native/ROI equivalence diagnostics."""
from dataclasses import replace
import gc
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from . import generation as g,geometry as geo


def component(sample,pitch,angle,mode='recessed_seam',b=.920694853227,smooth=True):
    # Six-mm Cartesian patch around a physical ring point, same latent surface at all pitches.
    n=round(6.4/pitch);axis=(np.arange(n)-(n-1)/2)*pitch
    xx=geo.RADIUS*np.cos(angle)+axis[None,:];yy=geo.RADIUS*np.sin(angle)+axis[:,None]
    h=geo.surface(np.hypot(xx,yy)-geo.RADIUS,np.arctan2(yy,xx),seed=2230410000,
        depth=.14,crown=.012,width=1.1,haz=1.,focus=.35,speed=800.,power=18.,amplitude=.1,mode=mode,smooth=smooth)
    normals=g.old.se.ScanEngine._normals_from_heightmap(h,pitch)
    q=g.old.chunked_quadrants(normals=normals,detector=sample.detector,material=sample.material,
        material_z_map=np.full(h.shape,sample.material.z,dtype='f4'),b_coeff=b)
    # Fixed 0.05 mm Gaussian PSF, then area integration at the common output pitch.
    # This is a component-level convergence test, not a full raster/acquisition replay.
    q=gaussian_filter(q,(.05/pitch,.05/pitch,0))
    factor=round(.05/pitch)
    q=q.reshape(128,factor,128,factor,4).mean(axis=(1,3))
    topo=g.baseline.form_channels(q).topo[32:96,32:96]
    height=h.reshape(128,factor,128,factor).mean(axis=(1,3))[32:96,32:96]
    return topo,height


def run():
    g.baseline.c.verify_sources(g.OUT)
    folder=g.OUT/'diagnostics';folder.mkdir(exist_ok=True)
    cfg=g.old.load_config_bundle(g.ROOT/'DataSimulation/config')
    sample,_=g.baseline.c.sampled(cfg,2230400100,0)
    rows=[];arrays={}
    for angle in (0.,np.pi/4,np.pi/2):
        outputs=[]
        for pitch in (.05,.025,.0125):
            topo,h=component(sample,pitch,angle);outputs.append(topo)
            arrays[f'a{round(np.degrees(angle))}_p{pitch}']=topo
        ref=outputs[-1]
        for pitch,value in zip((.05,.025,.0125),outputs):
            rows.append(dict(angle_deg=float(np.degrees(angle)),pitch_mm=pitch,
                relative_l2=float(np.linalg.norm(value-ref)/max(np.linalg.norm(ref),1e-12))))
        print('convergence',angle,rows[-3:],flush=True)
    base,h=component(sample,.025,0.)
    assumptions={}
    for name,kwargs in [('closed_fusion',dict(mode='closed_fusion')),('lambertian_B0',dict(b=0.)),('hard_transitions',dict(smooth=False))]:
        t,hh=component(sample,.025,0.,**kwargs)
        assumptions[name]=dict(relative_topo_l2=float(np.linalg.norm(t-base)/max(np.linalg.norm(base),1e-12)),
            height_min_mm=float(hh.min()),height_max_mm=float(hh.max()))
        arrays[name]=t;arrays[name+'_height']=hh
    arrays['recessed']=base;arrays['recessed_height']=h
    np.savez_compressed(folder/'component_arrays.npz',**arrays)
    g.write(folder/'component_checks.json',dict(convergence=rows,assumptions=assumptions,
        interpretation='Fixed geometry and 0.05 mm Gaussian PSF, native quadrature, common 0.05 mm area-integrated output. Relative to 0.0125 mm, not proof of absolute convergence or measured accuracy.',
        nominal_within_five_percent=all(r['relative_l2']<=.05 for r in rows if r['pitch_mm']==.05),
        scope='Surface/normals/quadrant component only; no pore/noise/acquisition calibration'))
    # Full Cartesian quiet images must agree inside the exported 96-row annulus.
    errors={};engines=[g.Engine(sample.seed,.1),g.Engine(sample.seed,.1)]
    for kind in g.baseline.c.KINDS:
        s=replace(sample,defects=g.plan(kind,sample.seed,0));values=[]
        for engine,full in zip(engines,(False,True)):
            result,raw,quiet,mask=g.render(engine,s,full=full)
            values.append((quiet,mask));del result,raw;gc.collect()
        err=float(np.max(abs(values[0][0]-values[1][0])))
        assert err<1e-7,(kind,err)
        assert np.array_equal(values[0][1],values[1][1])
        errors[kind]=dict(max_quiet_error=err,labels_identical=True)
        print('full versus ROI',kind,errors[kind],flush=True)
    del engines;gc.collect()
    g.write(folder/'roi_equivalence.json',dict(status='passed',checks=errors,
        noise_limit='Quiet agreement only. Exterior zero placeholder changes the Poisson random draw sequence, not the local distribution.'))


def visibility():
    rows=[]
    for split in ('train','val','test'):
        for r in g.read(g.OUT/f'datasets/{split}/index.json'):
            if r['source_class']=='clean':continue
            p=r['parameters']['defects'][0]['params']
            rows.append(dict(scene_id=r['scene_id'],split=split,source_class=r['source_class'],amplitude=r['amplitude'],
                diameter_mm=p['diameter_mm'],diameter_nominal_pixels=p['diameter_mm']/.05,
                angle_deg=float(np.degrees(p['angle_rad'])),radial_offset_mm=p['radial_offset_mm'],
                area_px=r['area_px'],signal_change_in_mask=r['signal_change_in_mask'],noise_rms=r['noise_rms'],
                signal_to_noise_proxy=r['signal_change_in_mask']/max(r['noise_rms'],1e-12)))
    g.write(g.OUT/'diagnostics/visibility.json',dict(rows=rows,
        definition='Mean absolute paired quiet pore-minus-clean contrast in geometric mask divided by whole-ring raw-minus-quiet RMS; a descriptive proxy, not calibrated physical SNR.',
        selection='No exclusion or contrast tuning based on these values'))


def network_contracts():
    import torch
    from . import training
    torch.set_num_threads(4)
    with np.load(g.OUT/'datasets/train/train-000-keyhole_pore.npz') as a:
        x=a['raw'];x=x/(np.max(abs(x))+1e-6);mask=a['mask'][16:80]
    data=dict(x=x[None],mask=mask[None],meta=[{}])
    centre=int(np.median(np.flatnonzero(mask.any(0))));results=[]
    for variant in ('fixed_negatives','larger_context','capacity24','topo_tx_ty'):
        config=training.configuration(variant);model=training.network(config)
        xx,yy=training.extract(data,[(0,centre)],config['input_size'],config['channels'])
        logits=model(torch.from_numpy(xx))
        if config['input_size']==96:logits=logits[:,:,16:80,16:80]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(yy));loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        results.append(dict(variant=variant,input=list(xx.shape),output=list(logits.shape),finite_gradients=True))
    g.write(g.OUT/'diagnostics/network_contract_checks.json',results)
    return results
