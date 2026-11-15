"""Trace one declared pilot sample through the actual renderer and CV methods."""
from dataclasses import replace
import numpy as np
import torch
from . import campaign as c,rendering as r,learning as learn


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from box_development.methods import aligned_reference
    from circular_study import median,metrics
    torch.set_num_threads(4);frozen=learn.verify_freeze()
    cfg=r.old.load_config_bundle(c.ROOT/'DataSimulation/config');seed=c.BASES['pilot']
    sample,_=r.oldg.baseline.c.sampled(cfg,seed,0)
    sample=replace(sample,defects=c.plans(seed,0)['bulge']);engine=r.Engine(seed,0.)
    arrays,meta=r.render(engine,sample)
    norm=c.read(c.OUT/'normalization.json');normalized=learn.normalize(arrays['raw'][0],norm)
    y=64;x=16
    cut=lambda a:learn.tile(a,y,x)
    target=cut(normalized);context=np.stack([learn.tile(normalized[y-32:y+32],32,x+k) for k in (-256,-192,-128,-64,64,128,192,256)])
    reference=aligned_reference(target,context)
    residual,_,_=median.residual(normalized[y-32:y+32],x,'neighbours8')
    calibration=frozen['median']['spec']['calibration'][1]
    mscore=abs(residual-np.asarray(calibration['centre']))/np.asarray(calibration['scale'])
    model=learn.model_for(frozen['cnn'])
    with torch.inference_mode():cscore=torch.sigmoid(model(torch.from_numpy(target[None,None])))[0,0].numpy()
    normals=r.old.se.ScanEngine._normals_from_heightmap(engine.last_relief,.05)
    inclination=np.degrees(np.arccos(np.clip(r.extract(normals[...,2]),0,1)))
    quadrants=np.stack([r.extract(engine.acquisition_quiet_quadrants[...,i]) for i in range(4)])
    fullscores=dict(cnn=learn.predict_image(normalized,model),median=learn.median_image(normalized,frozen['median']['spec']))
    predicted={}
    for method,score in fullscores.items():
        op=frozen[method]['selection']['best'];predicted[method]=metrics.binary(score,op['threshold'],op['min_area'],op['closing'])[0]
    # Scored image has 16 fewer leading rows than full extracted arrays.
    pcrop=lambda a:learn.tile(a,y-16,x)
    panels=[('Height (mm)',cut(arrays['height']),'viridis'),('Inclination (degrees)',cut(inclination),'magma')]
    panels += [(f'Quiet quadrant {name}',cut(quadrants[i]),'gray') for i,name in enumerate('ABCD')]
    panels += [('Quiet TOPO',cut(arrays['quiet']),'gray'),('Acquired TOPO',cut(arrays['raw'][0]),'gray'),
        ('Train-normalized TOPO',target,'gray'),('Surface truth',cut(arrays['mask']),'gray'),
        ('CNN score',cscore,'magma'),('CNN final regions',pcrop(predicted['cnn']),'gray'),
        ('Median reference',reference,'gray'),('Smoothed residual',residual,'coolwarm'),
        ('Median score',mscore,'magma'),('Median final regions',pcrop(predicted['median']),'gray')]
    fig,axes=plt.subplots(4,4,figsize=(14,13),layout='constrained')
    for ax,(title,value,cmap) in zip(axes.ravel(),panels):
        im=ax.imshow(value,origin='lower',cmap=cmap);ax.set_title(title,fontsize=10);ax.set_xticks([0,32,63]);ax.set_yticks([0,32,63]);fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle('One pilot bulge: geometry, detector signals and both CV paths\n64 x 64 tile, 0.05 mm radial pitch; only central 32 x 32 scores contribute to full-ring stitching',fontsize=13)
    folder=c.ROOT/'docs/figures/raised_crown_results_20261005';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'traced_sample.png',dpi=145);plt.close(fig)
    np.savez_compressed(c.OUT/'traced_sample.npz',**{f'panel_{i}':v for i,(_,v,_) in enumerate(panels)},context=context)
    c.write(c.OUT/'traced_sample.json',dict(seed=seed,kind='bulge',tile=dict(radial_centre=y,angular_centre=x,size=64,retained=[16,48]),
        normalization=norm,metadata=meta,panel_titles=[a[0] for a in panels],
        note='Pilot teaching example, not an additional confirmation specimen. Truth is displayed for teaching only and is never passed to the inference models.'))
    print('Traced pilot sample saved',flush=True)
