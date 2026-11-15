"""Native circular views of the fixed first confirmation background.

Re-render the identical saved samples and verify exported annuli first. The
Cartesian exterior is invalid and is explicitly NaN, never a training image.
"""
from dataclasses import replace
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import generation as g


def run():
    g.baseline.c.verify_sources(g.OUT)
    cfg=g.old.load_config_bundle(g.ROOT/'DataSimulation/config');seed=g.BASES['test']
    sample,_=g.baseline.c.sampled(cfg,seed,0);engine=g.Engine(seed,0.)
    folder=g.OUT/'diagnostics/native_circular_previews';folder.mkdir(parents=True,exist_ok=True)
    norm=g.read(g.OUT/'normalization.json');axis=(np.arange(g.old.N)-(g.old.N-1)/2)*.05
    valid=abs(np.hypot(axis[:,None],axis[None,:])-40)<=1.6
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained');checks={}
    cmap=plt.colormaps['gray'].copy();cmap.set_bad('#e8e8e8')
    for ax,kind in zip(axes,g.baseline.c.KINDS):
        sid=f'test-000-{kind}';s=replace(sample,defects=g.plan(kind,seed,0),randomize_defect_location=True)
        result,raw,quiet,mask=g.render(engine,s)
        with np.load(g.OUT/f'datasets/test/{sid}.npz') as saved:
            checks[kind]={k:bool(np.array_equal(v,saved[k])) for k,v in [('raw',raw),('quiet',quiet),('mask',mask)]}
            assert all(checks[kind].values()),checks[kind]
        native=np.where(valid,result.channels.topo,np.nan)
        np.savez_compressed(folder/f'{sid}.npz',topo=native,valid=valid,mask=result.defect_mask)
        display=np.sqrt(np.clip((native-norm['low'])/(norm['high']-norm['low']),0,1))
        ax.imshow(display,cmap=cmap,vmin=0,vmax=1,origin='lower',extent=[-48.025,48.025,-48.025,48.025])
        ax.set_title(kind);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        del result,raw,quiet,mask;gc.collect()
    fig.suptitle('Native 80 mm circular weld views; grey exterior is outside the valid inspection annulus')
    figure=g.ROOT/'docs/figures/circular_study_20261004/native_circles.png';figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure,dpi=170);plt.close(fig)
    g.write(folder/'verification.json',dict(exact_saved_annulus_reproduction=checks,background='test-000',
        independent_new_background=False,exterior='Invalid, stored as NaN',cartesian_pixel_size_mm=.05))
    print(figure,flush=True)


if __name__=='__main__':run()
