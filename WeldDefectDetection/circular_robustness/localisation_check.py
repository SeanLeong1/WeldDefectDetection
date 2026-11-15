"""Quantify the extra polar-resampling approximation against native extraction."""
from dataclasses import replace
import numpy as np
from scipy.ndimage import map_coordinates
from . import campaign as c,assessment as a

class Capture(c.Engine):
    def run(self,sample):
        result=super().run(sample);self.native=result.channels.topo.copy();return result

def run():
    c.prepare();cfg=c.g.old.load_config_bundle(c.ROOT/'DataSimulation/config');seed=c.BASE
    sample,_=c.g.baseline.c.sampled(cfg,seed,0);engine=Capture(seed,0.);plans=c.plans(seed,0);rows=[]
    angles=2*np.pi*np.arange(5024)/5024
    errors=[c.PERTURBATIONS[0],next(e for e in c.PERTURBATIONS if e['name']=='dx_+0.50'),next(e for e in c.PERTURBATIONS if e['name']=='dr_-0.50'),c.PERTURBATIONS[-1]]
    for variant in ('clean','single_keyhole','single_intrinsic'):
        raw,quiet,mask,instances,metadata=c.render(engine,replace(sample,defects=plans[variant],randomize_defect_location=True))
        with np.load(c.OUT/f'datasets/robust-000-{variant}.npz') as saved:assert np.array_equal(raw,saved['raw'])
        for error in errors:
            radius=40+error['dr']+(np.arange(96)-47.5)*.05
            x=error['dx']+radius[:,None]*np.cos(angles);y=error['dy']+radius[:,None]*np.sin(angles)
            direct=map_coordinates(engine.native,[(y+(c.g.old.N-1)*.025)/.05,(x+(c.g.old.N-1)*.025)/.05],order=1,mode='nearest')
            approximate=a.input_ring(raw,error)[0];difference=approximate-direct
            rows.append(dict(variant=variant,condition=error['name'],relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(direct),1e-12)),
                mean_absolute_error=float(abs(difference).mean()),max_absolute_error=float(abs(difference).max())))
    c.write(c.OUT/'investigation/localisation_resampling.json',dict(rows=rows,exact_nominal_reproduction=True,
        scope='Fixed first background, three variants, four conditions. Compares the saved-annulus interpolation used in the stress sweep with direct native Cartesian extraction. It quantifies an approximation, not model or physical validation.'))
    print(rows,flush=True)

if __name__=='__main__':run()
