"""Direct native extraction, replacing a diagnosed polar-resampling shortcut.

The original scene arrays are verified exactly on rerender. No model/threshold,
geometry, noise parameter, pore or population is changed.
"""
from dataclasses import replace
from pathlib import Path
import gc,time
import numpy as np
from scipy.ndimage import map_coordinates
from . import campaign as c

class Capture(c.Engine):
    def run(self,sample):
        result=super().run(sample);self.native=result.channels.topo.copy();return result

def extract(native,error):
    theta=2*np.pi*np.arange(5024)/5024
    radius=40+error['dr']+(np.arange(96)-47.5)*.05
    x=error['dx']+radius[:,None]*np.cos(theta);y=error['dy']+radius[:,None]*np.sin(theta)
    out=map_coordinates(native,[(y+(native.shape[0]-1)*.025)/.05,(x+(native.shape[1]-1)*.025)/.05],order=1,mode='nearest')
    if not np.isfinite(out).all():raise ValueError('Native image does not cover requested ring estimate')
    return out[None]

def prepare():
    c.prepare();folder=c.OUT/'native_inputs';folder.mkdir(exist_ok=True)
    sources=[Path(__file__),Path(__file__).with_name('assessment_native.py')]
    record=dict(schema='circular-robustness-native-extraction/v2',
        supersedes='Original saved-annulus interpolation in protocol.json; original source remains archived and no scores from it are promoted.',
        reason='Before stress scoring, native extraction check found approximately 7% relative input error for centre perturbations from a second polar interpolation.',
        change='Re-render the identical saved scenes, verify raw/quiet/mask/instances, retain native Cartesian TOPO and extract every estimated ring directly.',
        unchanged=['all 24 backgrounds','all seven pore layouts','all error conditions','all model weights','normalization','all thresholds and postprocessing','fixed true evaluation band'],
        original_protocol_sha256=c.sha(c.OUT/'protocol.json'),diagnostic_sha256=c.sha(c.OUT/'investigation/localisation_resampling.json'),
        model_freeze_sha256=c.sha(c.g.OUT/'frozen_models.json'),sources={str(p):c.sha(p) for p in sources})
    path=c.OUT/'protocol_native_v2.json'
    if path.exists():assert c.read(path)==record,'Changed active native extraction protocol'
    else:
        assert not list((c.OUT/'predictions').glob('*.json')),'Amendment must precede stress scores'
        c.write(path,record)
    return folder

def generate():
    folder=prepare();index=c.read(c.OUT/'index.json');cfg=c.g.old.load_config_bundle(c.ROOT/'DataSimulation/config');evidence=[]
    axis=(np.arange(c.g.old.N)-(c.g.old.N-1)/2)*.05
    valid=abs(np.hypot(axis[:,None],axis[None,:])-40)<=3.15
    for i in range(c.COUNT):
        seed=c.BASE+i;sample,_=c.g.baseline.c.sampled(cfg,seed,i);plans=c.plans(seed,i)
        engine=Capture(seed,(0.,.05,.1)[i%3])
        for variant in c.VARIANTS:
            sid=f'robust-{i:03d}-{variant}';path=folder/(sid+'.npz');receipt=folder/(sid+'.json')
            if receipt.exists():
                r=c.read(receipt);assert c.sha(path)==r['sha256'];evidence.append(r);continue
            raw,quiet,mask,instances,meta=c.render(engine,replace(sample,defects=plans[variant],randomize_defect_location=True))
            with np.load(c.OUT/'datasets'/(sid+'.npz')) as original:
                checks={k:bool(np.array_equal(v,original[k])) for k,v in dict(raw=raw,quiet=quiet,mask=mask,instances=instances).items()}
                assert all(checks.values()),(sid,checks)
            native=np.where(valid,engine.native,np.nan).astype('f4')
            np.testing.assert_array_equal(extract(native,c.PERTURBATIONS[0])[0],raw[0,16:112])
            for error in c.PERTURBATIONS:extract(native,error)
            np.savez_compressed(path,topo=native)
            r=dict(scene_id=sid,path=str(path),sha256=c.sha(path),original_scene_sha256=c.sha(c.OUT/'datasets'/(sid+'.npz')),exact_reproduction=checks)
            c.write(receipt,r);evidence.append(r);print('native',sid,flush=True)
            del raw,quiet,mask,instances,native;gc.collect()
        del engine;gc.collect()
    c.write(folder/'index.json',evidence)

if __name__=='__main__':generate()
