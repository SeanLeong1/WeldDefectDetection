"""Optional quiet signed-channel diagnostic captured without changing the dataset."""
from unittest.mock import patch
import numpy as np
from box_comparison.core import sha
from . import generation as g


def generate_test():
    frozen=g.read(g.OUT/'frozen_models.json')
    if frozen['cnn']['config']['channels']!=3:
        g.generate('test');return
    original=g.render;folder=g.OUT/'diagnostics/quiet_signed';folder.mkdir(parents=True,exist_ok=True)
    def capture(engine,sample,*args,**kwargs):
        result=original(engine,sample,*args,**kwargs)
        channels=g.baseline.form_channels(engine.acquisition_quiet_quadrants)
        gain=float(np.interp(sample.beam.focusing_current_norm,[0,1],[1.10,.92]))
        index=sample.seed-g.BASES['test'];kind=sample.defects[0]['type'] if sample.defects else 'clean'
        path=folder/f'test-{index:03d}-{kind}.npz'
        np.savez_compressed(path,signed=np.stack([g.unwrap(channels.tx*gain),g.unwrap(channels.ty*gain)]).astype('f4'))
        g.write(path.with_suffix('.json'),dict(sha256=sha(path),definition='Quiet Tx/Ty from the same pre-noise quadrants as retained quiet TOPO, with identical focus gain and annular sampling'))
        return result
    with patch.object(g,'render',capture):g.generate('test')


def add_signed(data):
    norm=g.read(g.OUT/'normalization.json');signed=[]
    for row in data['meta']:
        path=g.OUT/'diagnostics/quiet_signed'/f'{row["scene_id"]}.npz'
        if not path.exists():return None
        assert sha(path)==g.read(path.with_suffix('.json'))['sha256']
        with np.load(path) as a:signed.append(np.clip(a['signed']/norm['signed_scale'],-1,1))
    data['x']=np.concatenate([data['x'],np.stack(signed)],axis=1)
    return data
