"""Reuse unaffected pre-training images after a documented volume-only repair.

No bulge image is reused. No trained model or opened confirmation set exists in
the source version. The changed volume factor does not enter open/clean images.
"""
from pathlib import Path
import copy,shutil
import numpy as np
from . import campaign as c

def run():
    source=c.OUT.with_name('raised_crown_20261005_v4')
    assert not (source/'frozen_models.json').exists()
    assert c.read(c.OUT/'pilot/gates.json')['numerical_gate']=='passed'
    for kind in ('clean','open_keyhole','open_intrinsic','hidden'):
        with np.load(source/f'pilot/{kind}.npz') as a,np.load(c.OUT/f'pilot/{kind}.npz') as b:
            for key in ('raw','quiet','mask','instances','height'):np.testing.assert_array_equal(a[key],b[key])
    reused=[];target=c.OUT/'datasets/train';target.mkdir(parents=True,exist_ok=True)
    rows=c.read(source/'datasets/train/index.json')
    for row in rows:
        if row['source_class']=='bulge':continue
        assert row['source_class'] in ('clean','open_keyhole','open_intrinsic')
        oldpath=Path(row['npz_path']);assert c.sha(oldpath)==row['npz_sha256']
        newpath=target/oldpath.name;shutil.copy2(oldpath,newpath)
        item=copy.deepcopy(row);item['npz_path']=str(newpath)
        for pore in item['metadata']['pore_records']:
            pore['pore_volume_mm3']*=1+pore['radial_offset_mm']/40.
        item['metadata']['generation_reuse']=dict(source=str(oldpath),sha256=row['npz_sha256'],
            original_renderer_archive_sha256=c.sha(source/'sources.zip'),
            reason='Only nominal pore-volume/Jacobian metadata changed; no surface/image change for this class',
            new_renderer_archive_sha256=c.sha(c.OUT/'sources.zip'))
        c.write(newpath.with_suffix('.json'),item);reused.append(dict(scene_id=item['scene_id'],source=str(oldpath),target=str(newpath),sha256=c.sha(newpath)))
    c.write(c.OUT/'development_reuse.json',dict(source_version=source.name,target_version=c.OUT.name,
        checkpoint_preexisted=False,confirmation_preexisted=False,representative_all_array_equality=True,
        bulges_reused=False,records=reused,source=str(Path(__file__)),source_sha256=c.sha(__file__)))
    print('Reused',len(reused),'unaffected development scenes; all displacement scenes require corrected generation',flush=True)

if __name__=='__main__':run()
