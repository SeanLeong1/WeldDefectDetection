"""Independent integrity and metric replay checks for the new confirmation."""
from pathlib import Path
from datetime import datetime
import numpy as np
import median_exploration_20261115 as ex

def main():
    frozen=ex.verify(True);rows=ex.read(ex.OUT/'datasets/test/index.json')
    assert len(rows)==ex.N_FRESH*18
    ids=[r['scene_id'] for r in rows];assert len(set(ids))==len(ids)
    backgrounds={r['background_id'] for r in rows};assert len(backgrounds)==ex.N_FRESH
    old_seeds=set()
    for split in ('train','val','test'):old_seeds.update(r['seed'] for r in ex.read(ex.OLD/f'datasets/{split}/index.json'))
    assert not {r['seed'] for r in rows}&old_seeds
    freeze_time=datetime.fromisoformat(frozen['frozen_utc']).timestamp();geometries={}
    for i,r in enumerate(rows):
        p=Path(r['raw_path']);assert p.stat().st_mtime>=freeze_time
        assert ex.sha(p)==r['raw_sha256']
        a=np.load(p,mmap_mode='r');assert a.shape==(160,5024) and np.isfinite(a).all()
        geometries[r['geometry_path']]=r['geometry_sha256']
        if (i+1)%144==0:print('AUDITED RAW',i+1,flush=True)
    for path,h in geometries.items():assert ex.sha(path)==h
    methods=ex.read(ex.OUT/'confirmation_median.json');methods['cnn']=ex.read(ex.OUT/'confirmation_cnn.json')
    checked=0
    for name,result in methods.items():
        assert [r['scene_id'] for r in result['rows']]==ids
        for r in result['rows']:
            reference=ex.metrics.match(r['predicted_boxes'],r['truth_boxes'],5024)
            assert all(reference[k]==r[k] for k in ('tp','fp','fn')),(name,r['scene_id'])
            checked+=1
        assert ex.summary(result['rows'])['by_count']==result['by_count']
    physics=ex.OUT/'renderer_sensitivity'
    physics_count=0;changed={'B0971':0,'bounded_yield':0};original_replays=0
    original_lookup={r['scene_id']:r for r in rows}
    for p in physics.glob('*/index.json'):
        for r in ex.read(p):
            assert ex.sha(r['raw_path'])==r['raw_sha256'];physics_count+=1
            a=np.load(r['raw_path']);quiet=np.load(r['quiet_path'])
            assert a.shape==quiet.shape==(160,5024) and np.isfinite(a).all() and np.isfinite(quiet).all()
            sid=r['scene_id'][len(r['mode'])+1:];ref=original_lookup[sid]
            assert r['geometry_path']==ref['geometry_path']
            equal=np.array_equal(a,np.load(ref['raw_path']))
            if r['mode']=='original':assert equal;original_replays+=1
            else:changed[r['mode']]+=int(not equal)
    assert physics_count==8*3*3
    assert original_replays==24 and all(v>0 for v in changed.values())
    sources=ex.read(ex.OUT/'protocol.json')['sources']
    # Source checks include the historical generator, calibration and CNN weights.
    ex.verify(True)
    audit=dict(status='passed',independent_backgrounds=len(backgrounds),scenes=len(rows),geometry_files=len(geometries),source_dependencies=len(sources),
               fresh_created_after_freeze=True,train_validation_original_test_seed_disjoint=True,raw_and_geometry_hashes_verified=True,
               prediction_records_replayed_with_original_metric=checked,renderer_sensitivity_scenes_hashed=physics_count,
               renderer_original_bit_exact_replays=original_replays,renderer_changed_scenes=changed,
               cnn_retrained=False,physical_validation=False)
    ex.write(ex.OUT/'final_audit.json',audit);ex.write(ex.REPORT/'audit.json',audit);print(audit)

if __name__=='__main__':main()
