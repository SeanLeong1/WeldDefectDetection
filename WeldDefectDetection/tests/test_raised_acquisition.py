"""Guard against leakage and changes to the matched-arm experiment."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'DataSimulation'),str(ROOT/'WeldDefectDetection')]
import numpy as np
import pytest
from raised_acquisition import campaign as c, learning as learn


def test_dwell_coverage_and_disjoint_backgrounds():
    seeds=[]
    for split,n in c.COUNTS.items():
        bins=[]
        for i in range(n):
            value,b= c.dwell(split,i)
            assert c.EDGES[b]<=value<=c.EDGES[b+1]
            bins.append(b); seeds.append(c.BASES[split]+i)
        assert np.bincount(bins).tolist()==[n//4]*4
    assert len(seeds)==len(set(seeds))
    assert c.dwell('train',0)[0]==.1
    assert c.dwell('train',23)[0]==20.


def test_matched_arms_preserve_positive_draws_and_batch_budget():
    positive=np.array([[1,64,x] for x in range(20)])
    negative=np.array([[2,64,x] for x in range(20)])
    hard=np.array([[3,64,x] for x in range(20)])
    a=learn.epoch_indices(positive,negative,None,123,8)
    b=learn.epoch_indices(positive,negative,hard,123,8)
    assert a.shape==b.shape==(1200,3)
    np.testing.assert_array_equal(a[a[:,0]==1],b[b[:,0]==1])
    assert sum(b[:,0]==1)==600 and sum(b[:,0]==2)==300 and sum(b[:,0]==3)==300


def test_mining_rejects_nontraining_clean_images():
    data=dict(meta=[dict(source_class='clean',split='test')],fullmask=np.zeros((1,160,5024),bool))
    with pytest.raises(AssertionError): learn.clean_candidates(data,5)
    data['meta'][0]['split']='train'
    candidates=learn.clean_candidates(data,5)
    assert candidates.shape==(168,3) and len(np.unique(candidates,axis=0))==168
    data['fullmask'][0,70,2]=True
    with pytest.raises(AssertionError): learn.clean_candidates(data,5)


def test_paired_intervals_keep_identical_models_at_zero():
    from raised_acquisition.assessment import paired_difference
    rows=[dict(scene_id=str(i),background_id=str(i),dwell_bin=i%4,source_class='clean',
        tp=0,fp=i%2,fn=0,intersection=0,union=1,predicted_regions=i%2) for i in range(8)]
    result=paired_difference(rows,rows,repeats=20)
    assert all(x['difference']==0 and x['interval_95']==[0.,0.] for x in result.values())


def test_current_inspector_rejects_incompatible_geometry_and_renderer(tmp_path):
    import json
    from raised_acquisition.gui import infer
    path=tmp_path/'wrong.npz'
    path.with_suffix('.json').write_text(json.dumps(dict(metadata=dict(geometry={}))))
    with pytest.raises(ValueError,match='Geometry'): infer(path)
    metadata=dict(geometry=c.prior.geo.specification(),profile='recessed')
    path.with_suffix('.json').write_text(json.dumps(dict(metadata=metadata)))
    with pytest.raises(ValueError,match='renderer contract'): infer(path)


def test_gui_generation_rejects_out_of_contract_dwell_and_unknown_defect():
    from raised_acquisition.gui import generate
    for dwell in (-1,0,20.1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='dwell range'): generate(123,'clean',dwell)
    with pytest.raises(ValueError,match='Unknown pore'): generate(123,'unfamiliar_profile',1.)
