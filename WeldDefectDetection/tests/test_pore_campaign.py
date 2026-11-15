from pathlib import Path
import numpy as np
import pytest
from pore_campaign.generation import pore_parameters,DATASET_CLASSES
from pore_campaign.audit import quiet_data
from pore_campaign.campaign import bootstrap,freeze
from box_refinement.campaign import receipt,require
from box_comparison.core import write


def test_pore_policy_rejects_nonpore_and_spans_declared_grid():
    rng=np.random.default_rng(14)
    with pytest.raises(ValueError,match='Only clean'):pore_parameters('crack_solidification',rng)
    for kind,diameter,depth in [('keyhole_pore',(.5,1.2),(.25,.6)),('intrinsic_pore',(.5,1.),(.25,.65))]:
        cases=[pore_parameters(kind,rng,i) for i in range(9)]
        assert all(len(c)==1 and c[0]['type']==kind for c in cases)
        pairs={(c[0]['params']['diameter_mm'],c[0]['params']['depth_mm']) for c in cases}
        assert len(pairs)==9
        assert (min(p[0] for p in pairs),max(p[0] for p in pairs))==diameter
        assert (min(p[1] for p in pairs),max(p[1] for p in pairs))==depth


def test_quiet_pair_reuses_frozen_preprocessing_and_target_excluded_context(tmp_path):
    np.savez(tmp_path/'sequence-0000.npz',sample_ids=np.array(['a','b','c','d','e']),
             raw=np.stack([np.full((64,64),v,dtype=np.float32) for v in (.1,.2,.9,.3,.4)]))
    data=dict(meta=[dict(sample_id='c',context_sample_ids=['a','b','d','e'])],mask=np.ones((1,64,64),np.uint8))
    out=quiet_data(data,tmp_path,dict(raw_low=0.,raw_high=1.,tone_curve='sem_sqrt'))
    assert np.array_equal(out['mask'],data['mask'])
    assert np.all(out['target']>out['context'].max())
    assert np.array_equal(out['reference'],np.median(out['context'],axis=1))


def test_dataset_receipt_detects_modified_raw_artifact(tmp_path):
    raw=tmp_path/'raw.npy';np.save(raw,np.ones((3,3)))
    from box_comparison.core import sha
    manifest=tmp_path/'inventory.json';write(manifest,{str(raw):sha(raw)})
    receipt(tmp_path,'generated_development',[manifest],[manifest]);require(tmp_path,'generated_development')
    np.save(raw,np.zeros((3,3)))
    with pytest.raises(ValueError,match='Integrity mismatch'):require(tmp_path,'generated_development')


def test_freeze_cannot_bypass_preparation(tmp_path):
    with pytest.raises(ValueError,match='Missing prerequisite'):freeze(tmp_path)


def test_bootstrap_rejects_unpaired_target_populations():
    rows={'cnn':[dict(sequence_id='s',window_id='a')], 'median':[dict(sequence_id='s',window_id='b')]}
    with pytest.raises(ValueError,match='identical unique targets'):bootstrap(rows,'cnn',10)


def test_pore_inference_rejects_labels_and_wrong_domain(tmp_path):
    from pore_campaign.inference import load_input
    f=tmp_path/'input.npz'
    np.savez(f,mask=np.zeros((64,64)))
    with pytest.raises(ValueError,match='Labels'):load_input(f,{},'cnn')
    np.savez(f,energy_kev=45.,pixel_size_mm=.05)
    with pytest.raises(ValueError,match='physical domain'):load_input(f,{},'cnn')
