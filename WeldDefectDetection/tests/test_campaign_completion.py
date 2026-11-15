import json
import hashlib
import numpy as np
import pytest
from src.evaluation.campaign_skill import skill_intervals
from src.algorithms.frozen_inference import prepare_raw,infer_frozen
from src.algorithms.classical.cnn_detector import TinyDefectSegNet,torch
from src.utils.image_preprocessing import PREPROCESSING_VERSION

def test_skill_rejects_always_positive_and_accepts_perfect():
    truth=np.array([0]*20+[1]*20,dtype=bool)
    counts=np.array([[0,0,0]]*20+[[10,0,0]]*20)
    baseline=np.array([[0,20,0]]*20+[[5,15,5]]*20)
    assert skill_intervals(counts,baseline,truth,truth,seed=1,iterations=100)['demonstrated_skill']
    assert not skill_intervals(counts,baseline,truth,np.ones(40,dtype=bool),seed=1,iterations=100)['demonstrated_skill']

def test_raw_preparation_matches_saved_u16_policy():
    norm={'raw_low':-1.,'raw_high':2.,'source_split':'train','tone_curve':'sem_sqrt'}
    raw=np.array([[-2,0,1,3]],dtype=np.float32)
    expected=((.14+.72*np.clip((raw+1)/3,0,1)**1.05)*65535).astype(np.uint16).astype(np.float32)/65535
    assert np.array_equal(prepare_raw(raw,norm),expected)

def test_static_inference_without_labels_and_normalization_tamper(tmp_path):
    norm=tmp_path/'norm.json'
    norm.write_text(json.dumps({'raw_low':0.,'raw_high':1.,'source_split':'train','tone_curve':'sem_sqrt'}))
    model=TinyDefectSegNet(base_channels=4)
    checkpoint=tmp_path/'model.pt'
    torch.save({'state_dict':model.state_dict(),'config':{'cnn':{'base_channels':4,'input_size':32}},'threshold':.5,'min_defect_area':2,
                'normalization_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),'preprocessing_version':PREPROCESSING_VERSION},checkpoint)
    input_path=tmp_path/'raw.npy'
    np.save(input_path,np.random.default_rng(1).random((32,32)).astype(np.float32))
    result=infer_frozen('static',input_path,checkpoint,norm)
    assert result['mask'].shape==(32,32) and np.isfinite(result['probability']).all()
    norm.write_text('{}')
    with pytest.raises(ValueError,match='normalization identity'):
        infer_frozen('static',input_path,checkpoint,norm)


def test_spatial_inference_disallows_target_overlap(tmp_path):
    from src.algorithms.temporal_predictor import BidirectionalSurfacePredictor
    norm=tmp_path/'norm.json'
    norm.write_text(json.dumps({'raw_low':0.,'raw_high':1.,'source_split':'train','tone_curve':'sem_sqrt'}))
    model=BidirectionalSurfacePredictor(context_channels=4,base_channels=4)
    path=tmp_path/'context.pt'
    torch.save({'state_dict':model.state_dict(),'base_channels':4,'context_radius':2,'stride_px':32,
                'context_preparation':'non_overlapping_native_coordinates','residual_threshold':.1,'min_area':2,
                'normalization_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),'preprocessing_version':PREPROCESSING_VERSION},path)
    inputs=tmp_path/'inputs.npz'
    rng=np.random.default_rng(2)
    context=rng.random((4,32,32)).astype(np.float32)
    target=rng.random((32,32)).astype(np.float32)
    np.savez(inputs,context=context,target=target,context_offsets_px=[-64,-32,32,64])
    result=infer_frozen('spatial',inputs,path,norm)
    assert result['mask'].shape==(32,32) and np.isfinite(result['reconstruction']).all()
    np.savez(inputs,context=context,target=target,context_offsets_px=[-64,-16,32,64])
    with pytest.raises(ValueError,match='disjoint crop'):
        infer_frozen('spatial',inputs,path,norm)
