import numpy as np
import pytest
import torch
from box_refinement.methods import residual_score,predicted_regions,select,augment_pair
from box_refinement.campaign import require,receipt
from box_comparison.core import sha,write


def test_continuous_weld_band_is_removed_but_local_defect_remains():
    image=np.zeros((64,64),np.float32)+.5;image[28:36]=.2
    assert np.max(residual_score(image,image,'thresholding',standardized=False))==0
    image[20:42,29:35]-=.1
    score=residual_score(image,image,'thresholding',standardized=False)
    assert score[25,32]>.08 and score[25,5]<1e-6


def test_hysteresis_keeps_only_weak_regions_with_strong_support():
    score=np.zeros((20,20),np.float32);score[2:6,2:6]=.4;score[3,3]=.8;score[12:16,12:16]=.4
    mask,boxes=predicted_regions(score,.6,1,low_ratio=.5)
    assert mask.sum()==16 and boxes==[[2,2,6,6]]


def test_padding_is_clipped_and_does_not_inflate_pixel_mask():
    score=np.zeros((8,8));score[0:2,0:2]=1
    base,_=predicted_regions(score,.5,1)
    mask,boxes=predicted_regions(score,.5,1,padding=4)
    assert np.array_equal(mask,base) and boxes==[[0,0,6,6]]


def test_augmentation_keeps_masks_and_reference_aligned():
    x=torch.zeros((12,1,8,8));x[:,:,2:4,1:3]=1
    target,reference,mask=augment_pair(x,x,x,torch.Generator().manual_seed(7))
    assert torch.equal(target,reference)
    assert torch.equal(target>.5,mask>0)


def test_selection_rejects_nonvalidation_and_bad_scores():
    with pytest.raises(ValueError):select({'meta':[{'split':'test'}]},np.zeros((1,8,8)),[.5])
    with pytest.raises(ValueError):predicted_regions(np.full((8,8),np.nan),.5,1)


def test_resumption_checks_files_referenced_by_inventory(tmp_path):
    data=tmp_path/'data.npy';data.write_bytes(b'original')
    inv=tmp_path/'inventory.json';write(inv,{str(data):sha(data)})
    receipt(tmp_path,'generate',[inv],[inv]);require(tmp_path,'generate')
    data.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Integrity'):require(tmp_path,'generate')


def test_missing_prerequisite_blocks_downstream(tmp_path):
    with pytest.raises(ValueError,match='prerequisite'):require(tmp_path,'freeze')


def test_simple_selection_records_features_without_duplicate_keywords(tmp_path,monkeypatch):
    from box_refinement import training
    def fake_select(*args,**kwargs):
        selected=dict(parameters={},metrics=dict(box_f1=.5,pixel_iou=.3,fp=1))
        return selected,[selected]
    monkeypatch.setattr(training,'select',fake_select)
    data=dict(target=np.zeros((1,8,8),np.float32),reference=np.zeros((1,8,8),np.float32))
    previous={a:dict(arm=a,parameters={},metrics=dict(box_f1=0.,pixel_iou=0.,fp=5)) for a in ('thresholding','median')}
    result=training.fit_simple(data,tmp_path,previous)
    assert result['thresholding']['implementation']=='refined'
    assert result['thresholding']['feature']==dict(sigma=1.,standardized=False)
