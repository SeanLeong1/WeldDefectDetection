import numpy as np
import pytest
from box_comparison.core import (median_reference, model_input, truth_box, match_boxes,
    threshold_mask, morphology_score, select_threshold, selection_key)
from box_comparison.campaign import verify_files, shuffled_references
from box_comparison.inference import normalize_raw
from bse_renderer.utils import apply_sem_tone_curve


def test_median_resists_one_corrupt_neighbour():
    context=np.ones((4,8,8),np.float32)*.2
    context[0]=1
    assert np.allclose(median_reference(context),.2)
    assert np.mean(context)>.2
    with pytest.raises(ValueError):median_reference(context[:3])


def test_hybrid_uses_target_reference_and_absolute_residual():
    x=model_input(np.ones((4,4)),np.full((4,4),.2),True)
    assert x.shape==(3,4,4)
    assert np.allclose(x[:,0,0],[1,.2,.8])
    assert model_input(x[0],x[1],False).shape==(1,4,4)


def test_region_truth_preserves_disconnected_defects_and_single_pixel():
    mask=np.zeros((10,10),np.uint8);mask[1,1]=1;mask[5:7,5:8]=1
    assert truth_box(mask)==[[1,1,2,2],[5,5,8,7]]


def test_box_matching_is_one_to_one():
    result=match_boxes([[0,0,5,5],[0,0,5,5]],[[0,0,5,5]])
    assert (result['tp'],result['fp'],result['fn'])==(1,1,0)
    assert match_boxes([[0,0,2,2]],[[2,0,4,2]])['tp']==0


def test_thin_crack_retained_and_bright_dark_supported():
    image=np.ones((64,64),np.float32)*.5
    image[20,4:60]=.05;image[40,4:60]=.95
    score=morphology_score(image,9,'both')
    mask=threshold_mask(score,.2,3,1)
    assert mask[20,4:60].all() and mask[40,4:60].all()


def test_selection_cannot_read_confirmation():
    with pytest.raises(ValueError,match='validation'):
        select_threshold({'meta':[{'split':'test'}]},np.zeros((1,3,3)),[.5])
    assert selection_key(dict(box_f1=.5,pixel_iou=.1,fp=100))>selection_key(dict(box_f1=.4,pixel_iou=.9,fp=0))


def test_hash_tampering_rejected(tmp_path):
    from box_comparison.core import sha
    p=tmp_path/'file';p.write_text('a');expected={str(p):sha(p)}
    verify_files(expected);p.write_text('b')
    with pytest.raises(ValueError,match='Integrity'):verify_files(expected)


def test_shuffled_control_excludes_own_sequence_without_labels():
    data={'meta':[{'sequence_id':str(s)} for s in range(4) for _ in range(5)],
          'reference':np.repeat(np.arange(4),5)[:,None,None]}
    out=shuffled_references(data)
    assert np.all(out['reference']!=data['reference'])


def test_raw_preprocessing_matches_saved_uint16_mapping():
    raw=np.array([[-1,.1,.3,2]],np.float32)
    normalization=dict(raw_low=0,raw_high=.4,tone_curve='sem_sqrt')
    expected=(apply_sem_tone_curve(np.clip(raw/.4,0,1))*65535).astype(np.uint16).astype(np.float32)/65535
    assert np.array_equal(normalize_raw(raw,normalization),expected)
    with pytest.raises(ValueError):normalize_raw(np.array([[np.nan]]),normalization)
