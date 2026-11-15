import numpy as np
import pytest
from box_comparison import inference
from box_comparison.core import write,sha


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    normal=tmp_path/'normal.json'
    write(normal,dict(raw_low=0,raw_high=1,tone_curve='sem_sqrt'))
    model=dict(arm='median',parameters=dict(threshold=.1,min_area=3,opening=1))
    write(tmp_path/'frozen_models.json',dict(normalization=dict(path=str(normal),sha256=sha(normal)),models={'median':model}))
    monkeypatch.setattr(inference,'check_protocol',lambda root:None)
    monkeypatch.setattr(inference,'require',lambda root,stage:None)
    return tmp_path


def save_input(path,**overrides):
    values=dict(target_raw=np.zeros((64,64),np.float32),context_raw=np.zeros((4,64,64),np.float32),
        energy_kev=60,pixel_size_mm=.05,layout='straight_seam',rotation_deg=0,
        context_offsets_xy=np.array([[-128,0],[-64,0],[64,0],[128,0]]))
    values.update(overrides);np.savez(path,**values)


@pytest.mark.parametrize('change',[dict(energy_kev=70),dict(layout='circular'),dict(rotation_deg=10),
    dict(pixel_size_mm=.1),dict(context_offsets_xy=np.array([[-64,0],[0,0],[64,0],[128,0]])),
    dict(mask=np.zeros((64,64))),dict(context_raw=np.zeros((3,64,64)))])
def test_unsupported_or_labelled_inputs_rejected(frozen,change):
    save_input(frozen/'input.npz',**change)
    with pytest.raises(ValueError):inference.infer(frozen,frozen/'input.npz',frozen/'out.json','median')


def test_public_inference_contains_boxes_without_probabilities(frozen):
    target=np.zeros((64,64),np.float32);target[20:25,30:35]=1
    save_input(frozen/'input.npz',target_raw=target)
    result=inference.infer(frozen,frozen/'input.npz',frozen/'out.json','median')
    assert result['boxes_xyxy']==[[30,20,35,25]]
    assert 'probability' not in str(result) and 'confidence' not in str(result)
