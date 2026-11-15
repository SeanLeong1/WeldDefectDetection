import numpy as np
import pytest
from box_development.methods import aligned_reference,fit_scale,rectangle_targets,evaluate_new,make_network
from box_development.data import concatenate,shuffle_context
from box_development.inference import load_input


def test_profile_alignment_preserves_local_target_anomaly():
    context=np.zeros((4,64,64),np.float32)+.3
    context+=np.arange(4,dtype=np.float32)[:,None,None]*.02
    target=np.zeros((64,64),np.float32)+.5;target[20:25,20:25]=.8
    ref=aligned_reference(target,context)
    assert np.allclose(ref,.5,atol=1e-5)
    assert target[22,22]-ref[22,22]>.29


def test_scale_rejects_validation_and_has_floor():
    data=dict(target=np.zeros((12,64,64),np.float32),context=np.zeros((12,4,64,64),np.float32),
        mask=np.zeros((12,64,64),np.uint8),meta=[dict(split='train',context_defective_count=0) for _ in range(12)])
    scale,_=fit_scale(data,'median');assert np.all(scale==np.float32(.003))
    data['meta'][0]['split']='val'
    with pytest.raises(ValueError,match='training data'):fit_scale(data,'median')


def test_rectangle_supervision_does_not_modify_truth():
    masks=np.zeros((1,64,64),np.uint8);masks[0,10:20,10]=1;masks[0,19,10:20]=1
    original=masks.copy();rect=rectangle_targets(masks)
    assert rect.sum()==100 and np.array_equal(masks,original)


def test_rectangle_boxes_do_not_inflate_pixel_iou():
    mask=np.zeros((1,64,64),np.uint8);mask[0,10:20,10:20]=1
    data=dict(mask=mask,meta=[dict(truth_boxes=[[10,10,20,20]])])
    scores=np.zeros((1,2,64,64),np.float32);scores[0,0,12:15,12:15]=1;scores[0,1,10:20,10:20]=1
    rows=evaluate_new(data,scores,dict(threshold=.5,min_area=1,closing=1,low_ratio=1.,padding=0),1)
    assert rows[0]['tp']==1 and rows[0]['pixel_tp']==9 and rows[0]['pixel_fn']==91


def test_two_output_network_backpropagates():
    import torch
    torch.set_num_threads(2)
    model=make_network(1,True);x=torch.randn(2,1,32,32);out=model(x)
    assert out.shape==(2,2,32,32)
    out.square().mean().backward()
    assert torch.isfinite(model.head.weight.grad).all()


def test_training_merge_rejects_held_out_and_duplicate_ids():
    part=dict(target=np.zeros((1,2,2)),meta=[dict(window_id='a',split='train')])
    with pytest.raises(ValueError,match='Duplicate'):concatenate([part,part])
    part['meta'][0]['split']='val'
    with pytest.raises(ValueError,match='held-out'):concatenate([part])


def test_shuffled_context_is_cross_sequence_and_keeps_target():
    context=np.stack([np.full((4,2,2),i) for i in range(3)])
    data=dict(context=context,target=np.arange(3),reference=np.median(context,axis=1),meta=[dict(sequence_id=str(i)) for i in range(3)])
    shuffled=shuffle_context(data)
    assert np.all(shuffled['context'][:,0,0,0]!=np.arange(3))
    assert np.array_equal(shuffled['target'],data['target'])


def test_inference_rejects_labels_and_overlap(tmp_path):
    values=dict(target_raw=np.ones((64,64)),context_raw=np.ones((4,64,64)),energy_kev=60.,pixel_size_mm=.05,
        layout='straight_seam',rotation_deg=0.,context_offsets_xy=np.array([[-128,0],[-64,0],[64,0],[128,0]]))
    norm=dict(raw_low=0.,raw_high=2.,tone_curve='sem_sqrt')
    p=tmp_path/'input.npz';np.savez(p,**values,mask=np.zeros((64,64)))
    with pytest.raises(ValueError,match='Labels'):load_input(p,norm,'median')
    values['context_offsets_xy'][0]=[0,0];np.savez(p,**values)
    with pytest.raises(ValueError,match='non-overlapping'):load_input(p,norm,'median')


def test_inference_rejects_unsupported_energy(tmp_path):
    p=tmp_path/'input.npz';np.savez(p,energy_kev=70.,pixel_size_mm=.05)
    with pytest.raises(ValueError,match='domain'):load_input(p,{},'cnn')


def test_receipt_checks_referenced_dataset_content(tmp_path):
    from box_refinement.campaign import receipt,require
    from box_comparison.core import write,sha
    raw=tmp_path/'raw.npy';raw.write_bytes(b'original');inventory=tmp_path/'inventory.json'
    write(inventory,{str(raw):sha(raw)})
    receipt(tmp_path,'expand',[inventory],[inventory]);require(tmp_path,'expand')
    raw.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Integrity'):require(tmp_path,'expand')
