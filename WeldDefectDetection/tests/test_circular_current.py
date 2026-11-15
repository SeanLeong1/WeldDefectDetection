from pathlib import Path
import json
import numpy as np
import pytest
import torch
from inspection.engine import FrozenInspector,DEFAULT_CAMPAIGN,Options,load_image,run
from circular_robustness import campaign as c,assessment as a
from circular_study import generation as g,training,median,metrics

def test_direct_native_extraction_and_missing_context():
    from circular_robustness.native_inputs import extract
    axis=(np.arange(1921)-960)*.05
    field=np.broadcast_to(axis[None,:],(1921,1921)).astype('f4')
    theta=2*np.pi*np.arange(5024)/5024
    for error in c.PERTURBATIONS:
        expected=error['dx']+(40+error['dr']+(np.arange(96)-47.5)*.05)[:,None]*np.cos(theta)
        np.testing.assert_allclose(extract(field,error)[0],expected,atol=5e-6)
    missing=field.copy();missing[:]=np.nan
    with pytest.raises(ValueError,match='does not cover'):extract(missing,c.PERTURBATIONS[0])

def test_error_sampler_uses_physical_centre_and_radius():
    theta=2*np.pi*np.arange(5024)/5024
    radius=40+(np.arange(128)-63.5)*.05
    field=(radius[:,None]*np.cos(theta)).astype('f4')
    raw=np.stack([field]*3);error=dict(dx=.25,dy=-.25,dr=.25)
    result=a.input_ring(raw,error)[0]
    expected=.25+(40.25+(np.arange(96)-47.5)*.05)[:,None]*np.cos(theta)
    np.testing.assert_allclose(result,expected,atol=2e-5)
    for e in c.PERTURBATIONS:assert a.input_ring(raw,e).shape==(3,96,5024)

def test_restore_counts_uncovered_truth_as_missed():
    score=np.ones((64,5024),'f4');error=dict(dx=0.,dy=0.,dr=.5)
    restored,coverage=a.restore(score,error)
    assert not coverage[:9].any() and not restored[:9].any()
    truth=np.zeros_like(score,'u1');truth[1:4,20:25]=1
    meta=dict(scene_id='a',background_id='a',variant='single_keyhole',source_class='keyhole_pore',amplitude=0.)
    row=a.row_result(restored,truth,truth[None],meta,dict(threshold=.5,min_area=1,closing=1),coverage)
    assert row['fn']==1 and row['truth_pixels_outside_inspection_band']==15

def test_overlap_keeps_instance_denominator():
    one=np.zeros((64,5024),'u1');two=one.copy();one[25:35,100:115]=1;two[25:35,110:125]=1
    union=one|two;meta=dict(scene_id='a',background_id='a',variant='pair_overlap',source_class='multiple',amplitude=0.)
    row=a.row_result(union.astype('f4'),union,np.stack([one,two]),meta,dict(threshold=.5,min_area=1,closing=1),np.ones_like(union,bool))
    assert row['tp']==1 and row['instance']['tp']==1 and row['instance']['fn']==1

def test_current_adapter_replays_frozen_ring_scores():
    torch.set_num_threads(4);inspector=FrozenInspector();frozen=g.read(g.OUT/'frozen_models.json')
    path=DEFAULT_CAMPAIGN/'datasets/test/test-000-keyhole_pore.npz'
    item=load_image(path,Options(mode='raw',pixel_size_mm=.05),inspector.norm)
    data=dict(x=item['annulus'][None],mask=np.zeros((1,64,5024),'u1'),meta=[dict(scene_id='example')])
    model=training.load_model(frozen['cnn'])
    np.testing.assert_array_equal(inspector.score(item,'cnn'),training.predict(data,model,frozen['cnn']['config'])[0])
    np.testing.assert_array_equal(inspector.score(item,'median'),median.scores(data,frozen['median']['spec'])[0])
    for method in ('cnn','median'):
        _,_,boxes=inspector.predict(item,method)
        name='cnn_capacity24_100482003' if method=='cnn' else 'median_neighbours8'
        expected=next(r for r in g.read(g.OUT/f'predictions/{name}_best.json') if r['scene_id']=='test-000-keyhole_pore')
        assert boxes==expected['predicted_boxes']

def test_circular_import_rejects_wrong_contract_and_ignores_labels(tmp_path):
    inspector=FrozenInspector();path=DEFAULT_CAMPAIGN/'datasets/test/test-000-keyhole_pore.npz'
    with pytest.raises(ValueError,match='complete raw'):load_image(path,Options(),inspector.norm)
    with np.load(path) as original:raw=original['raw']
    p=tmp_path/'example.npz';np.savez_compressed(p,raw=raw,pixel_size_mm=.05,mask=np.zeros((2,2)))
    first=load_image(p,Options(mode='raw'),inspector.norm)
    np.savez_compressed(p,raw=raw,pixel_size_mm=.05,mask=np.ones((2,2)))
    np.testing.assert_array_equal(first['annulus'],load_image(p,Options(mode='raw'),inspector.norm)['annulus'])
    with pytest.raises(ValueError,match='disagrees'):load_image(p,Options(mode='raw',pixel_size_mm=.1),inspector.norm)
    p.with_name(p.name+'.json').write_text(json.dumps(dict(radius_mm=45,pixel_size_mm=.05)))
    with pytest.raises(ValueError,match='radius'):load_image(p,Options(mode='raw'),inspector.norm)

def test_circular_portable_checkpoint_and_coordinate_contract(tmp_path):
    import shutil
    data=g.read(DEFAULT_CAMPAIGN/'frozen_models.json');folder=tmp_path/'model';(folder/'training').mkdir(parents=True)
    for name in ('frozen_models.json','normalization.json'):shutil.copy2(DEFAULT_CAMPAIGN/name,folder/name)
    checkpoint=Path(data['cnn']['checkpoint']).name
    shutil.copy2(DEFAULT_CAMPAIGN/'training'/checkpoint,folder/'training'/checkpoint)
    inspector=FrozenInspector(folder);inspector._network('cnn')
    assert len(inspector.source_hashes())==4
    coordinates=inspector.coordinates([[5018,30,5030,35]])
    assert coordinates['boxes_angle_radial'][0][2]>360
    assert coordinates['boxes_mm'][0][1]==pytest.approx(-.1)
    assert inspector.coordinates([[0,0,1,1]])['boxes_angle_radial'][0][0]==pytest.approx(-.5*360/5024)
