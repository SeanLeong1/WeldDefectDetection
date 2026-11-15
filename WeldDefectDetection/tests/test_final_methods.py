"""Regression against frozen scientific results and raw-image input contract."""
import json
import numpy as np
import pytest
from inspection.final_methods import FinalInspector,load_raw,example_path
from median_localisation_20261110 import OUT


@pytest.fixture(scope='module')
def inspector():
    return FinalInspector()


@pytest.mark.parametrize('scene',[
    'test-023-open_intrinsic-c64-r0', 'test-000-open_intrinsic-c1-r0',
    'test-012-open_intrinsic-c64-r0', 'test-000-clean-c4-r0'])
def test_matches_frozen_scores_and_boxes(inspector,scene):
    rows=json.loads((OUT/'datasets/test/index.json').read_text())
    i=next(i for i,r in enumerate(rows) if r['scene_id']==scene)
    result=inspector.infer(rows[i]['raw_path'])
    for method,name,score_name in [('cnn','cnn','cnn'),('median','combined','median_sigma1')]:
        expected=json.loads((OUT/f'confirmation/{name}.json').read_text())['rows'][i]
        score=np.load(OUT/f'confirmation/{score_name}.npy',mmap_mode='r')[i]
        np.testing.assert_allclose(result['scores'][method],score,atol=1e-6,rtol=0)
        assert sorted(result['boxes'][method])==sorted(expected['predicted_boxes'])


def test_npz_labels_do_not_affect_inference(inspector,tmp_path):
    raw=load_raw(example_path())
    p=tmp_path/'image.npz';np.savez(p,raw=raw,mask=np.ones(raw.shape,dtype='u1'))
    first=inspector.infer(p,('median',))
    np.savez(p,raw=raw,mask=np.zeros(raw.shape,dtype='u1'))
    second=inspector.infer(p,('median',))
    assert first['boxes']==second['boxes']
    np.testing.assert_array_equal(first['scores']['median'],second['scores']['median'])
    np.testing.assert_array_equal(load_raw(p),raw)


@pytest.mark.parametrize('shape,value', [((128,5024),1.),((160,5024),float('nan')),((160,5024),0.)])
def test_rejects_incompatible_images(tmp_path,shape,value):
    p=tmp_path/'bad.npy';np.save(p,np.full(shape,value,dtype='f4'))
    with pytest.raises(ValueError):load_raw(p)


def test_rejects_presentation_png(tmp_path):
    p=tmp_path/'display.png';p.write_bytes(b'not raw scientific data')
    with pytest.raises(ValueError,match='raw TOPO'):load_raw(p)
