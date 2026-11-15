from pathlib import Path
import json
import threading

import numpy as np
from PIL import Image
import pytest

from inspection.engine import LEGACY_CAMPAIGN as DEFAULT_CAMPAIGN, FrozenInspector, Options, load_image, run


@pytest.fixture(scope='module')
def inspector():
    return FrozenInspector(DEFAULT_CAMPAIGN)


@pytest.mark.parametrize('name', ['clean_0.00', 'keyhole_pore_0.00', 'intrinsic_pore_0.00',
                                 'clean_0.20', 'keyhole_pore_0.20', 'intrinsic_pore_0.20'])
def test_frozen_inference_matches_recorded_boxes(inspector, name):
    item = load_image(DEFAULT_CAMPAIGN/f'inference_smoke/{name}.npz', Options(mode='raw'), inspector.norm)
    for arm in ('cnn', 'median', 'thresholding'):
        score, mask, boxes = inspector.predict(item, arm)
        expected = json.loads((DEFAULT_CAMPAIGN/f'inference_smoke/{name}_{arm}.json').read_text())
        assert boxes == expected['boxes_xyxy']
        assert score.shape == mask.shape == (64, 64)


def test_tiling_preserves_edges_without_resizing(inspector):
    rng = np.random.default_rng(47)
    target = rng.random((71, 139), dtype=np.float32)
    score = inspector.score(dict(target=target, context=None), 'cnn')
    assert score.shape == target.shape
    assert np.isfinite(score).all()
    patch = target[-64:, -64:]
    reference = inspector.score(dict(target=patch, context=None), 'cnn')
    # The far bottom-right pixel is covered only by the final patch.
    np.testing.assert_allclose(score[-1, -1], reference[-1, -1], atol=1e-7)


def test_display_16bit_preserves_code_values(tmp_path, inspector):
    p = tmp_path/'sample.tif'
    raw = np.arange(64*64, dtype=np.uint16).reshape(64, 64)*10
    Image.fromarray(raw).save(p)
    item = load_image(p, Options(), inspector.norm)
    np.testing.assert_allclose(item['target'], raw/65535., atol=1e-7)
    assert item['pixel_size_mm'] is None
    assert any('exploratory' in w.lower() for w in item['warnings'])


def test_raw_scale_conflict_rejected(inspector):
    p = DEFAULT_CAMPAIGN/'inference_smoke/clean_0.00.npz'
    with pytest.raises(ValueError, match='disagrees'):
        load_image(p, Options(mode='raw', pixel_size_mm=.001), inspector.norm)
    with pytest.raises(ValueError, match='Raw TOPO'):
        load_image(p, Options(), inspector.norm)


@pytest.mark.parametrize('array, message', [
    (np.full((64, 64), np.nan), 'finite'),
    (np.ones((64, 64)), 'no intensity'),
    (np.ones((3, 4, 5, 6)), 'grayscale'),
])
def test_invalid_arrays_rejected(tmp_path, inspector, array, message):
    p = tmp_path/'bad.npy'
    np.save(p, array)
    with pytest.raises(ValueError, match=message):
        load_image(p, Options(mode='raw', pixel_size_mm=.05), inspector.norm)


def test_colour_is_opt_in_and_median_requires_context(tmp_path, inspector):
    p = tmp_path/'colour.png'
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[8:24, :, 0] = 240
    Image.fromarray(a).save(p)
    with pytest.raises(ValueError, match='Colour image'):
        load_image(p, Options(), inspector.norm)
    item = load_image(p, Options(convert_colour=True), inspector.norm)
    with pytest.raises(ValueError, match='actual spatial neighbours'):
        inspector.predict(item, 'median')


def test_folder_continues_on_bad_file_and_marks_duplicates(tmp_path):
    source = tmp_path/'in'
    source.mkdir()
    a = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    Image.fromarray(a).save(source/'a.tif')
    (source/'b.tif').write_bytes((source/'a.tif').read_bytes())
    (source/'broken.png').write_bytes(b'not an image')
    result = run(source, tmp_path/'out', Options(arms=('thresholding',)),campaign=DEFAULT_CAMPAIGN)
    statuses = [r['status'] for r in result['records']]
    assert statuses.count('complete') == statuses.count('duplicate') == statuses.count('error') == 1
    assert Path(result['output'], 'summary.csv').exists()
    with pytest.raises(ValueError, match='outside'):
        run(source, source/'out', Options(),campaign=DEFAULT_CAMPAIGN)


def test_cancel_has_no_predictions(tmp_path):
    event = threading.Event()
    event.set()
    result = run(DEFAULT_CAMPAIGN/'inference_smoke/clean_0.00.npz', tmp_path, Options(mode='raw'), cancel=event,campaign=DEFAULT_CAMPAIGN)
    assert result['cancelled'] and not result['records']


def test_portable_model_ignores_old_absolute_path(tmp_path, inspector):
    # Force a relocated filename resolution without changing the frozen artifact.
    old = inspector.frozen['models']['cnn']['checkpoint']
    inspector.frozen['models']['cnn']['checkpoint'] = 'Z:/missing/training/'+Path(old).name
    try:
        inspector.models.clear()
        inspector._network('cnn')
    finally:
        inspector.frozen['models']['cnn']['checkpoint'] = old


def test_missing_scale_and_unknown_methods_rejected(tmp_path, inspector):
    p = tmp_path/'raw.npy'
    np.save(p, np.arange(4096, dtype=np.float32).reshape(64, 64)/50000)
    with pytest.raises(ValueError, match='0.05'):
        load_image(p, Options(mode='raw'), inspector.norm)
    with pytest.raises(ValueError, match='method'):
        Options(arms=('fake',)).check()


def test_metadata_resolves_scale_and_rejects_channel_mismatch(tmp_path, inspector):
    p = tmp_path/'raw.npy'
    np.save(p, np.arange(4096, dtype=np.float32).reshape(64, 64)/50000)
    meta = p.with_name(p.name+'.json')
    meta.write_text(json.dumps(dict(pixel_size_mm=.05, channel='topo_magnitude')))
    item = load_image(p, Options(mode='raw'), inspector.norm)
    assert item['pixel_size_mm'] == .05 and item['metadata_sha256']
    meta.write_text(json.dumps(dict(pixel_size_mm=.05, channel='signed_difference')))
    with pytest.raises(ValueError, match='magnitude'):
        load_image(p, Options(mode='raw'), inspector.norm)
