import hashlib
import json

import cv2
import numpy as np
import pytest
import yaml

from src.algorithms.classical.cnn_detector import CNNDetector, torch
from src.algorithms.temporal_predictor import BidirectionalSurfacePredictor
from src.data.datasimulation_dataset import DataSimulationSegmentationDataset
from src.data.normalization_audit import check_frozen_normalization
from src.data.temporal_window_dataset import TemporalWindowDataset
from src.evaluation.evaluate_dataset import _predict_mask
from src.utils.image_preprocessing import normalize_grayscale, filter_small_components
from test_dataset_audits import _write_static, _write_sequence


def test_dtype_scaling_and_evaluator_preserve_cnn_precision():
    image = np.array([[0, 1, 32769, 65535]], dtype=np.uint16)
    gray = normalize_grayscale(image)
    assert gray[0, 1] == pytest.approx(1/65535)
    assert normalize_grayscale(np.array([[1]], dtype=np.uint16)).item() == pytest.approx(1/65535)
    class Capture:
        def predict_binary_mask(self, value):
            self.value = value.copy()
            return value > .5
    detector = Capture()
    _predict_mask('cnn', detector, gray)
    np.testing.assert_array_equal(detector.value, gray)
    _predict_mask('morphology', detector, gray)
    assert detector.value.dtype == np.uint8
    with pytest.raises(ValueError, match='already'):
        normalize_grayscale(np.array([[12.]], np.float32))


def test_binary_masks_and_boxes_share_minimum_pixel_area():
    detector = CNNDetector({'cnn': {'input_size': 16, 'base_channels': 2, 'min_defect_area': 8}})
    probability = np.zeros((16, 16), np.float32)
    probability[0, 0] = 1
    probability[5:7, 5:9] = 1  # eight pixels, contour area only three
    detector.predict_mask = lambda image: probability
    mask = detector.predict_binary_mask(probability)
    boxes, debug = detector.detect_with_debug(probability)
    assert mask.sum() == 8 and not mask[0, 0]
    np.testing.assert_array_equal(mask, debug.binary_mask)
    assert len(boxes) == 1 and boxes[0].box == (5, 5, 9, 7)


def test_rectangular_resize_preserves_physical_area(tmp_path):
    root = _write_static(tmp_path/'data', seed_offset=1, value_offset=0)
    records = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    row = records[0]; row['pixel_size_mm'] = .2
    cv2.imwrite(str(root/row['channels']['topo']), np.ones((8, 12), np.uint16))
    cv2.imwrite(str(root/row['mask_path']), np.full((8, 12), 255, np.uint8))
    (root/'samples.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in records))
    sample = DataSimulationSegmentationDataset(root, 'train', input_size=24)[0]
    assert sample['pixel_size_x_mm'] == pytest.approx(.1)
    assert sample['pixel_size_y_mm'] == pytest.approx(.2/3)
    assert sample['mask'].sum().item()*sample['pixel_area_mm2'] == pytest.approx(8*12*.2**2)
    assert sample['image'].max().item() == pytest.approx(1/65535)


def test_loader_rejects_actual_target_overlap(tmp_path):
    root = _write_sequence(tmp_path/'seq', seed=1, value_offset=0)
    descriptor = yaml.safe_load((root/'dataset_contract.yaml').read_text())
    descriptor['stride_px'] = 2
    records = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    for row in records: row['crop_x0_px'] = 2*row['frame_index']
    (root/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (root/'dataset_contract.yaml').write_text(yaml.safe_dump(descriptor))
    with pytest.raises(ValueError, match='overlap'):
        TemporalWindowDataset(root, 'train')


def test_radius_one_and_three_models_accept_declared_channels():
    for radius in (1, 3):
        model = BidirectionalSurfacePredictor(context_channels=2*radius, base_channels=2).eval()
        with torch.no_grad():
            assert model(torch.zeros(1, 2*radius, 16, 16)).shape == (1, 1, 16, 16)


def test_normalization_hash_and_parameters_must_match_source(tmp_path):
    development = _write_static(tmp_path/'dev', seed_offset=1, value_offset=0)
    confirmation = _write_static(tmp_path/'confirm', seed_offset=10, value_offset=1000)
    source = development/'topo_normalization.json'
    reference = json.loads(source.read_text()); reference['fit_performed_on_this_dataset'] = True
    source.write_text(json.dumps(reference))
    mapping = {**reference, 'fit_performed_on_this_dataset': False,
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    destination = confirmation/'topo_normalization.json'
    destination.write_text(json.dumps(mapping))
    assert not check_frozen_normalization(confirmation, [development])
    mapping['raw_high'] = 2
    destination.write_text(json.dumps(mapping))
    assert 'parameters differ' in check_frozen_normalization(confirmation, [development])[0]
