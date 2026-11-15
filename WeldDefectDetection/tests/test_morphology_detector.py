from __future__ import annotations

import cv2
import numpy as np

from src.algorithms.classical.morphology_detector import MorphologyDetector


def _config(**overrides):
    values = {
        "blur_kernel": 3,
        "thresh_c": 8,
        "min_defect_area": 2,
        "max_defect_area_fraction": 0.10,
        "max_component_aspect_ratio": 20.0,
        "max_component_span_fraction": 0.40,
        "blackhat_kernel_size": 15,
        "opening_kernel_size": 1,
        "weld_roi_enabled": True,
        "weld_roi_close_width": 21,
        "weld_roi_close_height": 5,
        "weld_roi_erode_size": 1,
    }
    values.update(overrides)
    return {"classical": values}


def test_predict_binary_mask_is_filtered_and_binary():
    image = np.full((96, 96), 20, dtype=np.uint8)
    cv2.ellipse(image, (48, 48), (40, 10), 0, 0, 360, 210, -1)
    cv2.circle(image, (48, 48), 4, 35, -1)
    image[47:49, 12:84] = 35

    detector = MorphologyDetector(_config())
    prediction = detector.predict_binary_mask(image)

    assert prediction.shape == image.shape
    assert set(np.unique(prediction)).issubset({0, 1})
    count, _, stats, _ = cv2.connectedComponentsWithStats(prediction, connectivity=8)
    for label in range(1, count):
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        assert max(width / image.shape[1], height / image.shape[0]) <= 0.40


def test_minimum_area_filter_applies_to_scored_mask():
    image = np.full((64, 64), 210, dtype=np.uint8)
    image[32, 32] = 0
    detector = MorphologyDetector(
        _config(
            blur_kernel=1,
            min_defect_area=8,
            blackhat_kernel_size=9,
            weld_roi_enabled=False,
        )
    )
    assert int(detector.predict_binary_mask(image).sum()) == 0
