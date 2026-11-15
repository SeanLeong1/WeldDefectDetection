from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.evaluate_dataset import (
    _bootstrap_by_image,
    _defect_contrast_band,
    _defect_size_band,
)
from src.evaluation.evaluate_gdxray import _tile_starts, predict_tiled_probability
from src.evaluation.temporal_reconstruction import run_temporal_reconstruction
from src.evaluation.tune_classical import tune_classical_baseline
from src.training.train_cnn import _build_dataloader, _metrics_for_thresholds, torch


def test_classical_tuner_refuses_test_split() -> None:
    with pytest.raises(ValueError, match="test split"):
        tune_classical_baseline("missing.yaml", split="test", output_dir="missing")


def test_temporal_tuner_refuses_test_split() -> None:
    with pytest.raises(ValueError, match="test split"):
        run_temporal_reconstruction(
            dataset_root="missing", output_dir="missing", selection_split="test"
        )


def test_temporal_frozen_parameters_must_be_supplied_together() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        run_temporal_reconstruction(
            dataset_root="missing",
            output_dir="missing",
            frozen_threshold=0.1,
        )


def test_temporal_confirmation_requires_development_root() -> None:
    with pytest.raises(ValueError, match="development_dataset_root"):
        run_temporal_reconstruction(
            dataset_root="missing",
            output_dir="missing",
            evaluation_split="all",
            frozen_threshold=0.1,
            frozen_min_area=5,
        )


def test_static_bootstrap_reports_image_resampling() -> None:
    rows = [
        {
            "tp": 1,
            "tn": 9,
            "fp": 0,
            "fn": 0,
            "target_is_defective": True,
            "predicted_is_defective": True,
        },
        {
            "tp": 0,
            "tn": 9,
            "fp": 1,
            "fn": 0,
            "target_is_defective": False,
            "predicted_is_defective": True,
        },
    ]
    result = _bootstrap_by_image(rows, seed=7, iterations=20)
    assert result["resampling_unit"] == "image"
    assert result["iterations"] == 20
    assert 0.0 <= result["pixel_iou"]["ci_lower"] <= 1.0


def test_static_robustness_bands_use_fixed_physical_boundaries() -> None:
    assert _defect_size_band(0.0) == "clean"
    assert _defect_size_band(0.249) == "small_lt_0.25_mm2"
    assert _defect_size_band(0.25) == "medium_0.25_to_1.0_mm2"
    assert _defect_size_band(1.0) == "large_ge_1.0_mm2"
    assert _defect_contrast_band(None) == "clean"
    assert _defect_contrast_band(0.049) == "low_lt_0.05"
    assert _defect_contrast_band(0.05) == "medium_0.05_to_0.15"
    assert _defect_contrast_band(0.15) == "high_ge_0.15"


@pytest.mark.skipif(torch is None, reason="PyTorch is optional")
def test_cnn_threshold_selection_prefers_separating_threshold() -> None:
    probabilities = [
        np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32),
    ]
    targets = [np.array([[0, 0], [1, 1]], dtype=np.uint8)]
    threshold, metrics = _metrics_for_thresholds(
        probabilities, targets, np.array([0.15, 0.5, 0.85], dtype=np.float32)
    )
    assert threshold == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1.0)


@pytest.mark.skipif(torch is None, reason="PyTorch is optional")
def test_cnn_loader_accepts_variable_length_defect_metadata() -> None:
    class VariableMetadataDataset:
        records = [{"defects": []}, {"defects": ["pore", "crack"]}]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "image": torch.zeros((1, 4, 4)),
                "mask": torch.zeros((1, 4, 4)),
                "defects": self.records[index]["defects"],
            }

    loader = _build_dataloader(
        VariableMetadataDataset(),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        seed=7,
    )
    batch = next(iter(loader))
    assert tuple(batch["image"].shape) == (2, 1, 4, 4)
    assert batch["defects"] == [[], ["pore", "crack"]]


def test_tile_starts_cover_last_edge() -> None:
    assert _tile_starts(3512, 128, 96)[-1] == 3512 - 128
    assert _tile_starts(64, 128, 96) == [0]


def test_tiled_prediction_preserves_shape_and_covers_edges() -> None:
    class MeanDetector:
        @staticmethod
        def _normalize_grayscale(image):
            return np.asarray(image, dtype=np.float32)

        @staticmethod
        def predict_mask(tile):
            return np.asarray(tile, dtype=np.float32)

    image = np.arange(7 * 11, dtype=np.float32).reshape(7, 11) / 76.0
    probability = predict_tiled_probability(
        MeanDetector(), image, tile_size=4, stride=3
    )
    assert probability.shape == image.shape
    assert np.allclose(probability, image)
