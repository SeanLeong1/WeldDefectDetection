from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.temporal_predictor import BidirectionalSurfacePredictor, torch
from src.data.temporal_window_dataset import align_context_frames


def test_registered_context_recovers_target_coordinates() -> None:
    scene = np.tile(np.arange(14, dtype=np.float32), (4, 1))
    frames = [scene[:, start:start + 6] for start in (0, 2, 6, 8)]
    aligned = align_context_frames(frames, [0, 1, 3, 4], target_index=2, stride_px=2)
    target = scene[:, 4:10]
    assert aligned.shape == (4, 4, 6)
    assert np.allclose(aligned, target[None])


@pytest.mark.skipif(torch is None, reason="PyTorch is optional")
def test_temporal_predictor_preserves_target_shape() -> None:
    model = BidirectionalSurfacePredictor(context_channels=4, base_channels=4)
    output = model(torch.zeros((2, 4, 32, 32)))
    assert tuple(output.shape) == (2, 1, 32, 32)
    assert bool(torch.all((output >= 0.0) & (output <= 1.0)))
