from __future__ import annotations

from src.algorithms.classical.cnn_detector import TinyDefectSegNet, TorchNotInstalledError, torch


class BidirectionalSurfacePredictor(TinyDefectSegNet):
    """Small U-Net regressor that predicts a target TOPO crop from the declared context neighbours."""

    def __init__(self, context_channels: int = 4, base_channels: int = 8):
        super().__init__(in_channels=context_channels, base_channels=base_channels)

    def forward(self, x):
        return torch.sigmoid(super().forward(x))


__all__ = ["BidirectionalSurfacePredictor", "TorchNotInstalledError", "torch"]
