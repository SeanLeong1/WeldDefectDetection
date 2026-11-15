from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.utils.image_preprocessing import normalize_grayscale, filter_small_components

from src.core.base_detector import BaseDetector
from src.core.interfaces import DetectionResult

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class TorchNotInstalledError(ImportError):
    pass


if nn is not None:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class TinyDefectSegNet(nn.Module):
        """
        Small U-Net-style scaffold for binary defect segmentation.
        """

        def __init__(self, in_channels: int = 1, base_channels: int = 16):
            super().__init__()
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4
            self.enc1 = ConvBlock(in_channels, c1)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = ConvBlock(c1, c2)
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(c2, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
            self.dec2 = ConvBlock(c2 + c2, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
            self.dec1 = ConvBlock(c1 + c1, c1)
            self.head = nn.Conv2d(c1, 1, kernel_size=1)

        def forward(self, x):
            skip1 = self.enc1(x)
            skip2 = self.enc2(self.pool1(skip1))
            bottleneck = self.bottleneck(self.pool2(skip2))
            x = self.up2(bottleneck)
            x = torch.cat([x, skip2], dim=1)
            x = self.dec2(x)
            x = self.up1(x)
            x = torch.cat([x, skip1], dim=1)
            x = self.dec1(x)
            return self.head(x)
else:
    class TinyDefectSegNet:  # pragma: no cover - fallback placeholder
        def __init__(self, *args, **kwargs):
            raise TorchNotInstalledError(
                "PyTorch is required for TinyDefectSegNet. Install torch before using the CNN scaffold."
            )


@dataclass(frozen=True)
class CNNDebugOutput:
    probability_map: np.ndarray
    binary_mask: np.ndarray


class CNNDetector(BaseDetector):
    """
    Starter CNN detector built around a binary segmentation head.

    This is intentionally a scaffold:
    - loads a small segmentation CNN
    - runs inference on a single grayscale image
    - thresholds the mask
    - converts connected components into DetectionResult boxes
    """

    def __init__(self, config: dict):
        super().__init__(config)
        if torch is None or nn is None:
            raise TorchNotInstalledError(
                "PyTorch is not installed. Add torch to the environment before using CNNDetector."
            )

        params = self.config.get("cnn", {})
        self.device = torch.device(params.get("device", "cpu"))
        self.threshold = float(params.get("threshold", 0.50))
        self.min_defect_area = int(params.get("min_defect_area", 8))
        self.input_size = int(params.get("input_size", 256))
        self.in_channels = int(params.get("in_channels", 1))
        base_channels = int(params.get("base_channels", 16))
        self.model = TinyDefectSegNet(
            in_channels=self.in_channels,
            base_channels=base_channels,
        ).to(self.device)
        self.model.eval()

        checkpoint_path = params.get("checkpoint_path")
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        if torch is None:
            raise TorchNotInstalledError("PyTorch is not installed.")
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            if "min_defect_area" in state:
                self.min_defect_area = int(state["min_defect_area"])
            if "threshold" in state:
                self.threshold = float(state["threshold"])
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        results, _ = self.detect_with_debug(image)
        return results

    def detect_with_debug(self, image: np.ndarray) -> tuple[List[DetectionResult], CNNDebugOutput]:
        probability_map = self.predict_mask(image)
        binary_mask = filter_small_components(probability_map >= self.threshold, self.min_defect_area)
        results = self._mask_to_detections(binary_mask)
        return results, CNNDebugOutput(
            probability_map=probability_map,
            binary_mask=binary_mask,
        )

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        if torch is None:
            raise TorchNotInstalledError("PyTorch is not installed.")
        prepared, original_shape = self._prepare_input(image)
        with torch.no_grad():
            logits = self.model(prepared)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        if probs.shape != original_shape:
            probs = cv2.resize(
                probs,
                (int(original_shape[1]), int(original_shape[0])),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
        return probs

    def predict_binary_mask(self, image: np.ndarray) -> np.ndarray:
        probability_map = self.predict_mask(image)
        return filter_small_components(probability_map >= self.threshold, self.min_defect_area)

    @staticmethod
    def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
        return normalize_grayscale(image)

    def _prepare_input(self, image: np.ndarray):
        if torch is None:
            raise TorchNotInstalledError("PyTorch is not installed.")
        gray = self._normalize_grayscale(image)
        original_shape = gray.shape
        resized = cv2.resize(gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized[None, None, ...]).float().to(self.device)
        return tensor, original_shape

    def _mask_to_detections(self, binary_mask: np.ndarray) -> List[DetectionResult]:
        filtered = filter_small_components(binary_mask, self.min_defect_area)
        _, _, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
        results: List[DetectionResult] = []
        for x, y, w, h, area in stats[1:]:
            x, y, w, h = int(x), int(y), int(w), int(h)
            results.append(
                DetectionResult(
                    box=(x, y, x + w, y + h),
                    confidence=1.0,
                    class_id=2,
                    label="defect",
                )
            )
        return results
