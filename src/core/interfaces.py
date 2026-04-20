from dataclasses import dataclass
from typing import Tuple

@dataclass
class DetectionResult:
    """
    Standard output for ALL algorithms (Classical, YOLO, ViT).
    """
    box: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float               # 0.0 to 1.0
    class_id: int                   # 0 = porosity, 1 = crack, etc.
    label: str                      # "porosity", "crack"