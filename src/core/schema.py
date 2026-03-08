from dataclasses import dataclass
from typing import Tuple

@dataclass
class DetectionResult:
    """
    Standardized output format for ALL algorithms.
    This ensures Program 1, 2, and 3 return data in the exact same way.
    """
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float                    # Confidence (0.0 to 1.0)
    class_id: int                   # 0 = porosity, 1 = crack, etc.
    label: str                      # "porosity", "crack"