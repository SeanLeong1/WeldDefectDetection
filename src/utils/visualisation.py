import cv2
import numpy as np
from typing import List
from src.core.interfaces import DetectionResult

def draw_detections(image: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
    output = image.copy()
    for res in results:
        x1, y1, x2, y2 = res.box
        # Draw Red Box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Draw Label
        cv2.putText(output, f"{res.label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output