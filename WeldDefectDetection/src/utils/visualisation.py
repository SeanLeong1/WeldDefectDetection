import cv2
import numpy as np
from typing import List, Sequence, Tuple
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

def build_stage_montage(
    stages: Sequence[Tuple[str, np.ndarray]],
    layout: str = "horizontal",
    spacer_size: int = 8,
) -> np.ndarray:
    if not stages:
        raise ValueError("stages must contain at least one (label, image) pair")

    base_h, base_w = stages[0][1].shape[:2]
    prepared = []

    for label, image in stages:
        panel = image.copy()
        if len(panel.shape) == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        if panel.shape[:2] != (base_h, base_w):
            panel = cv2.resize(panel, (base_w, base_h))

        cv2.rectangle(panel, (0, 0), (base_w, 28), (30, 30, 30), -1)
        cv2.putText(
            panel,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        prepared.append(panel)

    if layout not in {"horizontal", "vertical"}:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    montage = prepared[0]
    if layout == "horizontal":
        spacer = np.full((base_h, spacer_size, 3), 20, dtype=np.uint8)
        for panel in prepared[1:]:
            montage = np.hstack((montage, spacer, panel))
    else:
        spacer = np.full((spacer_size, base_w, 3), 20, dtype=np.uint8)
        for panel in prepared[1:]:
            montage = np.vstack((montage, spacer, panel))

    return montage

def fit_to_bounds(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image

    width_scale = max_width / w
    height_scale = max_height / h
    scale = min(width_scale, height_scale, 1.0)

    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image
