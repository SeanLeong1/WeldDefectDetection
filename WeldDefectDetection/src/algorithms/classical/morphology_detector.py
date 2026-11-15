from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.core.base_detector import BaseDetector
from src.core.interfaces import DetectionResult
from src.algorithms.classical.preprocessor import ClassicalPreprocessor

class MorphologyDetector(BaseDetector):
    """
    Final Optimal Baseline using Fast Gaussian Blur, 
    Black-Hat Morphology, and Morphological Opening.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        params = self.config.get("classical", {})
        
        # Initialize our fast preprocessor
        self.preprocessor = ClassicalPreprocessor(kernel_size=params.get("blur_kernel", 3))
        
        self.c_value = int(params.get("thresh_c", 12))
        self.min_area = int(params.get("min_defect_area", 5))
        self.max_area_fraction = float(params.get("max_defect_area_fraction", 0.05))
        self.max_component_aspect_ratio = float(params.get("max_component_aspect_ratio", 30.0))
        self.max_component_span_fraction = float(params.get("max_component_span_fraction", 0.50))
        self.blackhat_kernel_size = self._odd(params.get("blackhat_kernel_size", 35))
        self.opening_kernel_size = self._odd(params.get("opening_kernel_size", 3), allow_one=True)
        self.roi_enabled = bool(params.get("weld_roi_enabled", True))
        self.roi_close_width = self._odd(params.get("weld_roi_close_width", 31))
        self.roi_close_height = self._odd(params.get("weld_roi_close_height", 5))
        self.roi_erode_size = self._odd(params.get("weld_roi_erode_size", 1), allow_one=True)

    @staticmethod
    def _odd(value: int, *, allow_one: bool = False) -> int:
        minimum = 1 if allow_one else 3
        result = max(minimum, int(value))
        return result if result % 2 == 1 else result + 1

    def _weld_roi(self, processed: np.ndarray) -> np.ndarray:
        if not self.roi_enabled:
            return np.ones(processed.shape, dtype=np.uint8)
        _, bright = cv2.threshold(
            processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.roi_close_width, self.roi_close_height)
        )
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if count <= 1:
            return np.zeros(processed.shape, dtype=np.uint8)
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        roi = (labels == largest_label).astype(np.uint8)
        if self.roi_erode_size > 1:
            erode_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.roi_erode_size, self.roi_erode_size)
            )
            roi = cv2.erode(roi, erode_kernel)
        return roi

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        results, _, _ = self._run_detection(image, capture_debug=False)
        return results

    def detect_with_stages(self, image: np.ndarray) -> Tuple[List[DetectionResult], Dict[str, np.ndarray]]:
        results, stages, _ = self._run_detection(image, capture_debug=True)
        return results, stages

    def detect_with_mask(self, image: np.ndarray) -> Tuple[List[DetectionResult], np.ndarray]:
        results, _, binary_mask = self._run_detection(image, capture_debug=False)
        return results, binary_mask

    def predict_binary_mask(self, image: np.ndarray) -> np.ndarray:
        _, _, binary_mask = self._run_detection(image, capture_debug=False)
        return binary_mask

    def _run_detection(
        self, image: np.ndarray, capture_debug: bool = False
    ) -> Tuple[List[DetectionResult], Dict[str, np.ndarray], np.ndarray]:
        processed = self.preprocessor.process(image)
        img_area = processed.shape[0] * processed.shape[1]

        # 1. Limit the search to the connected bright weld band. This prevents
        # the normal weld/background boundary from becoming a long false defect.
        weld_roi = self._weld_roi(processed)

        # 2. Black-hat morphology isolates dark anomalies on the bright weld.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.blackhat_kernel_size, self.blackhat_kernel_size)
        )
        blackhat = cv2.morphologyEx(processed, cv2.MORPH_BLACKHAT, kernel)

        # 3. Global thresholding extracts the strongest local deficits.
        _, thresh = cv2.threshold(blackhat, self.c_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=(weld_roi * 255))

        # 4. Optional opening removes isolated pixels.
        if self.opening_kernel_size > 1:
            clean_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.opening_kernel_size, self.opening_kernel_size),
            )
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)
        else:
            opened = thresh

        # 5. Apply component filtering to the scored mask as well as boxes.
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (opened > 0).astype(np.uint8), connectivity=8
        )
        binary_mask = np.zeros(processed.shape, dtype=np.uint8)
        results: List[DetectionResult] = []
        max_area = max(self.min_area, int(img_area * self.max_area_fraction))
        for label_index in range(1, component_count):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            w = int(stats[label_index, cv2.CC_STAT_WIDTH])
            h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            aspect_ratio = max(w, h) / max(1, min(w, h))
            span_fraction = max(w / processed.shape[1], h / processed.shape[0])
            if (
                self.min_area <= area <= max_area
                and aspect_ratio <= self.max_component_aspect_ratio
                and span_fraction <= self.max_component_span_fraction
            ):
                binary_mask[labels == label_index] = 1
                results.append(DetectionResult(
                    box=(x, y, x + w, y + h),
                    confidence=1.0,
                    class_id=2, 
                    label="defect"
                ))

        stages: Dict[str, np.ndarray] = {}
        if capture_debug:
            if len(image.shape) == 2:
                original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                original_bgr = image.copy()

            contour_view = original_bgr.copy()
            contours, _ = cv2.findContours(
                (binary_mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(contour_view, contours, -1, (0, 255, 0), 1)

            stages = {
                "input": original_bgr,
                "weld_roi": weld_roi * 255,
                "blackhat": blackhat.copy(),
                "threshold": thresh.copy(),
                "opened": opened.copy(),
                "contours": contour_view,
            }

        return results, stages, binary_mask
