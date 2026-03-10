import cv2
import numpy as np
from typing import List
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
        
        # Extract tuned parameters
        self.c_value = params.get("thresh_c", 12)
        self.min_area = params.get("min_defect_area", 5)

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        processed = self.preprocessor.process(image)
        img_area = processed.shape[0] * processed.shape[1]

        # 1. Black-Hat Morphology (Isolates dark anomalies on bright backgrounds)
        filter_size = 35 # Large enough to encompass the biggest defect
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
        blackhat = cv2.morphologyEx(processed, cv2.MORPH_BLACKHAT, kernel)

        # 2. Global Thresholding (Extract the sharpest peaks)
        _, thresh = cv2.threshold(blackhat, self.c_value, 255, cv2.THRESH_BINARY)

        # 3. Morphological OPENING (Destroys tiny specs of remaining film grain)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel)

        # 4. Extract Contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 5. Size Filter (floor to block dust, ceiling to block whole-image boxes)
            if self.min_area <= area < (img_area * 0.05):
                x, y, w, h = cv2.boundingRect(cnt)
                results.append(DetectionResult(
                    box=(x, y, x + w, y + h),
                    confidence=1.0,
                    class_id=2, 
                    label="defect"
                ))

        return results