import cv2
import numpy as np
from typing import List
from src.core.base_detector import DefectDetector, Detection

class ClassicalMorphologyDetector(DefectDetector):
    def load_model(self):
        # Classical methods don't have weights, but we set params here
        self.blur_kernel = self.config.get('blur_kernel', (5, 5))
        self.threshold_value = self.config.get('threshold', 50)
        self.min_area = self.config.get('min_area', 100)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale and apply Gaussian Blur to reduce 
        Backscattered Electron (BSE) noise.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Denoising is critical for EBW images
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        return blurred

    def detect(self, image: np.ndarray) -> List[Detection]:
        # 1. Preprocess
        processed_img = self.preprocess(image)
        
        # 2. Adaptive Thresholding (Robust to uneven illumination in welding)
        # Inverts image so defects (dark) become white
        thresh = cv2.adaptiveThreshold(
            processed_img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 3. Morphological Operations (Clean up speckle noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter by area (Ignore tiny noise)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Create standardized Detection object
                det = Detection(
                    bbox=(x, y, x+w, y+h),
                    confidence=1.0, # Classical methods don't output probability
                    class_id=0,
                    label="defect"
                )
                detections.append(det)
                
        return detections