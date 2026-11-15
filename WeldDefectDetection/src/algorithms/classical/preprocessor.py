import cv2
import numpy as np

class ClassicalPreprocessor:
    def __init__(self, kernel_size: int = 3): # Tiny kernel!
        self.kernel_size = (kernel_size, kernel_size)

    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Fast, gentle smoothing. Won't erase micro-cracks.
        blurred = cv2.GaussianBlur(gray, self.kernel_size, 0)
        return blurred