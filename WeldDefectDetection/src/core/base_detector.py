from abc import ABC, abstractmethod
import numpy as np
from typing import List
from src.core.interfaces import DetectionResult

class BaseDetector(ABC):
    """
    The Abstract Base Class.
    All three programs must inherit from this and implement 'detect'.
    """
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        pass