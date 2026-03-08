from abc import ABC, abstractmethod
import numpy as np
from typing import List
from src.core.schema import DetectionResult

class BaseDetector(ABC):
    """
    The Parent Class.
    Enforces that every algorithm MUST have a 'detect' method.
    """
    
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for the specific algorithm (e.g., normalize, resize)."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Run inference and return standard results."""
        pass