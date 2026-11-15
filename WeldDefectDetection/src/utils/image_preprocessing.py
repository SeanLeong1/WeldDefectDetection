"""Versioned intensity contract shared by datasets and CNN inference."""
import cv2
import numpy as np

PREPROCESSING_VERSION = "grayscale-dtype-range/v2"


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Image must be nonempty and finite.")
    if np.issubdtype(array.dtype, np.integer):
        if np.any(array < 0):
            raise ValueError("Integer image intensities must be nonnegative.")
        gray = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        gray = array.astype(np.float32)
        if np.any((gray < 0) | (gray > 1)):
            raise ValueError("Floating image intensities must already lie in [0, 1].")
    if gray.ndim == 3 and gray.shape[-1] in (3, 4):
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY if gray.shape[-1] == 3 else cv2.COLOR_BGRA2GRAY)
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale or BGR/BGRA image.")
    return gray


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Keep 8-connected components by foreground pixel count, including holes."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(min_area))
    return keep[labels]
