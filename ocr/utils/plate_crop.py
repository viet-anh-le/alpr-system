from __future__ import annotations

import numpy as np


def crop_plate(image: np.ndarray, bbox: list[float], margin: float = 0.05) -> np.ndarray:
    """Crop license plate region with optional margin."""
    raise NotImplementedError


def perspective_correct(image: np.ndarray, corners: np.ndarray, target_size: tuple[int, int] = (94, 24)) -> np.ndarray:
    """Apply perspective transform to straighten a tilted plate."""
    import cv2
    dst = np.array([[0, 0], [target_size[0], 0], [target_size[0], target_size[1]], [0, target_size[1]]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, target_size)
