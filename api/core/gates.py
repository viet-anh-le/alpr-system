"""
core/gates.py — Layer 1 pre-OCR quality filters.

Each gate is a standalone predicate so it can be unit-tested independently.
is_sharp() remains as the hard binary gate; the continuous score lives in
quality_scorer.py and is used for buffer eviction and attention bias.
"""
import cv2
import numpy as np

from .config import BLUR_THRESHOLD
from .quality_scorer import quality_score

_HARD_GATE_MIN = 0.05   # quality_score below this → crop is unusable


def is_sharp(crop: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    """
    Layer 1c: Hard gate — reject motion-blurred or out-of-focus plate crops.

    Returns False when quality_score < _HARD_GATE_MIN OR Laplacian variance
    is below threshold so that completely degraded crops never enter the buffer.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < threshold:
        return False
    return quality_score(crop) >= _HARD_GATE_MIN


def is_router_candidate(crop: np.ndarray) -> bool:
    """Return True for crops worth sending into the quality router.

    Unlike is_sharp(), this does not reject blur. The router needs degraded
    crops so it can decide between enhancement, tracklet fusion, or unreadable.
    """
    return crop.size > 0 and quality_score(crop) >= _HARD_GATE_MIN
