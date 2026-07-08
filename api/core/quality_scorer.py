"""
core/quality_scorer.py — Continuous plate-crop quality score in [0, 1].

Replaces the binary is_sharp() gate for buffer eviction and top-frame voting.
is_sharp() still acts as a hard gate (reject if quality_score < 0.05) to filter
completely unusable crops before buffering.

Aspect ratio is intentionally NOT used as a quality signal: Vietnamese plates
come in two types (long single-row ~4.7:1 and square two-row ~1.4:1), and the
detector can split a two-row plate into two single-row crops — each of which
has a ratio close to a long plate. Ratio therefore cannot reliably distinguish
a good detection from a partial one.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import LAP_MAX, MIN_PLATE_H, MIN_PLATE_W

_MIN_PLATE_AREA = MIN_PLATE_H * MIN_PLATE_W * 9   # 3× minimum area as normalisation ceiling


def quality_score(crop_bgr: np.ndarray) -> float:
    """
    Return a quality score in [0, 1] combining two signals:
      - Laplacian variance / sharpness  (weight 0.80)
      - Relative plate area             (weight 0.20)

    Higher is better. Used for buffer eviction and top-frame voting.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp = min(lap_var / LAP_MAX, 1.0)

    h, w = crop_bgr.shape[:2]
    size = min(h * w / _MIN_PLATE_AREA, 1.0)

    return 0.80 * sharp + 0.20 * size
