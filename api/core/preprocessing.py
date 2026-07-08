"""Video-frame preprocessing presets for offline ALPR input.

The functions here operate on BGR OpenCV frames and preserve shape/dtype so
they can be inserted before detection without changing downstream contracts.
"""
from __future__ import annotations

from typing import Iterator, Literal

import cv2
import numpy as np

from .frame_source import FrameSource


PreprocessMode = Literal["none", "night", "low_contrast", "fog", "rain", "glare"]
PREPROCESS_MODES: set[str] = {"none", "night", "low_contrast", "fog", "rain", "glare"}


def normalize_preprocess_mode(mode: str | None) -> PreprocessMode:
    value = (mode or "none").strip().lower()
    if not value:
        value = "none"
    if value not in PREPROCESS_MODES:
        allowed = ", ".join(sorted(PREPROCESS_MODES))
        raise ValueError(f"Invalid preprocess_mode '{mode}'. Expected one of: {allowed}")
    return value  # type: ignore[return-value]


def _clahe_luminance(frame: np.ndarray, *, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    enhanced_l = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def _gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def _unsharp(frame: np.ndarray, *, amount: float = 0.55, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def _stretch_luminance(frame: np.ndarray, *, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    lo, hi = np.percentile(l_chan, (low_pct, high_pct))
    if hi <= lo:
        return frame.copy()
    stretched = np.clip((l_chan.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((stretched, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def _night(frame: np.ndarray) -> np.ndarray:
    enhanced = _gamma(frame, 0.68)
    enhanced = _clahe_luminance(enhanced, clip_limit=2.4)
    return cv2.bilateralFilter(enhanced, 5, 35, 35)


def _low_contrast(frame: np.ndarray) -> np.ndarray:
    enhanced = _stretch_luminance(frame)
    enhanced = _clahe_luminance(enhanced, clip_limit=2.2)
    return _unsharp(enhanced, amount=0.45)


def _fog(frame: np.ndarray) -> np.ndarray:
    # Conservative haze reduction: remove a blurred luminance veil, then restore local contrast.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    veil = cv2.GaussianBlur(l_chan, (0, 0), 15)
    corrected_l = cv2.addWeighted(l_chan, 1.35, veil, -0.35, 18)
    corrected = cv2.cvtColor(cv2.merge((corrected_l, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
    corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _clahe_luminance(corrected, clip_limit=1.8)


def _rain(frame: np.ndarray) -> np.ndarray:
    denoised = cv2.medianBlur(frame, 3)
    denoised = cv2.bilateralFilter(denoised, 5, 30, 30)
    return _unsharp(denoised, amount=0.25)


def _glare(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((value > 220) & (saturation < 150)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    if int(mask.sum()) == 0:
        return _clahe_luminance(frame, clip_limit=1.8)

    repaired = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    blended = frame.copy()
    blended[mask > 0] = repaired[mask > 0]
    return _clahe_luminance(blended, clip_limit=1.8)


def apply_preprocessing(frame: np.ndarray, mode: str | None = "none") -> np.ndarray:
    normalized = normalize_preprocess_mode(mode)
    source = frame.copy()
    if normalized == "none":
        return source
    if normalized == "night":
        return _night(source)
    if normalized == "low_contrast":
        return _low_contrast(source)
    if normalized == "fog":
        return _fog(source)
    if normalized == "rain":
        return _rain(source)
    if normalized == "glare":
        return _glare(source)
    raise AssertionError(f"Unhandled preprocess mode: {normalized}")


class PreprocessedFrameSource:
    """FrameSource wrapper that enhances frames before downstream inference."""

    def __init__(self, source: FrameSource, mode: str | None = "none") -> None:
        self.source = source
        self.mode = normalize_preprocess_mode(mode)
        self.fps = source.fps
        self.frame_size = source.frame_size
        self.total_frames = source.total_frames

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        for frame_idx, frame, ts in self.source.iter_frames():
            yield frame_idx, apply_preprocessing(frame, self.mode), ts
