"""Route-specific OCR candidate generation and reranking."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .plate_format import chars_to_text, is_vn_plate_chars, mean_confidence
from .quality_router import DegradationTags


@dataclass(frozen=True)
class OcrCandidateResult:
    method: str
    char_probs: list[tuple[str, float]]
    risk_penalty: float = 0.0
    format_mode: str = "raw"

    @property
    def text(self) -> str:
        return chars_to_text(self.char_probs)

    @property
    def confidence(self) -> float:
        return mean_confidence(self.char_probs)

    @property
    def is_valid(self) -> bool:
        return is_vn_plate_chars(self.char_probs, format_mode=self.format_mode)


def rerank_ocr_candidates(
    candidates: list[OcrCandidateResult],
    *,
    temporal_texts: list[str] | None = None,
) -> OcrCandidateResult | None:
    if not candidates:
        return None

    temporal_counts = {
        text: temporal_texts.count(text)
        for text in set(temporal_texts or [])
        if text
    }

    def score(candidate: OcrCandidateResult) -> tuple[float, int]:
        value = candidate.confidence
        value += 0.35 if candidate.is_valid else -0.25
        value += 0.04 if candidate.method == "original" else -0.02
        value -= candidate.risk_penalty
        if temporal_counts:
            value += min(0.18, 0.06 * temporal_counts.get(candidate.text, 0))
        original_tie_breaker = 1 if candidate.method == "original" else 0
        return value, original_tie_breaker

    return max(candidates, key=score)


def build_candidate_crops(
    crop_bgr: np.ndarray,
    tags: DegradationTags,
) -> list[tuple[str, np.ndarray]]:
    """Return BGR OCR candidates. Original crop is always first."""
    candidates: list[tuple[str, np.ndarray]] = [("original", crop_bgr.copy())]
    added = {"original"}

    def add(method: str, image: np.ndarray) -> None:
        if method in added:
            return
        added.add(method)
        candidates.append((method, image))

    if tags.motion_blur or tags.low_res:
        add("sharpen", _unsharp(crop_bgr, amount=0.65))
    if tags.low_light:
        add("gamma", _gamma(crop_bgr, 0.68))
    if tags.low_light or tags.low_contrast:
        add("grayscale", _gray_bgr(crop_bgr))
        add("clahe", _clahe_luminance(crop_bgr, clip_limit=2.4))
    if tags.low_contrast:
        add("contrast_stretch", _stretch_luminance(crop_bgr))
    if tags.rain_or_haze:
        add("denoise", cv2.bilateralFilter(crop_bgr, 5, 30, 30))
        add("haze_contrast", _clahe_luminance(_stretch_luminance(crop_bgr), clip_limit=1.8))
    if tags.faulty_color:
        add("white_balance", _gray_world(crop_bgr))

    return candidates


def _gray_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _clahe_luminance(image: np.ndarray, *, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    enhanced_l = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def _gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def _unsharp(image: np.ndarray, *, amount: float = 0.55, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


def _stretch_luminance(image: np.ndarray, *, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    lo, hi = np.percentile(l_chan, (low_pct, high_pct))
    if hi <= lo:
        return image.copy()
    stretched = np.clip((l_chan.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return cv2.cvtColor(cv2.merge((stretched.astype(np.uint8), a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def _gray_world(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    scale = target / np.maximum(means, 1.0)
    balanced = np.clip(img * scale.reshape(1, 1, 3), 0, 255)
    return balanced.astype(np.uint8)
