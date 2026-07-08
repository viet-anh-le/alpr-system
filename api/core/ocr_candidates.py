"""Route-specific OCR candidate generation and reranking."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .plate_format import chars_to_text, is_vn_plate_chars, mean_confidence



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
) -> list[tuple[str, np.ndarray]]:
    """Return BGR OCR candidates. Original crop is always first."""
    return [("original", crop_bgr.copy())]

