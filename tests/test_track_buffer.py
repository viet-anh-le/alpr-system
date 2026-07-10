# tests/test_track_buffer.py
from __future__ import annotations

import numpy as np
import pytest

from api.core.tracker import TrackBuffer


def _crop(h: int = 24, w: int = 80) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _chars(avg_conf: float, n: int = 5) -> list[tuple[str, float]]:
    return [("A", avg_conf)] * n


class TestTrackBufferEviction:
    def test_high_visual_low_ocr_evicted_before_lower_visual_high_ocr(self):
        """
        Bug regression: a crop with high visual quality but low OCR confidence
        must NOT displace a crop with lower visual quality but high OCR confidence.

        combined(correct)  = 0.91 × 0.93 = 0.846
        combined(wrong)    = 0.94 × 0.35 = 0.329  ← should be evicted
        """
        buf = TrackBuffer(max_size=2)
        correct_chars = _chars(avg_conf=0.93)
        wrong_chars = _chars(avg_conf=0.35)

        buf.add(_crop(), quality_score=0.91, ocr_conf=0.93, char_probs=correct_chars, frame_idx=1)
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.93, char_probs=correct_chars, frame_idx=2)
        # Buffer full. Add a visually sharper but OCR-garbled crop.
        buf.add(_crop(), quality_score=0.94, ocr_conf=0.35, char_probs=wrong_chars, frame_idx=3)

        assert len(buf.crops) == 2
        crops, scores, prob_lists = buf.top_k(k=2)
        # Both retained entries must be the correct-OCR ones (avg_conf=0.93)
        for pl in prob_lists:
            assert all(abs(p - 0.93) < 1e-6 for _, p in pl)

    def test_eviction_removes_worst_combined_not_worst_visual(self):
        """
        When the buffer is full and a new crop arrives, the crop with the lowest
        combined score (quality × ocr_conf) is evicted — even if it has a
        higher visual quality_score than the new crop.
        """
        buf = TrackBuffer(max_size=1)
        # Existing crop: high visual, low OCR → combined = 0.95 × 0.20 = 0.19
        buf.add(_crop(), quality_score=0.95, ocr_conf=0.20, char_probs=_chars(0.20), frame_idx=1)

        # New crop: lower visual, high OCR → combined = 0.80 × 0.93 = 0.744 > 0.19
        buf.add(_crop(), quality_score=0.80, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=2)

        # New crop should have replaced the existing one
        assert len(buf.crops) == 1
        _, _, prob_lists = buf.top_k(k=1)
        assert abs(prob_lists[0][0][1] - 0.93) < 1e-6

    def test_new_crop_not_added_when_combined_score_below_all_existing(self):
        """
        A new crop whose combined score is lower than every existing entry is
        rejected immediately (never added then evicted — eviction still removes
        the worst, which in this case is the new crop itself).
        """
        buf = TrackBuffer(max_size=2)
        buf.add(_crop(), quality_score=0.91, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=1)
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=2)

        # New crop: combined = 0.99 × 0.10 = 0.099 < min existing 0.846 → rejected
        buf.add(_crop(), quality_score=0.99, ocr_conf=0.10, char_probs=_chars(0.10), frame_idx=3)

        assert len(buf.crops) == 2
        _, _, prob_lists = buf.top_k(k=2)
        for pl in prob_lists:
            assert all(abs(p - 0.93) < 1e-6 for _, p in pl)

    def test_top_k_ordered_by_combined_score(self):
        """top_k returns crops ranked by combined score descending."""
        buf = TrackBuffer(max_size=5)
        # Insert in reverse order of expected rank
        data = [
            (0.90, 0.93, 3),   # combined 0.837 — rank 2
            (0.95, 0.20, 1),   # combined 0.190 — rank 3 (worst)
            (0.91, 0.95, 4),   # combined 0.865 — rank 1 (best)
        ]
        for q, ocr_c, fidx in data:
            buf.add(_crop(), quality_score=q, ocr_conf=ocr_c, char_probs=_chars(ocr_c), frame_idx=fidx)

        crops, scores, prob_lists = buf.top_k(k=3)
        # Scores should be sorted descending
        assert scores[0] >= scores[1] >= scores[2]
        # Best entry: q=0.91 × ocr=0.95 = 0.865
        assert abs(scores[0] - 0.91 * 0.95) < 1e-4
        # Worst entry: q=0.95 × ocr=0.20 = 0.190
        assert abs(scores[2] - 0.95 * 0.20) < 1e-4

    def test_top_k_returns_prob_lists(self):
        """top_k third return value contains the cached char_probs for each crop."""
        buf = TrackBuffer(max_size=3)
        chars_a = [("3", 0.93), ("0", 0.93), ("G", 0.91)]
        chars_b = [("6", 0.45), ("0", 0.38)]
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.92, char_probs=chars_a, frame_idx=1)
        buf.add(_crop(), quality_score=0.90, ocr_conf=0.41, char_probs=chars_b, frame_idx=2)

        _, _, prob_lists = buf.top_k(k=2)
        # First (highest combined) should be chars_a
        assert prob_lists[0] == chars_a
        assert prob_lists[1] == chars_b

    def test_empty_char_probs_treated_as_low_conf(self):
        """A crop that produces no OCR output gets ocr_conf=0.1 (minimum penalty)."""
        buf = TrackBuffer(max_size=2)
        buf.add(_crop(), quality_score=0.92, ocr_conf=0.93, char_probs=_chars(0.93), frame_idx=1)
        # Empty char_probs → caller passes ocr_conf=0.1
        buf.add(_crop(), quality_score=0.99, ocr_conf=0.10, char_probs=[], frame_idx=2)

        assert len(buf.crops) == 2
        _, scores, _ = buf.top_k(k=2)
        # Good crop (0.92×0.93=0.856) must rank above empty crop (0.99×0.10=0.099)
        assert scores[0] > scores[1]
        assert abs(scores[0] - 0.92 * 0.93) < 1e-4
