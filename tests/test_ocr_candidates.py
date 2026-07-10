from __future__ import annotations

import numpy as np
import pytest


def _chars(text: str, conf: float) -> list[tuple[str, float]]:
    return [(c, conf) for c in text]


@pytest.mark.unit
def test_candidate_reranker_prefers_valid_plate_when_confidence_is_comparable() -> None:
    from api.core.ocr_candidates import OcrCandidateResult, rerank_ocr_candidates

    best = rerank_ocr_candidates([
        OcrCandidateResult("sharpen", _chars("XXXXXXXX", 0.92)),
        OcrCandidateResult("original", _chars("30G-51827", 0.88)),
    ])

    assert best is not None
    assert best.text == "30G-51827"


@pytest.mark.unit
def test_candidate_reranker_keeps_original_when_enhancement_breaks_format() -> None:
    from api.core.ocr_candidates import OcrCandidateResult, rerank_ocr_candidates

    best = rerank_ocr_candidates([
        OcrCandidateResult("original", _chars("51G-12345", 0.91)),
        OcrCandidateResult("clahe", _chars("S1G1234S", 0.96)),
    ])

    assert best is not None
    assert best.method == "original"


@pytest.mark.unit
def test_candidate_crop_builder_adds_route_specific_transforms() -> None:
    from api.core.ocr_candidates import build_candidate_crops
    from api.core.quality_router import DegradationTags

    crop = np.full((48, 96, 3), 90, dtype=np.uint8)
    candidates = build_candidate_crops(
        crop,
        DegradationTags(motion_blur=True, low_light=True, low_contrast=True),
    )
    methods = [method for method, _ in candidates]

    assert methods[0] == "original"
    assert "sharpen" in methods
    assert "clahe" in methods
    assert "gamma" in methods
