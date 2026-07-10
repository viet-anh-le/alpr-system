from __future__ import annotations

import pytest


def _sample_for_template(pattern: str) -> str:
    digits = iter("12345678901234567890")
    letters = iter("PPABCDEFGHJKLMNQRSTUVXYZ")
    chars: list[str] = []
    idx = 0
    while idx < len(pattern):
        if pattern.startswith("[SEP]", idx):
            chars.append("[SEP]")
            idx += len("[SEP]")
            continue
        token = pattern[idx]
        if token == "D":
            chars.append(next(digits))
        elif token == "L":
            chars.append(next(letters))
        else:
            chars.append(token)
        idx += 1
    return "".join(chars)


def _char_probs(text: str, conf: float = 0.99) -> list[tuple[str, float]]:
    return [(char, conf) for char in text]


@pytest.mark.unit
def test_plate_format_accepts_all_ctm_templates() -> None:
    from api.core.ocr_ctm import OCR_DATASET_TEMPLATE_PATTERNS
    from api.core.plate_format import is_vn_plate_text

    invalid = [
        (pattern, _sample_for_template(pattern))
        for pattern in OCR_DATASET_TEMPLATE_PATTERNS
        if not is_vn_plate_text(_sample_for_template(pattern))
    ]

    assert invalid == []


@pytest.mark.unit
def test_plate_format_accepts_red_military_ll_dd_dd_template() -> None:
    from api.core.plate_format import is_vn_plate_text, normalize_plate_text

    assert normalize_plate_text("PP-10-39") == "PP-10-39"
    assert is_vn_plate_text("PP-10-39") is True


@pytest.mark.unit
def test_normalize_plate_text_renders_sep_as_space() -> None:
    from api.core.plate_format import is_vn_plate_text, normalize_plate_text

    assert normalize_plate_text("59-U1[SEP]027.95") == "59-U1 02795"
    assert normalize_plate_text("59-U1 [SEP] 027.95") == "59-U1 02795"
    assert is_vn_plate_text("59-U1[SEP]027.95") is True


@pytest.mark.unit
def test_alnum_plate_format_ignores_sep_dash_and_dot_template_literals() -> None:
    from api.core.plate_format import is_vn_plate_text, normalize_plate_text

    assert normalize_plate_text("59-U1[SEP]027.95", format_mode="alnum") == "59U102795"
    assert is_vn_plate_text("59U102795", format_mode="alnum") is True
    assert is_vn_plate_text("59U102795") is False


@pytest.mark.unit
def test_ctm_fusion_marks_red_military_template_valid() -> None:
    from api.core.ocr_ctm import fuse_ocr_outputs_ctm

    result = fuse_ocr_outputs_ctm([_char_probs("PP-10-39")])

    assert result.text == "PP-10-39"
    assert result.unresolved_slots == []
    assert result.is_valid is True
