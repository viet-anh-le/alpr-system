from __future__ import annotations

import pytest


def _chars(text: str, conf: float = 0.92) -> list[tuple[str, float]]:
    return [(c, conf) for c in text]


def _ocr_tokens(text: str, conf: float = 0.92) -> list[tuple[str, float]]:
    tokens: list[tuple[str, float]] = []
    i = 0
    while i < len(text):
        if text.startswith("[SEP]", i):
            tokens.append(("[SEP]", conf))
            i += len("[SEP]")
        else:
            tokens.append((text[i], conf))
            i += 1
    return tokens


def _ocr_label_pattern(text: str) -> str:
    parts: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("[SEP]", i):
            parts.append("[SEP]")
            i += len("[SEP]")
            continue
        char = text[i]
        if char.isdigit():
            parts.append("D")
        elif char.isalpha() or char == "Đ":
            parts.append("L")
        else:
            parts.append(char)
        i += 1
    return "".join(parts)


@pytest.mark.unit
def test_ctm_aligns_dashless_and_dashed_ocr_outputs() -> None:
    from api.core.ocr_ctm import fuse_ocr_outputs_ctm

    result = fuse_ocr_outputs_ctm([
        _chars("30G51827"),
        _chars("30G-51827"),
        _chars("30G51827"),
    ])

    assert result.text == "30G-51827"
    assert result.unresolved_slots == []
    assert result.is_valid is True


@pytest.mark.unit
def test_ctm_recovers_one_ambiguous_character_by_majority_support() -> None:
    from api.core.ocr_ctm import fuse_ocr_outputs_ctm

    result = fuse_ocr_outputs_ctm([
        _chars("30G-51827"),
        _chars("30G-51827"),
        _chars("30G-51B27"),
    ])

    assert result.text == "30G-51827"
    assert result.unresolved_slots == []
    assert result.ctm_support


@pytest.mark.unit
def test_ctm_marks_slot_unresolved_without_majority_support() -> None:
    from api.core.ocr_ctm import fuse_ocr_outputs_ctm

    result = fuse_ocr_outputs_ctm([
        _chars("30G-51821"),
        _chars("30G-51822"),
        _chars("30G-51823"),
    ])

    assert "?" in result.text
    assert result.unresolved_slots
    assert result.is_valid is False


@pytest.mark.unit
def test_ctm_ignores_empty_ocr_outputs() -> None:
    from api.core.ocr_ctm import fuse_ocr_outputs_ctm

    result = fuse_ocr_outputs_ctm([
        [],
        _chars("51G-12345"),
        _chars("51G12345"),
    ])

    assert result.text == "51G-12345"
    assert result.is_valid is True


@pytest.mark.unit
def test_ctm_preserves_sep_and_dot_tokens_for_two_line_plates() -> None:
    from api.core.ocr_ctm import fuse_ocr_outputs_ctm

    result = fuse_ocr_outputs_ctm([
        _ocr_tokens("59-U1[SEP]027.95"),
        _ocr_tokens("59-U1[SEP]027.95"),
        _ocr_tokens("59-U1[SEP]02795"),
    ])

    assert result.text == "59-U1[SEP]027.95"
    assert result.unresolved_slots == []
    assert result.is_valid is True


@pytest.mark.unit
def test_ctm_templates_cover_ocr_dataset_label_patterns() -> None:
    from pathlib import Path

    from api.core.ocr_ctm import TEMPLATES

    root = Path("data/datasets/ocr")
    if not root.exists():
        pytest.skip("OCR dataset is not available")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dataset_patterns = {
        _ocr_label_pattern(path.name.split("#", 1)[0])
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    }
    template_patterns = {"".join(template.pattern) for template in TEMPLATES}

    assert dataset_patterns <= template_patterns
