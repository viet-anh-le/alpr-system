from __future__ import annotations

import pytest


def _chars(text: str, conf: float = 0.94) -> list[tuple[str, float]]:
    return [(c, conf) for c in text]


def _ocr_tokens(text: str, conf: float = 0.94) -> list[tuple[str, float]]:
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


def _text(chars: list[tuple[str, float]]) -> str:
    return "".join(char for char, _ in chars)


@pytest.mark.unit
def test_slot_aware_correction_converts_letters_in_digit_slots() -> None:
    from api.core.ocr_ambiguity import correct_ambiguous_chars

    result = correct_ambiguous_chars(_chars("30G-51B27"))

    assert _text(result.char_probs) == "30G-51827"
    assert result.changed_positions == [6]
    assert result.risk_penalty > 0


@pytest.mark.unit
def test_slot_aware_correction_converts_digits_in_letter_slots() -> None:
    from api.core.ocr_ambiguity import correct_ambiguous_chars

    result = correct_ambiguous_chars(_chars("308-51827"))

    assert _text(result.char_probs) == "30B-51827"
    assert result.changed_positions == [2]


@pytest.mark.unit
def test_slot_aware_correction_preserves_valid_serial_letters() -> None:
    from api.core.ocr_ambiguity import correct_ambiguous_chars

    result = correct_ambiguous_chars(_chars("30B-51827"))

    assert _text(result.char_probs) == "30B-51827"
    assert result.changed_positions == []


@pytest.mark.unit
def test_slot_aware_correction_handles_sep_two_line_numbers() -> None:
    from api.core.ocr_ambiguity import correct_ambiguous_chars

    result = correct_ambiguous_chars(_ocr_tokens("59-U1[SEP]O27.9S"))

    assert _text(result.char_probs) == "59-U1[SEP]027.95"
    assert result.changed_positions


@pytest.mark.unit
def test_slot_aware_correction_uses_alnum_templates_without_format_literals() -> None:
    from api.core.ocr_ambiguity import correct_ambiguous_chars

    result = correct_ambiguous_chars(_chars("59U10279S"), format_mode="alnum")

    assert _text(result.char_probs) == "59U102795"
    assert result.changed_positions
