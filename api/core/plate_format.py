"""Vietnamese license-plate format helpers shared by OCR fusion paths."""

from __future__ import annotations

import re

VN_PLATE_TEMPLATE_PATTERNS: tuple[str, ...] = (
    "DD-DDD[SEP]L",
    "DD-DDD[SEP]LD",
    "DD-DDD[SEP]LL",
    "DD-DDD[SEP]LL-DD",
    "DD-LD[SEP]DDD.DD",
    "DD-LD[SEP]DDDD",
    "DD-LD[SEP]LDDDD",
    "DD-LLD[SEP]DDD.DD",
    "DD-LLD[SEP]DDDD",
    "DD-LLD[SEP]DDDD.DD",
    "DD-LLLD[SEP]DDD.DD",
    "DD-LL[SEP]DDD-DD",
    "DD-LL[SEP]DDD.DD",
    "DD-LL[SEP]DDDD",
    "DDD-DDLL",
    "DDD-L-DDD",
    "DDD-L[SEP]DDD",
    "DDL-DDD.DD",
    "DDL-DDDD",
    "DDL-DDDD.DD",
    "DDLD[SEP]DDD.DD",
    "DDLD[SEP]DDDD",
    "DDLL-DDD.DD",
    "DDLL-DDDD",
    "DDLL[SEP]DDD.DD",
    "DDLL[SEP]DDDD",
    "DDLL[SEP]DDDD.DD",
    "DDLL[SEP]LDDLL",
    "DD[SEP]DDD-L",
    "DD[SEP]DDD-LD",
    "DD[SEP]DDD-LL",
    "DD[SEP]DDDL",
    "DL-LD[SEP]DDD.DD",
    "LL-DD-DD",
    "LL[SEP]DD-DD",
)

_TOKEN_REGEX = {
    "D": r"\d",
    "L": r"[A-ZĐ]",
    "A": r"[A-ZĐ0-9]",
}

def _template_to_normalized_regex(pattern: str, *, format_mode: str = "raw") -> str:
    if format_mode == "alnum":
        normalized_pattern = pattern.replace("[SEP]", "").replace("-", "").replace(".", "")
    else:
        normalized_pattern = pattern.replace("[SEP]", " ").replace(".", "")
    return "".join(_TOKEN_REGEX.get(token, re.escape(token)) for token in normalized_pattern)


def _compile_plate_regex(*, format_mode: str) -> re.Pattern[str]:
    return re.compile(
        r"^(?:"
        + "|".join(
            sorted(
                {
                    _template_to_normalized_regex(pattern, format_mode=format_mode)
                    for pattern in VN_PLATE_TEMPLATE_PATTERNS
                },
                key=lambda value: (-len(value), value),
            )
        )
        + r")$"
    )


VN_PLATE_RE = _compile_plate_regex(format_mode="raw")
VN_PLATE_ALNUM_RE = _compile_plate_regex(format_mode="alnum")


def chars_to_text(char_probs: list[tuple[str, float]]) -> str:
    return "".join(c for c, _ in char_probs)


def display_plate_text(text: str) -> str:
    return text.replace("[SEP]", " ")


def chars_to_display_text(char_probs: list[tuple[str, float]]) -> str:
    return display_plate_text(chars_to_text(char_probs))


def mean_confidence(char_probs: list[tuple[str, float]]) -> float:
    if not char_probs:
        return 0.0
    return sum(float(p) for _, p in char_probs) / len(char_probs)


def normalize_plate_text(text: str, *, format_mode: str = "raw") -> str:
    compact = " ".join(text.strip().upper().split())
    if format_mode == "alnum":
        no_sep = compact.replace("[SEP]", "")
        return re.sub(r"[^0-9A-ZĐ]", "", no_sep)
    if format_mode != "raw":
        raise ValueError("format_mode must be either 'raw' or 'alnum'")
    with_sep = compact.replace("[SEP]", " ").replace(".", "")
    return " ".join(with_sep.split())


def is_vn_plate_text(text: str, *, format_mode: str = "raw") -> bool:
    plate_re = VN_PLATE_ALNUM_RE if format_mode == "alnum" else VN_PLATE_RE
    return bool(plate_re.match(normalize_plate_text(text, format_mode=format_mode)))


def is_vn_plate_chars(char_probs: list[tuple[str, float]], *, format_mode: str = "raw") -> bool:
    return is_vn_plate_text(chars_to_text(char_probs), format_mode=format_mode)
