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
    "LD[SEP]DDD.DD",
    "LL-DD-DD",
    "LL[SEP]DD-DD",
)

_TOKEN_REGEX = {
    "D": r"\d",
    "L": r"[A-ZĐ]",
    "A": r"[A-ZĐ0-9]",
}


def _template_to_normalized_regex(pattern: str) -> str:
    normalized_pattern = pattern.replace("[SEP]", " ").replace(".", "")
    return "".join(_TOKEN_REGEX.get(token, re.escape(token)) for token in normalized_pattern)


VN_PLATE_RE = re.compile(
    r"^(?:"
    + "|".join(
        sorted(
            {_template_to_normalized_regex(pattern) for pattern in VN_PLATE_TEMPLATE_PATTERNS},
            key=lambda value: (-len(value), value),
        )
    )
    + r")$"
)


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


def normalize_plate_text(text: str) -> str:
    compact = " ".join(text.strip().upper().split())
    with_sep = compact.replace("[SEP]", " ").replace(".", "")
    return " ".join(with_sep.split())


def is_vn_plate_text(text: str) -> bool:
    return bool(VN_PLATE_RE.match(normalize_plate_text(text)))


def is_vn_plate_chars(char_probs: list[tuple[str, float]]) -> bool:
    return is_vn_plate_text(chars_to_text(char_probs))
