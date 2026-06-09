from __future__ import annotations

import re

VN_PLATE_PATTERN = re.compile(r"^\d{2}[A-Z]-?\d{4,5}$|^\d{2}[A-Z]{2}-?\d{4,5}$")

CHAR_CORRECTIONS: dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
}


def correct_plate(text: str) -> str:
    """Apply rule-based corrections for Vietnamese plate format."""
    raise NotImplementedError
