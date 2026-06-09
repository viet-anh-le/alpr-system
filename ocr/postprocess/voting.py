from __future__ import annotations

from collections import Counter


def majority_vote(candidates: list[str]) -> str:
    """Select most frequent plate string from per-frame candidates."""
    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]


def weighted_vote(candidates: list[tuple[str, float]]) -> str:
    """Select plate string with highest cumulative confidence."""
    scores: dict[str, float] = {}
    for text, conf in candidates:
        scores[text] = scores.get(text, 0.0) + conf
    return max(scores, key=scores.get) if scores else ""
