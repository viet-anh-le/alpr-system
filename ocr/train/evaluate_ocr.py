from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def character_accuracy(preds: list[str], targets: list[str]) -> float:
    total = sum(len(t) for t in targets)
    if total == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(preds, targets, strict=False))
    return correct / len(targets)


def plate_accuracy(preds: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(preds, targets, strict=False)) / len(targets)
