from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrackResult:
    track_id: int
    plate_text: str
    confidence: float
    frame_count: int
    candidates: list[str] = field(default_factory=list)


class ResultAggregator:
    """Aggregates per-frame OCR results per track into final plate strings."""

    def __init__(self) -> None:
        self._tracks: dict[int, list[tuple[str, float]]] = {}

    def add(self, track_id: int, plate_text: str, confidence: float) -> None:
        self._tracks.setdefault(track_id, []).append((plate_text, confidence))

    def finalize(self) -> list[TrackResult]:
        from ocr.postprocess.voting import weighted_vote
        results = []
        for track_id, candidates in self._tracks.items():
            plate = weighted_vote(candidates)
            avg_conf = sum(c for _, c in candidates) / len(candidates)
            results.append(TrackResult(track_id=track_id, plate_text=plate, confidence=avg_conf, frame_count=len(candidates)))
        return results
