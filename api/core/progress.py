"""Progress event helpers shared by ALPR pipelines."""
from __future__ import annotations


def make_progress_event(
    *,
    processed_frames: int,
    total_frames: int,
    source_frame: int | None = None,
    complete: bool = False,
) -> dict:
    processed = max(0, int(processed_frames))
    total = max(0, int(total_frames))
    denominator = max(total, processed, 1)
    if complete and processed > 0:
        pct = 100.0
    else:
        pct = min(100.0, max(0.0, round(processed / denominator * 100.0, 1)))

    event = {
        "type": "progress",
        "frame": processed,
        "total": total or processed,
        "pct": pct,
    }
    if source_frame is not None:
        event["source_frame"] = int(source_frame)
    return event
