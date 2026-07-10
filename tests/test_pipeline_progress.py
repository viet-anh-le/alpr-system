"""Tests for ALPR progress event semantics."""
from __future__ import annotations

import pytest

from api.core.progress import make_progress_event


@pytest.mark.unit
def test_progress_event_uses_processed_count_not_absolute_source_frame():
    ev = make_progress_event(processed_frames=10, total_frames=30, source_frame=900)

    assert ev["frame"] == 10
    assert ev["total"] == 30
    assert ev["source_frame"] == 900
    assert ev["pct"] == pytest.approx(33.3)


@pytest.mark.unit
def test_progress_event_clamps_pct_at_100():
    ev = make_progress_event(processed_frames=35, total_frames=30, source_frame=930)

    assert ev["frame"] == 35
    assert ev["total"] == 30
    assert ev["pct"] == 100.0


@pytest.mark.unit
def test_progress_event_final_interval_progress_reaches_100():
    ev = make_progress_event(processed_frames=28, total_frames=30, complete=True)

    assert ev == {
        "type": "progress",
        "frame": 28,
        "total": 30,
        "pct": 100.0,
    }
