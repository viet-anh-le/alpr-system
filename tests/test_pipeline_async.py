from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_process_frames_async_final_snapshot_includes_track_buffer(monkeypatch):
    """Final vehicle snapshots must preserve incident detail buffer data."""
    from api.core import pipeline_async

    class FakeTracker:
        def __init__(self) -> None:
            self._best = {7: [("3", 0.95)]}
            self._buffers = {}
            self._done = {7: True}
            self._cls = {7: "car"}
            self._plate_img_conf = {7: 0.88}

        def display_text(self, tid: int) -> str:
            return "30A-12345"

        def chars_json(self, tid: int) -> list[list[object]]:
            return [["3", 0.95]]

        def plate_b64(self, tid: int) -> str:
            return "plate-image"

        def vehicle_b64(self, tid: int) -> str:
            return "vehicle-image"

        def track_buffer_json(self, tid: int) -> list[dict]:
            return [{"frame_index": 12, "quality_score": 0.91}]

        def ocr_frames(self, tid: int) -> int:
            return 4

    source = MagicMock()
    source.total_frames = 0
    source.fps = 30.0
    source.iter_frames.return_value = iter([])

    models = MagicMock()
    models.vehicle_tracker.reset = MagicMock()

    monkeypatch.setattr(pipeline_async, "WebTrackletManager", FakeTracker)

    events: list[dict] = []
    result = pipeline_async.process_frames_async(source, emit=events.append, models=models)

    vehicle_events = [event for event in events if event["type"] == "vehicle"]
    assert result["total_vehicles"] == 1
    assert vehicle_events[0]["track_buffer"] == [
        {"frame_index": 12, "quality_score": 0.91}
    ]
