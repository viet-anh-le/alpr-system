from __future__ import annotations

import asyncio
import base64
from unittest.mock import MagicMock

import cv2
import numpy as np


def _char_probs(text: str, confidence: float = 0.95) -> list[tuple[str, float]]:
    return [(char, confidence) for char in text]


def _compact_plate(text: str) -> str:
    return "".join(char for char in text if char.isalnum())


def _solid_crop(value: int) -> np.ndarray:
    return np.full((20, 94, 3), value, dtype=np.uint8)


def _decode_jpeg_b64(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def _first_pixel_value(frame: dict) -> int:
    img = _decode_jpeg_b64(frame["image_b64"])
    return int(img[0, 0, 0])


def _build_mixed_cluster_tracker():
    from api.core.tracker import WebTrackletManager

    tid = 44
    tracker = WebTrackletManager()
    tracker._cls[tid] = "car"
    for frame_idx, quality, value in [(1, 0.95, 30), (2, 0.93, 31)]:
        tracker.buffer_crop(
            tid,
            _solid_crop(value),
            quality,
            0.95,
            _char_probs("30A-12345"),
            frame_idx,
        )
    for frame_idx, quality, value in [(3, 0.91, 150), (4, 0.89, 151)]:
        tracker.buffer_crop(
            tid,
            _solid_crop(value),
            quality,
            0.95,
            _char_probs("51F-99999"),
            frame_idx,
        )
    return tid, tracker


def test_cluster_event_includes_cluster_local_buffer_and_votes() -> None:
    from api.core.pipeline_core import _finalise_track_ocr

    tid, tracker = _build_mixed_cluster_tracker()
    events: list[dict] = []

    _finalise_track_ocr(tid, tracker, MagicMock(), events.append, "", None, None)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "vehicle"
    assert len(event["clusters"]) == 2

    clusters_by_plate = {
        _compact_plate(cluster["plate"]): cluster
        for cluster in event["clusters"]
    }
    assert set(clusters_by_plate) == {"30A12345", "51F99999"}

    cluster_a = clusters_by_plate["30A12345"]
    cluster_b = clusters_by_plate["51F99999"]
    assert {frame["frame_index"] for frame in cluster_a["track_buffer"]} == {1, 2}
    assert {frame["frame_index"] for frame in cluster_b["track_buffer"]} == {3, 4}
    assert {_first_pixel_value(frame) for frame in cluster_a["track_buffer"]} == {30, 31}
    assert {_first_pixel_value(frame) for frame in cluster_b["track_buffer"]} == {150, 151}
    assert {_compact_plate(text) for text in cluster_a["vote_summary"]} == {"30A12345"}
    assert {_compact_plate(text) for text in cluster_b["vote_summary"]} == {"51F99999"}


def test_record_save_persists_cluster_local_buffers_and_votes(monkeypatch) -> None:
    import api.core.pipeline as pipeline
    from api.core.pipeline_core import _finalise_track_ocr
    from api.database.models import RecognitionRecord

    tid, tracker = _build_mixed_cluster_tracker()
    events: list[dict] = []
    _finalise_track_ocr(tid, tracker, MagicMock(), events.append, "", None, None)

    captured: list[RecognitionRecord] = []

    async def fake_upsert_record(record):
        captured.append(record)

    class DoneFuture:
        def result(self, timeout=None):
            return None

    def fake_run_coroutine_threadsafe(coro, _loop):
        asyncio.run(coro)
        return DoneFuture()

    monkeypatch.setattr("api.database.mongodb.is_db_configured", lambda: True)
    monkeypatch.setattr("api.database.mongodb.upsert_record", fake_upsert_record)
    monkeypatch.setattr(pipeline, "_storage_upload", lambda _bucket, path, _data: path)
    monkeypatch.setattr(pipeline.asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    pipeline._record_save(
        "sess-cluster",
        tid,
        tracker,
        tracker._best[tid],
        "ocr_output_ctm",
        events[0]["vote_summary"],
        loop=object(),
        user_id="user-1",
    )

    assert len(captured) == 1
    record = captured[0]
    assert len(record.clusters) == 2
    clusters_by_plate = {
        _compact_plate(cluster.plate_text): cluster
        for cluster in record.clusters
    }
    assert set(clusters_by_plate) == {"30A12345", "51F99999"}
    assert {frame.frame_index for frame in clusters_by_plate["30A12345"].track_buffer} == {1, 2}
    assert {frame.frame_index for frame in clusters_by_plate["51F99999"].track_buffer} == {3, 4}
    assert {
        _compact_plate(text)
        for text in clusters_by_plate["30A12345"].ocr_vote_summary
    } == {"30A12345"}
    assert {
        _compact_plate(text)
        for text in clusters_by_plate["51F99999"].ocr_vote_summary
    } == {"51F99999"}
