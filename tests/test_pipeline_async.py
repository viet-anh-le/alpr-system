from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _frame(h: int = 120, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _chars(text: str, conf: float = 0.95) -> list[tuple[str, float]]:
    return [(char, conf) for char in text]


def _compact_plate(text: str) -> str:
    return "".join(char for char in text if char.isalnum())


@pytest.mark.unit
def test_run_job_uses_async_pipeline_by_default(monkeypatch):
    from api.core import frame_source, pipeline, pipeline_async

    class FakeFileFrameSource:
        fps = 30.0
        total_frames = 0
        frame_size = (0, 0)

        def __init__(self, path: str) -> None:
            self.path = path

        def iter_frames(self):
            return iter(())

    calls: list[dict] = []

    def fake_process_frames_async(source, emit, models, **kwargs):
        calls.append({"source": source, "models": models, "kwargs": kwargs})
        return {"total_vehicles": 0, "processed_frames": 0}

    monkeypatch.setattr(frame_source, "FileFrameSource", FakeFileFrameSource)
    monkeypatch.setattr(pipeline_async, "process_frames_async", fake_process_frames_async)
    monkeypatch.setattr(pipeline, "_session_create", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_session_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.os, "unlink", lambda path: None)

    loop = asyncio.new_event_loop()
    try:
        pipeline.run_job(
            "input.mp4",
            "job_async",
            asyncio.Queue(),
            loop,
            MagicMock(),
            {"job_async": object()},
        )
    finally:
        loop.close()

    assert len(calls) == 1
    assert calls[0]["kwargs"]["session_id"] == "job_async"


@pytest.mark.unit
def test_process_frames_async_final_snapshot_includes_track_buffer(monkeypatch):
    """Final vehicle snapshots must preserve event detail buffer data."""
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

        def identity_fields(self, tid: int) -> dict:
            return {
                "id": tid,
                "recognition_id": tid,
                "vehicle_track_id": tid,
                "plate_track_id": None,
            }

        def cluster_results(self, tid: int) -> list[dict]:
            return []

    source = MagicMock()
    source.total_frames = 0
    source.fps = 30.0
    source.iter_frames.return_value = iter([])

    models = MagicMock()
    models.create_vehicle_tracker = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(pipeline_async, "WebTrackletManager", FakeTracker)

    events: list[dict] = []
    result = pipeline_async.process_frames_async(source, emit=events.append, models=models)

    vehicle_events = [event for event in events if event["type"] == "vehicle"]
    assert result["total_vehicles"] == 1
    assert vehicle_events[0]["track_buffer"] == [
        {"frame_index": 12, "quality_score": 0.91}
    ]


@pytest.mark.unit
def test_process_frames_async_does_not_finalise_active_buffered_track(monkeypatch):
    from api.core import pipeline_async
    from api.core.config import MIN_FRAMES_FOR_OCR
    from api.core.quality_router import PlateQualityRouter

    class FakeAssociator:
        def __init__(self, *args, **kwargs) -> None:
            self.vehicle_cache = {32: (0, 0, 180, 140)}

        def process_frame(self, plate_tracks, vehicle_tracks):
            return [(32, plate) for plate in plate_tracks]

    frames = [_frame() for _ in range(MIN_FRAMES_FOR_OCR)]
    source = MagicMock()
    source.total_frames = len(frames)
    source.fps = 30.0
    source.iter_frames.return_value = iter([(idx, frame, idx / 30.0) for idx, frame in enumerate(frames)])

    v_pred = MagicMock()
    v_pred.boxes = MagicMock()
    v_pred.boxes.__len__ = lambda self: 0

    models = MagicMock()
    models.device = torch.device("cpu")
    models.vehicle.predict.return_value = [v_pred]
    models.vehicle.names = {5: "motorcycle"}
    mock_tracker = MagicMock()
    mock_tracker.track.return_value = (
        np.array([[0, 0, 180, 140]], dtype=np.int32),
        np.array([32], dtype=np.int64),
        np.array([5], dtype=np.int32),
    )
    models.create_vehicle_tracker = MagicMock(return_value=mock_tracker)
    models.quality_router = PlateQualityRouter(classifier=lambda crop: {"poor": 0.96})

    finalise_calls: list[int] = []

    def fake_finalise(tid, *_args, **_kwargs):
        finalise_calls.append(tid)

    monkeypatch.setattr(pipeline_async, "FRAME_STRIDE", 1)
    monkeypatch.setattr(pipeline_async, "TrajectoryAssociator", FakeAssociator)
    monkeypatch.setattr(
        pipeline_async,
        "detect_plate_tracks_cascade",
        lambda *args, **kwargs: [
            {
                "id": 65,
                "crop": np.full((48, 96, 3), 77, dtype=np.uint8),
                "box": [10, 10, 70, 30],
            }
        ],
    )
    monkeypatch.setattr(pipeline_async, "_finalise_track_ocr", fake_finalise)
    monkeypatch.setattr(pipeline_async, "select_ocr_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline_async,
        "preprocess_plate_for_model",
        lambda *_args, **_kwargs: torch.zeros((3, 48, 96)),
    )
    monkeypatch.setattr(
        pipeline_async,
        "ocr_batch",
        lambda _model, tensors, _device: [(_chars("30A12345"), True) for _ in range(len(tensors))],
    )

    pipeline_async.process_frames_async(source, emit=lambda event: None, models=models)

    assert finalise_calls == [32]


@pytest.mark.unit
def test_process_frames_async_emits_preview_frame_with_tracked_boxes(monkeypatch):
    from api.core import pipeline_async

    frames = [_frame()]
    source = MagicMock()
    source.total_frames = len(frames)
    source.fps = 30.0
    source.iter_frames.return_value = iter(
        [(idx, frame, idx / 30.0) for idx, frame in enumerate(frames)]
    )

    v_pred = MagicMock()
    v_pred.boxes = MagicMock()
    v_pred.boxes.__len__ = lambda self: 0

    models = MagicMock()
    models.device = torch.device("cpu")
    models.vehicle.predict.return_value = [v_pred]
    models.vehicle.names = {5: "motorcycle"}
    mock_tracker = MagicMock()
    mock_tracker.track.return_value = (
        np.array([[5, 10, 105, 110]], dtype=np.int32),
        np.array([32], dtype=np.int64),
        np.array([5], dtype=np.int32),
    )
    models.create_vehicle_tracker = MagicMock(return_value=mock_tracker)

    monkeypatch.setattr(pipeline_async, "ALPR_PREVIEW_FPS", 30.0)
    monkeypatch.setattr(pipeline_async, "FRAME_STRIDE", 1)
    monkeypatch.setattr(pipeline_async, "detect_plate_tracks_cascade", lambda *args, **kwargs: [])

    events: list[dict] = []
    pipeline_async.process_frames_async(source, emit=events.append, models=models)

    frame_events = [event for event in events if event["type"] == "frame"]
    assert len(frame_events) == 1
    assert frame_events[0]["boxes"][0]["box"] == [5, 10, 105, 110]
    assert frame_events[0]["boxes"][0]["label"] == "motorcycle #32"


@pytest.mark.unit
def test_process_frames_async_uses_preprocessing_for_vehicle_only(monkeypatch):
    from api.core import pipeline_async

    class FakeAssociator:
        def __init__(self, *args, **kwargs) -> None:
            self.vehicle_cache = {32: (0, 0, 80, 60)}

        def process_frame(self, plate_tracks, vehicle_tracks):
            return [(32, plate) for plate in plate_tracks]

    class FakeRecorder:
        def __init__(self) -> None:
            self.frame_means: list[float] = []

        def record_frame(self, frame: np.ndarray) -> None:
            self.frame_means.append(float(frame.mean()))

        def finish(self) -> None:
            pass

    raw_frame = np.full((90, 120, 3), 24, dtype=np.uint8)
    source = MagicMock()
    source.total_frames = 1
    source.fps = 30.0
    source.iter_frames.return_value = iter([(0, raw_frame, 0.0)])

    v_pred = MagicMock()
    v_pred.boxes = MagicMock()
    v_pred.boxes.__len__ = lambda self: 0

    captured: dict[str, float] = {}

    def fake_vehicle_predict(frame, **_kwargs):
        captured["vehicle_predict_mean"] = float(frame.mean())
        return [v_pred]

    def fake_track(_dets, frame):
        captured["tracker_mean"] = float(frame.mean())
        return (
            np.array([[0, 0, 80, 60]], dtype=np.int32),
            np.array([32], dtype=np.int64),
            np.array([5], dtype=np.int32),
        )

    def fake_detect_plate(frame, *_args, **_kwargs):
        captured["plate_detect_mean"] = float(frame.mean())
        return [
            {
                "id": 65,
                "crop": frame[10:30, 20:70].copy(),
                "box": [20, 10, 70, 30],
            }
        ]

    def fake_prepare_ocr_jobs(matched, *_args, **_kwargs):
        _tid, plate_crop, vehicle_crop = matched[0]
        captured["plate_crop_mean"] = float(plate_crop.mean())
        captured["vehicle_crop_mean"] = float(vehicle_crop.mean())
        return [], {32}

    models = MagicMock()
    models.vehicle.predict.side_effect = fake_vehicle_predict
    models.vehicle.names = {5: "motorcycle"}
    models.create_vehicle_tracker = MagicMock(return_value=MagicMock(track=fake_track))
    recorder = FakeRecorder()

    monkeypatch.setattr(pipeline_async, "FRAME_STRIDE", 1)
    monkeypatch.setattr(pipeline_async, "TrajectoryAssociator", FakeAssociator)
    monkeypatch.setattr(pipeline_async, "detect_plate_tracks_cascade", fake_detect_plate)
    monkeypatch.setattr(pipeline_async, "prepare_route_ocr_jobs", fake_prepare_ocr_jobs)

    pipeline_async.process_frames_async(
        source,
        emit=lambda event: None,
        models=models,
        preprocess_mode="night",
        preprocessed_frame_recorder=recorder,
    )

    raw_mean = float(raw_frame.mean())
    assert captured["vehicle_predict_mean"] > raw_mean
    assert captured["tracker_mean"] > raw_mean
    assert recorder.frame_means[0] > raw_mean
    assert captured["plate_detect_mean"] == raw_mean
    assert captured["plate_crop_mean"] == raw_mean
    assert captured["vehicle_crop_mean"] == raw_mean
