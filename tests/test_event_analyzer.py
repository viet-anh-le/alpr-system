"""Tests for api/core/event_analyzer.py.

Uses a fake ModelBundle + the short_clip fixture so the analyzer's event
translation can be verified end-to-end without real GPU inference."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from api.core.frame_source import FileFrameSource


@pytest.mark.unit
def test_event_analyzer_uses_async_pipeline_by_default():
    """Event analyzer should use the same fast pipeline as upload jobs."""
    from api.core import event_analyzer
    from api.core.pipeline_async import process_frames_async

    assert event_analyzer.process_frames is process_frames_async


@pytest.mark.unit
def test_run_event_emits_started_and_complete(monkeypatch):
    """run_event wraps process_frames; verify it emits event_started
    and event_complete with the correct event_id, regardless of
    intermediate events."""
    from api.core import event_analyzer

    events: list[dict] = []
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def fake_process_frames(source, emit, models, **kwargs):
        # Mimic the real pipeline_core: emit a vehicle event then return a summary
        emit({"type": "vehicle", "id": 7, "plate": "30A-12345",
              "chars": [["3", 0.99]], "cls": "car",
              "plate_b64": "", "vehicle_b64": "", "ocr_frames": 5})
        return {"total_vehicles": 1, "processed_frames": 30}

    monkeypatch.setattr(event_analyzer, "process_frames", fake_process_frames)
    monkeypatch.setattr(event_analyzer, "_persist_event", lambda *a, **kw: None)

    async def drain() -> None:
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.2)
                events.append(ev)
            except asyncio.TimeoutError:
                return

    event_analyzer.run_event(
        event_id="evt_test",
        session_id="ses_test",
        source=FileFrameSource("tests/fixtures/short_clip.mp4"),
        source_type="upload",
        source_ref="short_clip.mp4",
        window_start_sec=0.0,
        window_end_sec=1.0,
        queue=queue,
        loop=loop,
        models=MagicMock(),
    )
    loop.run_until_complete(drain())

    types = [e["type"] for e in events]
    assert "event_started" in types
    assert "event_vehicle" in types
    assert "event_complete" in types
    assert all(e.get("event_id") == "evt_test" for e in events
               if e["type"].startswith("event_"))


@pytest.mark.unit
def test_run_event_translates_vehicle_event(monkeypatch):
    from api.core import event_analyzer

    events: list[dict] = []
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def fake_process_frames(source, emit, models, **kwargs):
        emit({"type": "vehicle", "id": 1, "plate": "30A-12345",
              "chars": [["3", 0.9]], "cls": "car",
              "plate_b64": "", "vehicle_b64": "", "ocr_frames": 3})
        emit({"type": "rejected_vehicle", "id": 2, "plate": "????",
              "chars": [], "cls": "motorcycle",
              "plate_b64": "", "vehicle_b64": "", "ocr_frames": 1,
              "vote_summary": {}})
        return {"total_vehicles": 1, "processed_frames": 30}

    monkeypatch.setattr(event_analyzer, "process_frames", fake_process_frames)
    monkeypatch.setattr(event_analyzer, "_persist_event", lambda *a, **kw: None)

    event_analyzer.run_event(
        event_id="evt_xlate",
        session_id="ses",
        source=FileFrameSource("tests/fixtures/short_clip.mp4"),
        source_type="upload",
        source_ref="x",
        window_start_sec=0.0,
        window_end_sec=1.0,
        queue=queue,
        loop=loop,
        models=MagicMock(),
    )

    async def drain():
        while True:
            try:
                events.append(await asyncio.wait_for(queue.get(), timeout=0.2))
            except asyncio.TimeoutError:
                return
    loop.run_until_complete(drain())

    veh = [e for e in events if e["type"] == "event_vehicle"]
    rej = [e for e in events if e["type"] == "event_rejected_vehicle"]
    assert len(veh) == 1 and veh[0]["plate"] == "30A-12345"
    assert len(rej) == 1


@pytest.mark.unit
def test_run_event_emits_error_event_on_exception(monkeypatch):
    """Lines 192-214: run_event error path when process_frames raises."""
    from api.core import event_analyzer

    events: list[dict] = []
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def exploding_process(source, emit, models, **kwargs):
        raise RuntimeError("GPU exploded")

    monkeypatch.setattr(event_analyzer, "process_frames", exploding_process)
    monkeypatch.setattr(event_analyzer, "_persist_event", lambda *a, **kw: None)

    event_analyzer.run_event(
        event_id="evt_err",
        session_id="ses_err",
        source=MagicMock(total_frames=0),
        source_type="upload",
        source_ref="x.mp4",
        window_start_sec=0.0,
        window_end_sec=1.0,
        queue=queue,
        loop=loop,
        models=MagicMock(),
    )

    async def drain():
        while True:
            try:
                events.append(await asyncio.wait_for(queue.get(), timeout=0.2))
            except asyncio.TimeoutError:
                return

    loop.run_until_complete(drain())
    loop.close()

    error_events = [e for e in events if e["type"] == "event_error"]
    assert len(error_events) == 1
    assert "GPU exploded" in error_events[0]["message"]
    assert error_events[0]["event_id"] == "evt_err"


@pytest.mark.unit
def test_run_event_accepts_monitor_ocr_backend(monkeypatch):
    """Monitor dispatch passes ocr_backend; it must not crash before SSE events."""
    from api.core import event_analyzer

    events: list[dict] = []
    captured: dict[str, str] = {}
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def fake_process_frames(source, emit, models, **kwargs):
        captured["ocr_backend"] = kwargs.get("ocr_backend")
        emit({"type": "progress", "frame": 1, "total": 1, "pct": 100.0})
        return {"total_vehicles": 0, "processed_frames": 1}

    monkeypatch.setattr(event_analyzer, "process_frames", fake_process_frames)
    monkeypatch.setattr(event_analyzer, "_persist_event", lambda *a, **kw: None)

    event_analyzer.run_event(
        event_id="evt_backend",
        session_id="ses_backend",
        source=MagicMock(total_frames=1),
        source_type="upload",
        source_ref="x.mp4",
        window_start_sec=0.0,
        window_end_sec=1.0,
        queue=queue,
        loop=loop,
        models=MagicMock(),
        ocr_backend="smalllpr_line_ctc",
    )

    async def drain():
        while True:
            try:
                events.append(await asyncio.wait_for(queue.get(), timeout=0.2))
            except asyncio.TimeoutError:
                return

    loop.run_until_complete(drain())
    loop.close()

    assert captured["ocr_backend"] == "smalllpr_line_ctc"
    assert [ev["type"] for ev in events] == [
        "event_started",
        "event_progress",
        "event_complete",
    ]


@pytest.mark.unit
def test_run_event_passes_timings_when_debug_enabled(monkeypatch, caplog):
    from api.core import event_analyzer

    captured: dict[str, dict | None] = {}
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def fake_process_frames(source, emit, models, **kwargs):
        timings = kwargs.get("timings")
        captured["timings"] = timings
        if timings is not None:
            timings["total"] = 0.1234
        return {"total_vehicles": 0, "processed_frames": 1}

    monkeypatch.setattr(event_analyzer, "ALPR_DEBUG_TIMINGS", True)
    monkeypatch.setattr(event_analyzer, "process_frames", fake_process_frames)
    monkeypatch.setattr(event_analyzer, "_persist_event", lambda *a, **kw: None)
    caplog.set_level("INFO", logger="api.core.event_analyzer")

    try:
        event_analyzer.run_event(
            event_id="evt_timing",
            session_id="ses_timing",
            source=MagicMock(total_frames=1),
            source_type="upload",
            source_ref="x.mp4",
            window_start_sec=0.0,
            window_end_sec=1.0,
            queue=queue,
            loop=loop,
            models=MagicMock(),
        )
    finally:
        loop.close()

    assert captured["timings"] == {"total": 0.1234}
    assert "Event timings event=evt_timing" in caplog.text


@pytest.mark.unit
def test_persist_event_builds_event_document_and_calls_upsert(monkeypatch):
    """Lines 48-114: _persist_event when DB is configured — builds EventVehicle docs."""
    import api.database.mongodb as mongodb_mod

    monkeypatch.setattr(mongodb_mod, "is_db_configured", lambda: True)

    upserted = []

    async def fake_upsert(event):
        upserted.append(event)

    monkeypatch.setattr(mongodb_mod, "upsert_event", fake_upsert)

    from api.core import event_analyzer
    from datetime import datetime, timezone

    loop = asyncio.new_event_loop()
    now = datetime.now(timezone.utc)

    vehicles = [
        {
            "id": 3,
            "plate": "51G-11111",
            "chars": [["5", 0.95], ["1", 0.92]],
            "cls": "car",
            "plate_b64": "",
            "vehicle_b64": "",
            "ocr_frames": 8,
        }
    ]

    event_analyzer._persist_event(
        event_id="evt_db_test",
        session_id="ses_db_test",
        source_type="upload",
        source_ref="clip.mp4",
        window_start_sec=0.0,
        window_end_sec=5.0,
        status="completed",
        vehicles=vehicles,
        rejected=[],
        processing_ms=300,
        error_message=None,
        marked_at=now,
        loop=loop,
    )

    # run_coroutine_threadsafe schedules the upsert coroutine on the loop
    loop.run_until_complete(asyncio.sleep(0.05))
    loop.close()

    assert len(upserted) == 1
    assert upserted[0].event_id == "evt_db_test"
    assert len(upserted[0].vehicles) == 1
    assert upserted[0].vehicles[0].plate_text == "51G-11111"
