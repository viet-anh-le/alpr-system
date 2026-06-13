# Real-Time Multi-Source Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live RTSP multi-stream processing alongside the existing offline upload flow, using a crop-count OCR trigger instead of waiting for track loss or video end.

**Architecture:** A new `live_pipeline.py` mirrors `pipeline.py`'s frame loop but fires OCR as soon as `MIN_FRAMES_FOR_OCR` high-quality crops are buffered per track. `StreamManager` holds N `LiveSession` instances (one thread each). Live sessions use WebSocket (bidirectional) instead of SSE; offline upload keeps its existing SSE path untouched.

**Tech Stack:** Python 3.10+, FastAPI WebSocket, OpenCV (RTSP via `cv2.VideoCapture`), React + native `WebSocket` API, Tailwind CSS.

---

## File Map

```
CREATED
  api/core/live_pipeline.py          — run_live_stream() with crop-count OCR trigger
  api/core/stream_manager.py         — LiveSession dataclass + StreamManager class
  tests/api/conftest.py              — sys.path fixture so tests can import core.*
  tests/api/test_live_pipeline.py    — unit tests for trigger logic
  tests/api/test_stream_manager.py   — unit tests for StreamManager
  web/src/hooks/useStreamSession.js  — WebSocket hook for live sessions
  web/src/components/StreamCard.jsx  — per-session status card
  web/src/components/LiveStreamTab.jsx — tab container for live streams

MODIFIED
  api/core/config.py                 — add MAX_RECONNECT_ATTEMPTS, MAX_RECONNECT_DELAY_S
  api/database/models.py             — add stream_url field, extend status/source_type Literals
  api/main.py                        — add /streams POST/DELETE/GET + WS /streams/{id}/ws
  web/vite.config.js                 — proxy /streams with ws:true
  web/src/App.jsx                    — add tab switcher, import LiveStreamTab

UNCHANGED
  api/core/pipeline.py               — offline flow; functions imported by live_pipeline
  api/core/tracker.py
  api/core/association.py
  api/core/models.py
  api/core/gates.py
  api/core/quality_scorer.py
  api/core/video_processor.py
```

---

## Task 1: Config constants

**Files:**
- Modify: `api/core/config.py`

- [ ] **Step 1: Add constants after `LOST_THRESHOLD`**

Open `api/core/config.py`. After the line `LOST_THRESHOLD = 5`, add:

```python
MAX_RECONNECT_ATTEMPTS = 10   # retries before a live stream is marked as error
MAX_RECONNECT_DELAY_S  = 30   # exponential backoff ceiling in seconds
```

- [ ] **Step 2: Verify import works**

```bash
cd api && python -c "from core.config import MAX_RECONNECT_ATTEMPTS, MAX_RECONNECT_DELAY_S; print(MAX_RECONNECT_ATTEMPTS, MAX_RECONNECT_DELAY_S)"
```

Expected output: `10 30`

- [ ] **Step 3: Commit**

```bash
git add api/core/config.py
git commit -m "feat(config): add MAX_RECONNECT_ATTEMPTS and MAX_RECONNECT_DELAY_S"
```

---

## Task 2: Database model — live stream fields

**Files:**
- Modify: `api/database/models.py`

- [ ] **Step 1: Update `RecognitionSession`**

In `api/database/models.py`, replace the `RecognitionSession` class body with:

```python
class RecognitionSession(BaseModel):
    """
    One video-processing job submitted via POST /upload or a live stream session.

    Collection: recognition_sessions
    Unique index: session_id
    """

    id: PyObjectId | None = Field(default=None, alias="_id")
    session_id: str
    source_filename: str
    source_type: Literal["video", "image_dir", "rtsp", "live_stream"] = "video"
    stream_url: str | None = None
    status: Literal["queued", "processing", "completed", "failed", "stopped"] = "queued"
    total_records: int = 0
    processed_frames: int = 0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}
```

- [ ] **Step 2: Verify model serializes correctly**

```bash
cd api && python -c "
from database.models import RecognitionSession
s = RecognitionSession(session_id='abc', source_filename='rtsp://cam', source_type='live_stream', stream_url='rtsp://cam', status='stopped')
print(s.model_dump()['source_type'], s.model_dump()['status'], s.model_dump()['stream_url'])
"
```

Expected output: `live_stream stopped rtsp://cam`

- [ ] **Step 3: Commit**

```bash
git add api/database/models.py
git commit -m "feat(db): add stream_url field and live_stream/stopped values to RecognitionSession"
```

---

## Task 3: Test infrastructure + live pipeline unit tests

**Files:**
- Create: `tests/api/conftest.py`
- Create: `tests/api/test_live_pipeline.py`

- [ ] **Step 1: Create conftest so imports resolve**

Create `tests/api/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from core.tracker import ...` in tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "api"))
```

- [ ] **Step 2: Write failing tests**

Create `tests/api/test_live_pipeline.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from core.config import LOST_THRESHOLD, MIN_FRAMES_FOR_OCR
from core.tracker import WebTrackletManager


def _blank_crop() -> np.ndarray:
    return np.zeros((48, 96, 3), dtype=np.uint8)


class TestCropCountTrigger:
    def test_not_ready_before_threshold(self):
        tracker = WebTrackletManager()
        crop = _blank_crop()
        for i in range(MIN_FRAMES_FOR_OCR - 1):
            tracker.buffer_crop(1, crop, 0.8, i)
            assert not tracker.ready_for_multiframe_ocr(1)

    def test_ready_at_threshold(self):
        tracker = WebTrackletManager()
        crop = _blank_crop()
        for i in range(MIN_FRAMES_FOR_OCR):
            tracker.buffer_crop(1, crop, 0.8, i)
        assert tracker.ready_for_multiframe_ocr(1)

    def test_ready_above_threshold(self):
        tracker = WebTrackletManager()
        crop = _blank_crop()
        for i in range(MIN_FRAMES_FOR_OCR + 5):
            tracker.buffer_crop(1, crop, 0.8, i)
        assert tracker.ready_for_multiframe_ocr(1)


class TestDoneFlagPreventsRetrigger:
    def test_should_ocr_true_initially(self):
        tracker = WebTrackletManager()
        assert tracker.should_ocr(99)

    def test_should_ocr_false_after_done(self):
        tracker = WebTrackletManager()
        tracker._done[1] = True
        assert not tracker.should_ocr(1)

    def test_tracks_are_independent(self):
        tracker = WebTrackletManager()
        tracker._done[1] = True
        assert tracker.should_ocr(2)


class TestTrackLossFallback:
    def test_mark_lost_below_threshold(self):
        tracker = WebTrackletManager()
        for _ in range(LOST_THRESHOLD - 1):
            assert not tracker.mark_lost(1)

    def test_mark_lost_at_threshold(self):
        tracker = WebTrackletManager()
        for _ in range(LOST_THRESHOLD):
            result = tracker.mark_lost(1)
        assert result is True

    def test_reset_lost_clears_counter(self):
        tracker = WebTrackletManager()
        for _ in range(LOST_THRESHOLD - 1):
            tracker.mark_lost(1)
        tracker.reset_lost(1)
        # After reset, needs LOST_THRESHOLD more misses to fire
        for _ in range(LOST_THRESHOLD - 1):
            assert not tracker.mark_lost(1)
        assert tracker.mark_lost(1)
```

- [ ] **Step 3: Run tests — expect PASS (tracker logic already exists)**

```bash
cd /home/vietanh/Documents/DATN/ALPR_Vietnamese
pytest tests/api/test_live_pipeline.py -v
```

Expected: all 9 tests PASS (the tracker module is already implemented; we are verifying the logic our live pipeline will rely on).

- [ ] **Step 4: Commit**

```bash
git add tests/api/conftest.py tests/api/test_live_pipeline.py
git commit -m "test(live-pipeline): add unit tests for crop-count trigger and track-loss fallback"
```

---

## Task 4: Live pipeline implementation

**Files:**
- Create: `api/core/live_pipeline.py`

- [ ] **Step 1: Create the file**

Create `api/core/live_pipeline.py`:

```python
"""
core/live_pipeline.py — Live RTSP stream processing pipeline.

OCR trigger: crop-count threshold (MIN_FRAMES_FOR_OCR) fires while the
vehicle is still in frame. This replaces the track-loss / video-end triggers
used by the offline pipeline.py.

WS event types emitted:
  "connected"        — stream opened successfully
  "heartbeat"        — frame_count + active_tracks (every 10 frames)
  "vehicle"          — plate recognized (same shape as offline)
  "rejected_vehicle" — invalid plate format
  "reconnecting"     — stream dropped, retrying
  "stopped"          — clean shutdown
  "error"            — unrecoverable failure
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
from pathlib import Path as _Path

import cv2
import numpy as np

from .config import (
    FRAME_STRIDE,
    MAX_RECONNECT_ATTEMPTS,
    MAX_RECONNECT_DELAY_S,
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
    PLATE_PAD,
    VEHICLE_CLASSES,
)
from .gates import is_sharp
from .models import ModelBundle
from .pipeline import _run_multiframe_ocr, _session_update
from .quality_scorer import quality_score
from .tracker import WebTrackletManager
from .association import TrajectoryAssociator
from .video_processor import (
    crop_vehicle as _crop_vehicle,
    draw_annotated_frame as _draw_annotated_frame,
)

logger = logging.getLogger(__name__)

_BOTSORT_CFG = str(
    _Path(__file__).resolve().parents[2] / "configs/tracking/botsort_reid.yaml"
)
_PLATE_TRACKER_CFG = str(
    _Path(__file__).resolve().parents[2] / "configs/tracking/bytetrack_plate.yaml"
)


def _safe_put(q: asyncio.Queue, item: object) -> None:
    if not q.full():
        q.put_nowait(item)


def _live_session_create(
    stream_id: str, url: str, loop: asyncio.AbstractEventLoop
) -> None:
    try:
        from database.mongodb import is_db_configured, upsert_session
        from database.models import RecognitionSession

        if not is_db_configured():
            return
        session = RecognitionSession(
            session_id=stream_id,
            source_filename=url,
            source_type="live_stream",
            stream_url=url,
            status="processing",
        )
        asyncio.run_coroutine_threadsafe(upsert_session(session), loop).result(
            timeout=5
        )
    except Exception:
        logger.exception("MongoDB: failed to create live session %s", stream_id)


def run_live_stream(
    url: str,
    stream_id: str,
    event_queue: asyncio.Queue,
    mjpeg_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    models: ModelBundle,
    stop_event: threading.Event,
) -> None:
    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    def emit_frame(jpg: bytes) -> None:
        loop.call_soon_threadsafe(_safe_put, mjpeg_queue, jpg)

    _live_session_create(stream_id, url, loop)

    attempt = 0
    frame_idx = 0

    try:
        while not stop_event.is_set():
            cap = cv2.VideoCapture(url)

            if not cap.isOpened():
                attempt += 1
                if attempt > MAX_RECONNECT_ATTEMPTS:
                    emit({
                        "type": "error",
                        "message": f"Cannot open stream after {attempt} attempts",
                    })
                    _session_update(
                        stream_id, {"status": "failed"}, loop
                    )
                    return
                delay = min(2 ** attempt, MAX_RECONNECT_DELAY_S)
                emit({"type": "reconnecting", "attempt": attempt})
                if stop_event.wait(timeout=delay):
                    break
                continue

            attempt = 0
            emit({"type": "connected"})
            _session_update(stream_id, {"status": "processing"}, loop)

            tracker = WebTrackletManager()
            associator = TrajectoryAssociator(match_frames=5, agreement_ratio=0.6)
            previously_tracked: set[int] = set()

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # ── Vehicle tracking (every frame) ────────────────────────────
                v_res = models.vehicle.track(
                    frame,
                    persist=True,
                    tracker=_BOTSORT_CFG,
                    classes=VEHICLE_CLASSES,
                    verbose=False,
                )[0]

                tracked: list[dict] = []
                currently_tracked: set[int] = set()
                if v_res.boxes.id is not None:
                    boxes = v_res.boxes.xyxy.cpu().numpy().astype(int)
                    ids = v_res.boxes.id.cpu().numpy().astype(int)
                    clss = v_res.boxes.cls.cpu().numpy().astype(int)
                    for box, tid, cid in zip(boxes, ids, clss):
                        tid = int(tid)
                        tracker._cls[tid] = models.vehicle.names[int(cid)]
                        tracked.append({"id": tid, "box": box})
                        currently_tracked.add(tid)
                        if tid in tracker._lost_count:
                            tracker.reset_lost(tid)

                # ── Heartbeat every 10 frames ─────────────────────────────────
                if frame_idx % 10 == 0:
                    emit({
                        "type": "heartbeat",
                        "frame_count": frame_idx,
                        "active_tracks": len(currently_tracked),
                    })

                if frame_idx % FRAME_STRIDE != 0:
                    previously_tracked = currently_tracked
                    continue

                # ── Track loss fallback (vehicle left before threshold) ────────
                for tid in previously_tracked - currently_tracked:
                    if (
                        tracker.should_ocr(tid)
                        and tracker.mark_lost(tid)
                        and tracker.ready_for_multiframe_ocr(tid)
                    ):
                        _run_multiframe_ocr(
                            tid, tracker, models, emit,
                            session_id=stream_id, loop=loop,
                        )

                # ── Plate detection ───────────────────────────────────────────
                p_res = models.plate.track(
                    frame, persist=True, tracker=_PLATE_TRACKER_CFG, verbose=False
                )[0]

                active_tids: set[int] = set()
                plate_tracks: list[dict] = []

                if p_res.obb is not None and p_res.obb.id is not None:
                    H, W = frame.shape[:2]
                    obb_pts = p_res.obb.xyxyxyxy.cpu().numpy().astype(int)
                    obb_conf = p_res.obb.conf.cpu().numpy()
                    obb_ids = p_res.obb.id.cpu().numpy().astype(int)

                    for pts, det_conf, p_tid in zip(obb_pts, obb_conf, obb_ids):
                        if float(det_conf) < PLATE_DET_CONF:
                            continue
                        raw_rx, raw_ry, raw_rw, raw_rh = cv2.boundingRect(pts)
                        if raw_rw < MIN_PLATE_W or raw_rh < MIN_PLATE_H:
                            continue
                        rx = max(0, raw_rx - PLATE_PAD)
                        ry = max(0, raw_ry - PLATE_PAD)
                        rw = min(raw_rw + 2 * PLATE_PAD, W - rx)
                        rh = min(raw_rh + 2 * PLATE_PAD, H - ry)
                        plate_crop = frame[ry: ry + rh, rx: rx + rw]
                        if plate_crop.size == 0:
                            continue
                        if not is_sharp(plate_crop):
                            continue
                        plate_tracks.append({
                            "id": int(p_tid),
                            "box": [rx, ry, rx + rw, ry + rh],
                            "crop": plate_crop,
                            "conf": float(det_conf),
                        })

                firm_matches = associator.process_frame(plate_tracks, tracked)

                for v_tid, p in firm_matches:
                    v_box = associator.vehicle_cache.get(v_tid)
                    if v_box is None or not tracker.should_ocr(v_tid):
                        continue
                    vehicle_crop = _crop_vehicle(frame, v_box)
                    q = quality_score(p["crop"])
                    tracker.buffer_crop(v_tid, p["crop"], q, frame_idx)
                    tracker.update_vehicle_img(v_tid, vehicle_crop, q)
                    active_tids.add(v_tid)

                    # ── Crop-count OCR trigger (live-stream specific) ──────────
                    if tracker.ready_for_multiframe_ocr(v_tid):
                        _run_multiframe_ocr(
                            v_tid, tracker, models, emit,
                            session_id=stream_id, loop=loop,
                        )

                # ── MJPEG frame ───────────────────────────────────────────────
                box_dicts = [
                    {
                        "id": v["id"],
                        "box": [int(c) for c in v["box"]],
                        "state": (
                            "active" if v["id"] in active_tids
                            else "done" if tracker._done.get(v["id"])
                            else "tracked"
                        ),
                        "plate": tracker.display_text(v["id"]) or "",
                        "cls": tracker._cls.get(v["id"], "vehicle"),
                    }
                    for v in tracked
                ]
                emit_frame(_draw_annotated_frame(frame, box_dicts))

                previously_tracked = currently_tracked

                if frame_idx % 90 == 0:
                    gc.collect()

            cap.release()

            if stop_event.is_set():
                break

            # Stream dropped — reconnect with backoff
            attempt += 1
            if attempt > MAX_RECONNECT_ATTEMPTS:
                emit({
                    "type": "error",
                    "message": "Stream lost; max reconnection attempts reached",
                })
                _session_update(
                    stream_id,
                    {"status": "failed", "processed_frames": frame_idx},
                    loop,
                )
                return

            delay = min(2 ** attempt, MAX_RECONNECT_DELAY_S)
            emit({"type": "reconnecting", "attempt": attempt})
            stop_event.wait(timeout=delay)

    except Exception as exc:
        import traceback

        emit({"type": "error", "message": str(exc), "detail": traceback.format_exc()})
        _session_update(
            stream_id, {"status": "failed", "error_message": str(exc)}, loop
        )
        return

    # Clean stop
    emit({"type": "stopped"})
    _session_update(
        stream_id, {"status": "stopped", "processed_frames": frame_idx}, loop
    )
    loop.call_soon_threadsafe(_safe_put, mjpeg_queue, None)
```

- [ ] **Step 2: Verify imports resolve (no runtime errors)**

```bash
cd api && python -c "from core.live_pipeline import run_live_stream; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
git add api/core/live_pipeline.py
git commit -m "feat(live-pipeline): add run_live_stream with crop-count OCR trigger and RTSP reconnection"
```

---

## Task 5: StreamManager tests

**Files:**
- Create: `tests/api/test_stream_manager.py`

- [ ] **Step 1: Write failing tests**

Create `tests/api/test_stream_manager.py`:

```python
from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from core.stream_manager import LiveSession, StreamManager


def _make_manager() -> StreamManager:
    models = MagicMock()
    loop = asyncio.new_event_loop()
    return StreamManager(models=models, loop=loop)


class TestStart:
    def test_returns_live_session(self):
        mgr = _make_manager()
        with patch("core.stream_manager.run_live_stream"):
            session = mgr.start(url="rtsp://fake", name="Test")
        assert isinstance(session, LiveSession)
        assert session.url == "rtsp://fake"
        assert session.name == "Test"

    def test_session_appears_in_list(self):
        mgr = _make_manager()
        with patch("core.stream_manager.run_live_stream"):
            session = mgr.start(url="rtsp://fake", name="Cam")
        stream_ids = [s["stream_id"] for s in mgr.list()]
        assert session.stream_id in stream_ids

    def test_thread_is_started(self):
        mgr = _make_manager()
        started = threading.Event()

        def fake_run(*args, **kwargs):
            started.set()

        with patch("core.stream_manager.run_live_stream", side_effect=fake_run):
            mgr.start(url="rtsp://fake", name="Cam")
        started.wait(timeout=2)
        assert started.is_set()

    def test_two_sessions_have_different_ids(self):
        mgr = _make_manager()
        with patch("core.stream_manager.run_live_stream"):
            s1 = mgr.start(url="rtsp://cam1", name="A")
            s2 = mgr.start(url="rtsp://cam2", name="B")
        assert s1.stream_id != s2.stream_id


class TestStop:
    def test_stop_sets_event(self):
        mgr = _make_manager()
        with patch("core.stream_manager.run_live_stream"):
            session = mgr.start(url="rtsp://fake", name="Cam")
        mgr.stop(session.stream_id)
        assert session.stop_event.is_set()

    def test_stop_unknown_raises_key_error(self):
        mgr = _make_manager()
        with pytest.raises(KeyError):
            mgr.stop("does-not-exist")


class TestGet:
    def test_get_returns_session(self):
        mgr = _make_manager()
        with patch("core.stream_manager.run_live_stream"):
            session = mgr.start(url="rtsp://fake", name="Cam")
        assert mgr.get(session.stream_id) is session

    def test_get_unknown_returns_none(self):
        mgr = _make_manager()
        assert mgr.get("no-such-id") is None


class TestList:
    def test_list_contains_all_sessions(self):
        mgr = _make_manager()
        with patch("core.stream_manager.run_live_stream"):
            mgr.start(url="rtsp://cam1", name="A")
            mgr.start(url="rtsp://cam2", name="B")
        sessions = mgr.list()
        assert len(sessions) == 2
        urls = {s["url"] for s in sessions}
        assert "rtsp://cam1" in urls
        assert "rtsp://cam2" in urls

    def test_list_includes_required_fields(self):
        mgr = _make_manager()
        with patch("core.stream_manager.run_live_stream"):
            mgr.start(url="rtsp://cam1", name="Cam")
        s = mgr.list()[0]
        assert all(k in s for k in ("stream_id", "url", "name", "status", "started_at", "vehicle_count"))
```

- [ ] **Step 2: Run tests — expect FAIL (StreamManager not yet created)**

```bash
cd /home/vietanh/Documents/DATN/ALPR_Vietnamese
pytest tests/api/test_stream_manager.py -v
```

Expected: `ImportError: No module named 'core.stream_manager'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/api/test_stream_manager.py
git commit -m "test(stream-manager): add unit tests for start/stop/get/list"
```

---

## Task 6: StreamManager implementation

**Files:**
- Create: `api/core/stream_manager.py`

- [ ] **Step 1: Create the file**

Create `api/core/stream_manager.py`:

```python
"""
core/stream_manager.py — Manages N concurrent live RTSP stream sessions.

Each session runs in its own daemon thread executing run_live_stream().
StreamManager holds the session registry and provides start/stop/list/get.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from .live_pipeline import run_live_stream
from .models import ModelBundle


@dataclass
class LiveSession:
    stream_id: str
    url: str
    name: str
    status: Literal["connecting", "running", "reconnecting", "stopped", "error"]
    event_queue: asyncio.Queue
    mjpeg_queue: asyncio.Queue
    stop_event: threading.Event
    thread: threading.Thread
    started_at: datetime
    vehicle_count: int = 0


class StreamManager:
    def __init__(self, models: ModelBundle, loop: asyncio.AbstractEventLoop) -> None:
        self._models = models
        self._loop = loop
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    def start(
        self,
        url: str,
        name: str,
        stream_id: str | None = None,
    ) -> LiveSession:
        if stream_id is None:
            stream_id = uuid.uuid4().hex[:8]

        event_queue: asyncio.Queue = asyncio.Queue()
        mjpeg_queue: asyncio.Queue = asyncio.Queue(maxsize=60)
        stop_event = threading.Event()

        # Placeholder thread — replaced below before session is stored
        placeholder = threading.Thread(target=lambda: None)

        session = LiveSession(
            stream_id=stream_id,
            url=url,
            name=name,
            status="connecting",
            event_queue=event_queue,
            mjpeg_queue=mjpeg_queue,
            stop_event=stop_event,
            thread=placeholder,
            started_at=datetime.now(timezone.utc),
        )

        thread = threading.Thread(
            target=run_live_stream,
            args=(url, stream_id, event_queue, mjpeg_queue, self._loop, self._models, stop_event),
            daemon=True,
            name=f"live-{stream_id}",
        )
        session.thread = thread

        with self._lock:
            self._sessions[stream_id] = session

        thread.start()
        return session

    def stop(self, stream_id: str) -> None:
        with self._lock:
            session = self._sessions.get(stream_id)
        if session is None:
            raise KeyError(stream_id)
        session.stop_event.set()
        session.thread.join(timeout=10)
        session.status = "stopped"

    def get(self, stream_id: str) -> LiveSession | None:
        return self._sessions.get(stream_id)

    def list(self) -> list[dict]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [
            {
                "stream_id": s.stream_id,
                "name": s.name,
                "url": s.url,
                "status": s.status,
                "started_at": s.started_at.isoformat(),
                "vehicle_count": s.vehicle_count,
            }
            for s in sessions
        ]
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
cd /home/vietanh/Documents/DATN/ALPR_Vietnamese
pytest tests/api/test_stream_manager.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add api/core/stream_manager.py
git commit -m "feat(stream-manager): add LiveSession dataclass and StreamManager"
```

---

## Task 7: API endpoints

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Add imports at top of `api/main.py`**

After the existing imports block, add:

```python
from fastapi import WebSocket, WebSocketDisconnect
from core.stream_manager import LiveSession, StreamManager
```

- [ ] **Step 2: Add `_stream_mgr` global and update `lifespan`**

Add after the `_mjpeg_queues` global:

```python
_stream_mgr: StreamManager | None = None
```

Replace the existing `lifespan` body:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _stream_mgr
    _models = load_models()
    _stream_mgr = StreamManager(models=_models, loop=asyncio.get_event_loop())
    if MONGODB_URI:
        await init_db(MONGODB_URI, MONGODB_DB_NAME)
    else:
        logger.warning("MONGODB_URI not set — database persistence disabled.")
    yield
    await close_db()
```

- [ ] **Step 3: Add helper coroutines for the WebSocket handler**

Add these two functions before the route definitions (after `lifespan`):

```python
async def _ws_send_loop(websocket: WebSocket, session: LiveSession) -> None:
    """Drain event_queue and forward each event to the WebSocket client."""
    while True:
        ev = await session.event_queue.get()
        try:
            await websocket.send_json(ev)
        except Exception:
            return
        if ev.get("type") in ("stopped", "error"):
            return


async def _ws_recv_loop(websocket: WebSocket, session: LiveSession) -> None:
    """Read client → server messages. {"type":"stop"} sets stop_event."""
    async for msg in websocket.iter_json():
        if msg.get("type") == "stop":
            session.stop_event.set()
```

- [ ] **Step 4: Add the four /streams endpoints**

Add after the existing `/records/{job_id}/{track_id}` endpoint:

```python
@app.post("/streams")
async def create_stream(body: dict) -> dict:
    url  = body.get("url", "").strip()
    name = body.get("name", "Camera").strip() or "Camera"
    if not url:
        raise HTTPException(status_code=422, detail="url is required")
    stream_id = uuid.uuid4().hex[:8]
    _stream_mgr.start(url=url, name=name, stream_id=stream_id)
    return {"stream_id": stream_id}


@app.delete("/streams/{stream_id}", status_code=204)
async def stop_stream(stream_id: str) -> None:
    if _stream_mgr.get(stream_id) is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _stream_mgr.stop, stream_id)


@app.get("/streams")
async def list_streams() -> list[dict]:
    return _stream_mgr.list()


@app.websocket("/streams/{stream_id}/ws")
async def stream_ws(websocket: WebSocket, stream_id: str) -> None:
    session = _stream_mgr.get(stream_id)
    if session is None:
        await websocket.close(code=4004)
        return
    await websocket.accept()

    send_task = asyncio.create_task(_ws_send_loop(websocket, session))
    recv_task = asyncio.create_task(_ws_recv_loop(websocket, session))

    done, pending = await asyncio.wait(
        [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    for task in done:
        try:
            task.result()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass


@app.get("/streams/{stream_id}/mjpeg")
async def stream_live_mjpeg(stream_id: str) -> StreamingResponse:
    session = _stream_mgr.get(stream_id)
    if session is None:
        return HTMLResponse("Stream not found", status_code=404)

    async def gen():
        while True:
            try:
                frame_bytes = await asyncio.wait_for(
                    session.mjpeg_queue.get(), timeout=30.0
                )
            except asyncio.TimeoutError:
                break
            if frame_bytes is None:
                break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 5: Verify API starts without errors**

```bash
cd api && python -c "from main import app; print('ok')"
```

Expected output: `ok`

- [ ] **Step 6: Commit**

```bash
git add api/main.py
git commit -m "feat(api): add /streams POST/DELETE/GET and WS /streams/{id}/ws endpoints"
```

---

## Task 8: Vite proxy for /streams

**Files:**
- Modify: `web/vite.config.js`

- [ ] **Step 1: Add /streams proxy with WebSocket support**

Replace the existing proxy config in `web/vite.config.js`:

```js
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), "");
    const apiBase = env.VITE_API_BASE ?? "http://localhost:8000";

    return {
        plugins: [react()],
        server: {
            proxy: {
                "/upload":  apiBase,
                "/stream":  apiBase,
                "/records": apiBase,
                "/streams": {
                    target: apiBase,
                    ws: true,           // enable WebSocket proxying
                    changeOrigin: true,
                },
            },
        },
    };
});
```

- [ ] **Step 2: Verify Vite config parses**

```bash
cd web && node -e "const v = require('./vite.config.js'); console.log('ok')"
```

Expected output: `ok` (or a harmless ESM warning — as long as it doesn't throw).

- [ ] **Step 3: Commit**

```bash
git add web/vite.config.js
git commit -m "feat(vite): proxy /streams with WebSocket support"
```

---

## Task 9: Frontend WebSocket hook

**Files:**
- Create: `web/src/hooks/useStreamSession.js`

- [ ] **Step 1: Create the hook**

Create `web/src/hooks/useStreamSession.js`:

```js
import { useEffect, useRef, useCallback } from 'react'

const PING_INTERVAL_MS = 30_000

/**
 * WebSocket hook for a persistent live camera session.
 *
 * Connects to /streams/{streamId}/ws.
 * Returns { stop } — call stop() to send {"type":"stop"} to the server.
 */
export function useStreamSession(streamId, {
  onVehicle,
  onRejectedVehicle,
  onConnected,
  onReconnecting,
  onHeartbeat,
  onStopped,
  onError,
} = {}) {
  const wsRef   = useRef(null)
  const pingRef = useRef(null)

  const send = useCallback((msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg))
    }
  }, [])

  const stop = useCallback(() => send({ type: 'stop' }), [send])

  useEffect(() => {
    if (!streamId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/streams/${streamId}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      pingRef.current = setInterval(() => send({ type: 'ping' }), PING_INTERVAL_MS)
    }

    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data)
      switch (ev.type) {
        case 'connected':        onConnected?.(ev);        break
        case 'vehicle':          onVehicle?.(ev);          break
        case 'rejected_vehicle': onRejectedVehicle?.(ev);  break
        case 'reconnecting':     onReconnecting?.(ev);     break
        case 'heartbeat':        onHeartbeat?.(ev);        break
        case 'stopped':          onStopped?.(ev);          break
        case 'error':            onError?.(ev);            break
        default:                 break
      }
    }

    ws.onerror = () => onError?.({ message: 'WebSocket connection error' })
    ws.onclose = () => clearInterval(pingRef.current)

    return () => {
      clearInterval(pingRef.current)
      ws.close()
    }
  }, [streamId]) // eslint-disable-line react-hooks/exhaustive-deps

  return { stop }
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/hooks/useStreamSession.js
git commit -m "feat(frontend): add useStreamSession WebSocket hook"
```

---

## Task 10: StreamCard component

**Files:**
- Create: `web/src/components/StreamCard.jsx`

- [ ] **Step 1: Create the component**

Create `web/src/components/StreamCard.jsx`:

```jsx
const STATUS_COLORS = {
  connecting:   'bg-yellow-900/50 text-yellow-400',
  running:      'bg-emerald-900/50 text-emerald-400',
  reconnecting: 'bg-orange-900/50 text-orange-400',
  stopped:      'bg-slate-800 text-slate-400',
  error:        'bg-red-900/50 text-red-400',
}

const STATUS_LABELS = {
  connecting:   'Đang kết nối…',
  running:      'Đang chạy',
  reconnecting: 'Đang kết nối lại…',
  stopped:      'Đã dừng',
  error:        'Lỗi',
}

/**
 * StreamCard — clickable card for one live camera session.
 * Props:
 *   session      { streamId, url, name, status, vehicleCount }
 *   isSelected   boolean
 *   onSelect     () => void
 *   onStop       (streamId) => void
 */
export default function StreamCard({ session, isSelected, onSelect, onStop }) {
  const isActive = session.status === 'running' || session.status === 'reconnecting'

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left p-3 rounded-xl border transition-colors
        ${isSelected
          ? 'border-blue-500 bg-blue-900/20'
          : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-white truncate">{session.name}</p>
          <p className="text-[10px] text-slate-500 truncate mt-0.5">{session.url}</p>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap
                          ${STATUS_COLORS[session.status] ?? STATUS_COLORS.stopped}`}>
          {STATUS_LABELS[session.status] ?? session.status}
        </span>
      </div>

      <div className="flex items-center justify-between mt-2">
        <span className="text-xs text-slate-400">
          {session.vehicleCount} xe nhận diện
        </span>
        {isActive && (
          <button
            onClick={(e) => { e.stopPropagation(); onStop(session.streamId) }}
            className="text-[10px] text-red-400 hover:text-red-300 border border-red-900/50
                       hover:border-red-700 px-2 py-0.5 rounded transition-colors"
          >
            Dừng
          </button>
        )}
      </div>
    </button>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/StreamCard.jsx
git commit -m "feat(frontend): add StreamCard component"
```

---

## Task 11: LiveStreamTab component

**Files:**
- Create: `web/src/components/LiveStreamTab.jsx`

- [ ] **Step 1: Create the component**

Create `web/src/components/LiveStreamTab.jsx`:

```jsx
import { useState, useEffect, useCallback } from 'react'

import StreamCard     from './StreamCard'
import VehiclePanel   from './VehiclePanel'
import { useStreamSession } from '../hooks/useStreamSession'

function normalizeSession(raw) {
  return {
    streamId:     raw.stream_id,
    url:          raw.url,
    name:         raw.name,
    status:       raw.status,
    vehicleCount: raw.vehicle_count ?? 0,
  }
}

/**
 * LiveStreamTab — manages multiple RTSP camera sessions.
 *
 * - Lists existing sessions (fetched on mount from GET /streams).
 * - Lets the operator add a new stream (POST /streams).
 * - Clicking a session card selects it: shows its MJPEG feed + vehicle sidebar.
 * - WebSocket events for the selected session are handled by useStreamSession.
 */
export default function LiveStreamTab() {
  const [sessions,    setSessions]    = useState([])
  const [selectedId,  setSelectedId]  = useState(null)
  const [vehicles,    setVehicles]    = useState({})
  const [formUrl,     setFormUrl]     = useState('')
  const [formName,    setFormName]    = useState('')
  const [adding,      setAdding]      = useState(false)

  // Load existing sessions on mount
  useEffect(() => {
    fetch('/streams')
      .then(r => r.json())
      .then(data => setSessions(data.map(normalizeSession)))
      .catch(() => {})
  }, [])

  // ── WS callbacks for the selected session ────────────────────────────────
  const handleVehicle = useCallback((data) => {
    setVehicles(prev => {
      const isNew = !prev[data.id]
      if (isNew) {
        setSessions(all => all.map(s =>
          s.streamId === selectedId
            ? { ...s, vehicleCount: s.vehicleCount + 1 }
            : s
        ))
      }
      return { ...prev, [data.id]: data }
    })
  }, [selectedId])

  const handleConnected = useCallback(() => {
    setSessions(all => all.map(s =>
      s.streamId === selectedId ? { ...s, status: 'running' } : s
    ))
  }, [selectedId])

  const handleReconnecting = useCallback(() => {
    setSessions(all => all.map(s =>
      s.streamId === selectedId ? { ...s, status: 'reconnecting' } : s
    ))
  }, [selectedId])

  const handleStopped = useCallback(() => {
    setSessions(all => all.map(s =>
      s.streamId === selectedId ? { ...s, status: 'stopped' } : s
    ))
  }, [selectedId])

  const handleError = useCallback(() => {
    setSessions(all => all.map(s =>
      s.streamId === selectedId ? { ...s, status: 'error' } : s
    ))
  }, [selectedId])

  const { stop } = useStreamSession(selectedId, {
    onVehicle:    handleVehicle,
    onConnected:  handleConnected,
    onReconnecting: handleReconnecting,
    onStopped:    handleStopped,
    onError:      handleError,
  })

  // ── Actions ───────────────────────────────────────────────────────────────
  const handleAddStream = async (e) => {
    e.preventDefault()
    if (!formUrl.trim()) return
    setAdding(true)
    try {
      const res = await fetch('/streams', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url:  formUrl.trim(),
          name: formName.trim() || 'Camera',
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const { stream_id } = await res.json()
      setSessions(prev => [
        ...prev,
        { streamId: stream_id, url: formUrl.trim(), name: formName.trim() || 'Camera',
          status: 'connecting', vehicleCount: 0 },
      ])
      setSelectedId(stream_id)
      setVehicles({})
      setFormUrl('')
      setFormName('')
    } catch {
      // Silently ignore — operator will see the error in the session card
    } finally {
      setAdding(false)
    }
  }

  const handleStop = async (streamId) => {
    await fetch(`/streams/${streamId}`, { method: 'DELETE' })
    setSessions(prev => prev.map(s =>
      s.streamId === streamId ? { ...s, status: 'stopped' } : s
    ))
  }

  const handleSelect = (streamId) => {
    setSelectedId(streamId)
    setVehicles({})
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  const vehicleList    = Object.values(vehicles).sort((a, b) => a.id - b.id)
  const totalDone      = vehicleList.filter(v => v.done).length
  const selectedStatus = sessions.find(s => s.streamId === selectedId)?.status

  return (
    <div className="flex gap-4 items-start">

      {/* ── Left: controls + MJPEG (65%) ─────────────────────────────────── */}
      <div className="flex-1 min-w-0 space-y-4">

        {/* Add stream form */}
        <form
          onSubmit={handleAddStream}
          className="bg-slate-800/50 rounded-xl p-4 border border-slate-700"
        >
          <p className="text-xs font-semibold text-slate-300 mb-3">Thêm camera mới</p>
          <div className="flex gap-2">
            <input
              value={formUrl}
              onChange={e => setFormUrl(e.target.value)}
              placeholder="rtsp://camera-ip:554/stream"
              className="flex-1 bg-slate-900 text-white text-xs rounded-lg px-3 py-2
                         border border-slate-700 focus:border-blue-500 outline-none"
            />
            <input
              value={formName}
              onChange={e => setFormName(e.target.value)}
              placeholder="Tên camera"
              className="w-28 bg-slate-900 text-white text-xs rounded-lg px-3 py-2
                         border border-slate-700 focus:border-blue-500 outline-none"
            />
            <button
              type="submit"
              disabled={adding || !formUrl.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs
                         font-medium px-4 py-2 rounded-lg transition-colors"
            >
              {adding ? '…' : 'Bắt đầu'}
            </button>
          </div>
        </form>

        {/* Session cards */}
        {sessions.length > 0 && (
          <div className="grid grid-cols-2 gap-2">
            {sessions.map(s => (
              <StreamCard
                key={s.streamId}
                session={s}
                isSelected={s.streamId === selectedId}
                onSelect={() => handleSelect(s.streamId)}
                onStop={handleStop}
              />
            ))}
          </div>
        )}

        {/* MJPEG preview for selected session */}
        {selectedId && (
          <div className="relative bg-black rounded-2xl overflow-hidden shadow-xl">
            <img
              src={`/streams/${selectedId}/mjpeg`}
              alt="Camera stream"
              className="w-full block"
              style={{ maxHeight: '60vh', objectFit: 'contain' }}
            />
            {selectedStatus === 'reconnecting' && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                <p className="text-orange-400 text-sm font-medium animate-pulse">
                  Đang kết nối lại…
                </p>
              </div>
            )}
            {selectedStatus === 'connecting' && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                <p className="text-yellow-400 text-sm font-medium animate-pulse">
                  Đang kết nối…
                </p>
              </div>
            )}
          </div>
        )}

        {/* Empty state */}
        {sessions.length === 0 && (
          <div className="flex flex-col items-center justify-center text-slate-600 py-16 gap-3">
            <svg className="w-12 h-12 opacity-30" fill="none" viewBox="0 0 24 24"
                 stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round"
                    d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53
                       l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0
                       0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
            </svg>
            <p className="text-sm">Chưa có camera nào. Thêm URL RTSP ở trên.</p>
          </div>
        )}
      </div>

      {/* ── Right: vehicle detection panel (35%) ─────────────────────────── */}
      <div className="w-80 flex-shrink-0" style={{ height: 'calc(100vh - 72px)' }}>
        <VehiclePanel
          vehicles={vehicleList}
          totalDone={totalDone}
          jobId={selectedId}
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/LiveStreamTab.jsx
git commit -m "feat(frontend): add LiveStreamTab with multi-session management"
```

---

## Task 12: App.jsx tab integration

**Files:**
- Modify: `web/src/App.jsx`

- [ ] **Step 1: Add import and tab state**

At the top of `web/src/App.jsx`, add the import after the existing imports:

```js
import LiveStreamTab from './components/LiveStreamTab'
```

Inside the `App()` function, after the `const [showHistory, setShowHistory] = useState(false)` line, add:

```js
const [activeTab, setActiveTab] = useState('upload')
```

- [ ] **Step 2: Replace the main flex container**

Replace this block:

```jsx
      {/* ── Main 2-column layout ── */}
      <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5
                      flex gap-4 items-start">

        {/* ── LEFT: Video / Drop zone (65%) ── */}
        <div className="flex-1 min-w-0">
          {isIdle ? (
            /* Drop zone shown when idle */
            <DropZone onFileSelect={handleFileSelect} dark />
          ) : (
            <>
              {/* Live annotated frame during processing, original video when done */}
              <LiveFrame
                jobId={jobId}
                videoUrl={videoUrl}
                progress={progress}
                status={status}
              />

              {/* OCR Statistics panel */}
              <OcrStatsPanel
                vehicles={vehicleList}
                rejectedVehicles={rejectedList}
                jobId={jobId}
              />
            </>
          )}
        </div>

        {/* ── RIGHT: Vehicle detection panel (35%) ── */}
        <div className="w-80 flex-shrink-0" style={{ height: 'calc(100vh - 72px)' }}>
          <VehiclePanel vehicles={vehicleList} totalDone={totalDone} jobId={jobId} />
        </div>
      </div>
```

With:

```jsx
      {/* ── Main layout ── */}
      <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5 flex flex-col gap-4">

        {/* Tab switcher */}
        <div className="flex gap-1 bg-slate-800/50 p-1 rounded-xl w-fit">
          <button
            onClick={() => setActiveTab('upload')}
            className={`text-xs px-4 py-1.5 rounded-lg transition-colors font-medium
              ${activeTab === 'upload'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-white'}`}
          >
            Tải video lên
          </button>
          <button
            onClick={() => setActiveTab('live')}
            className={`text-xs px-4 py-1.5 rounded-lg transition-colors font-medium
              ${activeTab === 'live'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-white'}`}
          >
            Camera trực tiếp
          </button>
        </div>

        {activeTab === 'live' ? (
          <LiveStreamTab />
        ) : (
          <div className="flex gap-4 items-start">
            {/* ── LEFT: Video / Drop zone (65%) ── */}
            <div className="flex-1 min-w-0">
              {isIdle ? (
                <DropZone onFileSelect={handleFileSelect} dark />
              ) : (
                <>
                  <LiveFrame
                    jobId={jobId}
                    videoUrl={videoUrl}
                    progress={progress}
                    status={status}
                  />
                  <OcrStatsPanel
                    vehicles={vehicleList}
                    rejectedVehicles={rejectedList}
                    jobId={jobId}
                  />
                </>
              )}
            </div>

            {/* ── RIGHT: Vehicle detection panel (35%) ── */}
            <div className="w-80 flex-shrink-0" style={{ height: 'calc(100vh - 72px)' }}>
              <VehiclePanel vehicles={vehicleList} totalDone={totalDone} jobId={jobId} />
            </div>
          </div>
        )}
      </div>
```

- [ ] **Step 3: Start the dev stack and verify both tabs render**

```bash
# Terminal 1 — backend
cd api && uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd web && npm run dev
```

Open `http://localhost:5173`. Verify:
- "Tải video lên" tab shows the existing drop-zone UI
- "Camera trực tiếp" tab shows the add-stream form and empty state
- Switching tabs does not cause any console errors

- [ ] **Step 4: Smoke test the /streams API**

With the backend running:

```bash
# Create a stream
curl -s -X POST http://localhost:8000/streams \
  -H "Content-Type: application/json" \
  -d '{"url":"rtsp://fake/stream","name":"Test"}' | python3 -m json.tool

# Expected: {"stream_id": "xxxxxxxx"}

# List streams
curl -s http://localhost:8000/streams | python3 -m json.tool

# Expected: [{"stream_id": "...", "name": "Test", "url": "rtsp://fake/stream", "status": "connecting", ...}]

# Stop stream (replace <id> with actual stream_id)
curl -s -X DELETE http://localhost:8000/streams/<id> -o /dev/null -w "%{http_code}"

# Expected: 204
```

- [ ] **Step 5: Commit**

```bash
git add web/src/App.jsx
git commit -m "feat(frontend): add tab switcher integrating LiveStreamTab alongside upload flow"
```

---

## Self-Review Checklist

| Spec requirement | Covered by |
|---|---|
| Multiple concurrent RTSP streams | StreamManager (Task 6) — one thread per session |
| Offline + live coexist | Separate `_jobs` / `_stream_mgr` globals in main.py (Task 7) |
| Persistent sessions, operator stops manually | DELETE /streams + `{"type":"stop"}` WS message (Tasks 7, 9, 11) |
| Crop-count OCR trigger | `ready_for_multiframe_ocr` check after `buffer_crop` in live_pipeline (Task 4) |
| No re-trigger after `_done` | `should_ocr()` guard, tested in Task 3 |
| Track-loss fallback for short stays | Same `mark_lost` / `ready_for_multiframe_ocr` check as offline, tested Task 3 |
| Exponential backoff reconnect | `stop_event.wait(timeout=min(2^N, 30))` in live_pipeline (Task 4) |
| WebSocket bidirectional (stop command) | `_ws_recv_loop` sets `stop_event` (Task 7) |
| MJPEG preview for live sessions | `/streams/{id}/mjpeg` endpoint (Task 7), `<img src>` in LiveStreamTab (Task 11) |
| DB: source_type=live_stream, stream_url | `_live_session_create` in live_pipeline (Task 4), model update (Task 2) |
| DB: status=stopped on clean stop | `_session_update(stream_id, {"status":"stopped",...})` at end of run_live_stream |
| Frontend: session cards, add-stream form | LiveStreamTab + StreamCard (Tasks 10, 11) |
| Frontend: WS hook | useStreamSession (Task 9) |
| Vite proxy for WS | ws:true in vite.config.js (Task 8) |
