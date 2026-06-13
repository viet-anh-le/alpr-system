# Incident Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new "Incident Monitor" page that lets operators observe a live RTSP camera (via MediaMTX/WebRTC) or play back an uploaded video, mark incidents, and get fast targeted ALPR analysis on a short window around each mark.

**Architecture:** Extract a pure `pipeline_core.process_frames` function from the existing `run_job` so live snapshots and uploaded-video intervals share the same detect→track→OCR→vote logic. A `FrameSource` Protocol abstracts where frames come from. MediaMTX is the single RTSP consumer of cameras; its HTTP API is used to add/remove paths dynamically per session. The browser uses WebRTC (WHEP) for low-latency display with an MJPEG fallback. Marks return immediately with an `incident_id`; results stream back over SSE.

**Tech Stack:** Python 3.10+, FastAPI, Motor (async MongoDB), OpenCV, MediaMTX, React 18 + Vite, native browser RTCPeerConnection (no third-party WebRTC lib).

**Spec:** [docs/superpowers/specs/2026-05-20-incident-monitor-design.md](../specs/2026-05-20-incident-monitor-design.md)

---

## Phase 0 — Test fixtures & directory layout

### Task 0.1: Create test fixture video

**Files:**
- Create: `tests/fixtures/short_clip.mp4` (binary, ~30 frames, 1 second @ 30fps, 640×360)
- Create: `tests/fixtures/README.md`

- [ ] **Step 1: Generate the fixture programmatically**

Run this one-off script (it does not need to be committed):

```bash
python - <<'PY'
import cv2
import numpy as np
from pathlib import Path

out = Path("tests/fixtures")
out.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out / "short_clip.mp4"), fourcc, 30.0, (640, 360))

for i in range(30):
    frame = np.full((360, 640, 3), 80, dtype=np.uint8)
    cv2.putText(frame, f"frame {i}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    writer.write(frame)

writer.release()
print("ok")
PY
```

- [ ] **Step 2: Add the fixture README**

```markdown
# Test fixtures

- `short_clip.mp4` — 30 frames @ 30fps, 640×360. No plates. Used for frame-source tests
  that do not need real detections.
```

Write to `tests/fixtures/README.md`.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/short_clip.mp4 tests/fixtures/README.md
git commit -m "test: add short_clip.mp4 fixture for frame-source tests"
```

---

## Phase 1 — `FrameSource` abstraction (no behavior change yet)

### Task 1.1: Define `FrameSource` Protocol and `FileFrameSource`

**Files:**
- Create: `api/core/frame_source.py`
- Create: `tests/test_frame_source.py`

- [ ] **Step 1: Write the failing tests**

Write to `tests/test_frame_source.py`:

```python
"""Tests for api/core/frame_source.py."""
from __future__ import annotations

import numpy as np
import pytest

from api.core.frame_source import FileFrameSource, LiveBufferFrameSource


FIXTURE = "tests/fixtures/short_clip.mp4"


@pytest.mark.unit
def test_file_source_yields_all_frames_when_unrestricted():
    src = FileFrameSource(FIXTURE)
    frames = list(src.iter_frames())
    assert len(frames) == 30
    idx, frame, ts = frames[0]
    assert idx == 0
    assert frame.shape == (360, 640, 3)
    assert ts == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_file_source_reports_metadata():
    src = FileFrameSource(FIXTURE)
    assert src.fps == pytest.approx(30.0, abs=0.1)
    assert src.frame_size == (640, 360)
    assert src.total_frames == 30


@pytest.mark.unit
def test_file_source_respects_t_start():
    src = FileFrameSource(FIXTURE, t_start=0.5)  # start at frame ~15
    frames = list(src.iter_frames())
    assert 13 <= len(frames) <= 17  # allow ±2 frames for codec seek imprecision
    first_idx = frames[0][0]
    assert first_idx >= 13


@pytest.mark.unit
def test_file_source_respects_t_end():
    src = FileFrameSource(FIXTURE, t_start=0.0, t_end=0.5)  # ~first 15 frames
    frames = list(src.iter_frames())
    assert 13 <= len(frames) <= 17
    last_ts = frames[-1][2]
    assert last_ts <= 0.6


@pytest.mark.unit
def test_file_source_t_end_beyond_duration_clamps_to_eof():
    src = FileFrameSource(FIXTURE, t_start=0.0, t_end=999.0)
    frames = list(src.iter_frames())
    assert len(frames) == 30  # never errors, just stops at EOF


@pytest.mark.unit
def test_live_buffer_source_passthrough():
    fake_frames = [
        (i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
        for i in range(10)
    ]
    src = LiveBufferFrameSource(fake_frames, fps=30.0, frame_size=(640, 360))
    out = list(src.iter_frames())
    assert out == fake_frames
    assert src.fps == 30.0
    assert src.total_frames == 10
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_frame_source.py -v
```

Expected: `ImportError: cannot import name 'FileFrameSource' from 'api.core.frame_source'` or "No module named …".

- [ ] **Step 3: Write the implementation**

Write to `api/core/frame_source.py`:

```python
"""FrameSource protocol and implementations.

A FrameSource yields (frame_idx, frame_bgr, timestamp_sec) tuples and exposes
fps / frame_size / total_frames metadata. Used by pipeline_core.process_frames
so the inference loop is decoupled from where frames come from.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource(Protocol):
    fps: float
    frame_size: tuple[int, int]      # (width, height)
    total_frames: int | None         # None when unknown / unbounded

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        ...


class FileFrameSource:
    """A FrameSource backed by a video file on disk.

    Seeks to ``t_start`` and stops yielding when frame timestamp ≥ ``t_end``.
    ``t_end=None`` means "until end of file".
    """

    def __init__(self, path: str | Path, t_start: float = 0.0, t_end: float | None = None) -> None:
        self.path = str(path)
        self.t_start = float(t_start)
        self.t_end = None if t_end is None else float(t_end)

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total if total > 0 else None
        cap.release()

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        cap = cv2.VideoCapture(self.path)
        try:
            if self.t_start > 0.0:
                cap.set(cv2.CAP_PROP_POS_MSEC, self.t_start * 1000.0)
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # POS_MSEC reports timestamp of the next frame on some codecs;
                # compute timestamp from index + start for stability.
                ts = self.t_start + (idx / self.fps)
                if self.t_end is not None and ts >= self.t_end:
                    break
                yield idx, frame, ts
                idx += 1
        finally:
            cap.release()


class LiveBufferFrameSource:
    """A FrameSource that wraps an already-decoded list of frames.

    Used by the incident analyzer after snapshotting a LiveSession's rolling
    buffer. Iteration is a no-op pass-through.
    """

    def __init__(
        self,
        frames: list[tuple[int, np.ndarray, float]],
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self._frames = frames
        self.fps = float(fps)
        self.frame_size = frame_size
        self.total_frames = len(frames)

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        yield from self._frames
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_frame_source.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/core/frame_source.py tests/test_frame_source.py
git commit -m "feat(core): add FrameSource protocol with file + live-buffer impls"
```

---

## Phase 2 — Pipeline core extraction (regression guard)

### Task 2.1: Capture current pipeline output as a golden file (regression baseline)

**Files:**
- Create: `tests/test_pipeline_core_parity.py` (skeleton — only the golden-capture part)
- Create: `tests/fixtures/golden_run_job_events.json` (auto-generated; committed)

This task captures what `run_job` emits TODAY so we can prove the refactored version emits the same thing.

- [ ] **Step 1: Write a helper test that captures events into a file**

Write to `tests/test_pipeline_core_parity.py`:

```python
"""Regression-guard test: refactored process_frames must match run_job's output."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

GOLDEN = Path("tests/fixtures/golden_run_job_events.json")


def _normalize_event(ev: dict) -> dict:
    """Strip non-deterministic / image-blob fields so we can compare reliably."""
    drop = {"plate_b64", "vehicle_b64", "detail"}
    return {k: v for k, v in ev.items() if k not in drop}


@pytest.mark.integration
@pytest.mark.skipif(not GOLDEN.exists(), reason="golden file not yet captured")
def test_process_frames_matches_run_job_golden():
    """After refactor: process_frames(FileFrameSource(video)) must produce the
    same event stream as the legacy run_job(video). Compares normalized events."""
    from api.core.frame_source import FileFrameSource
    from api.core.models import load_models
    from api.core.pipeline_core import process_frames

    fixture = "tests/fixtures/short_clip.mp4"
    captured: list[dict] = []

    def emit(ev: dict) -> None:
        captured.append(_normalize_event(ev))

    models = load_models()
    source = FileFrameSource(fixture)
    summary = process_frames(source, emit=emit, models=models)
    captured.append({"type": "complete", **{k: v for k, v in summary.items() if k != "duration_ms"}})

    golden = json.loads(GOLDEN.read_text())
    assert captured == golden
```

- [ ] **Step 2: Write a one-off capture script (NOT a test)**

Write to `tests/_capture_golden.py` (will be deleted in a later task):

```python
"""One-off: run the current run_job against the fixture and dump the event
stream. Run with: python tests/_capture_golden.py"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from api.core.models import load_models
from api.core.pipeline import run_job


def main() -> None:
    fixture = "tests/fixtures/short_clip.mp4"
    captured: list[dict] = []

    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def drain() -> None:
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                break
            drop = {"plate_b64", "vehicle_b64", "detail"}
            captured.append({k: v for k, v in ev.items() if k not in drop})

    models = load_models()
    jobs: dict = {}
    run_job(
        video_path=fixture,
        job_id="golden",
        queue=queue,
        loop=loop,
        models=models,
        jobs=jobs,
        filename="short_clip.mp4",
        mjpeg_queue=None,
    )
    loop.run_until_complete(drain())

    out = Path("tests/fixtures/golden_run_job_events.json")
    out.write_text(json.dumps(captured, indent=2, default=str))
    print(f"wrote {len(captured)} events to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the capture script**

```bash
python tests/_capture_golden.py
```

Expected: prints `wrote N events to tests/fixtures/golden_run_job_events.json`.

The fixture has no plates, so the event stream will mostly be `progress` and a final `complete`. That's fine — what we are guarding against is *order changes* and *count changes* during the refactor.

- [ ] **Step 4: Verify the parity test is now collected but currently skipped (pipeline_core does not exist)**

```bash
pytest tests/test_pipeline_core_parity.py -v
```

Expected: 1 collected, 1 skipped (because `process_frames` import fails). The skip-reason check in Step 1 also handles missing-file; for now the missing module raises before the skip, so it may error. **That is expected** until Task 2.3 lands. Move on.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_core_parity.py tests/fixtures/golden_run_job_events.json tests/_capture_golden.py
git commit -m "test: capture legacy run_job event stream as golden baseline"
```

### Task 2.2: Create empty `pipeline_core` module with a stub `process_frames`

**Files:**
- Create: `api/core/pipeline_core.py`

- [ ] **Step 1: Write a stub that exists but is intentionally not yet correct**

Write to `api/core/pipeline_core.py`:

```python
"""pipeline_core — pure inference loop shared by run_job and incident_analyzer.

Will be filled in by Task 2.3 by extracting logic from pipeline.run_job.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .frame_source import FrameSource
from .models import ModelBundle

logger = logging.getLogger(__name__)


def process_frames(
    source: FrameSource,
    emit: Callable[[dict], None],
    models: ModelBundle,
    *,
    session_id: str = "",
    loop: asyncio.AbstractEventLoop | None = None,
    mjpeg_queue: asyncio.Queue | None = None,
) -> dict:
    """Run detect → track → buffer → OCR → vote on the given FrameSource.

    Returns a summary dict: {total_vehicles, processed_frames}.
    """
    raise NotImplementedError("filled in by Task 2.3")
```

- [ ] **Step 2: Commit**

```bash
git add api/core/pipeline_core.py
git commit -m "chore(core): scaffold empty pipeline_core module"
```

### Task 2.3: Extract `process_frames` from `run_job`

**Files:**
- Modify: `api/core/pipeline_core.py` (replace stub)
- Modify: `api/core/pipeline.py` (refactor `run_job` to delegate)

This is the largest task. The goal: move every line of inference logic out of `run_job` into `process_frames`, leaving `run_job` as a thin wrapper that owns video opening / file cleanup / session bookkeeping.

- [ ] **Step 1: Re-read the current `run_job` to map what moves where**

```bash
sed -n '300,532p' api/core/pipeline.py
```

Identify three sections:

- **A. Setup** (lines ~326-340): MongoDB session create, `cv2.VideoCapture` open, fps/total/orig_w/orig_h, `WebTrackletManager`, `TrajectoryAssociator`. ← **stays in `run_job`** (file handling) BUT the tracker/associator construction moves into `process_frames`.
- **B. Per-frame loop** (lines ~342-485): detection, tracking, plate detection, buffering, MJPEG emit, progress emit, lost-track finalization. ← **moves to `process_frames`**.
- **C. Teardown** (lines ~488-532): finalise remaining tracks, final snapshot, complete event, MongoDB session update, file cleanup. ← **finalisation moves to `process_frames`; file cleanup and session_update stay in `run_job`**.

- [ ] **Step 2: Write `process_frames` by extracting from the current loop**

Replace the contents of `api/core/pipeline_core.py` with:

```python
"""pipeline_core — pure inference loop shared by run_job and incident_analyzer.

Runs detect → track → buffer → OCR → vote on any FrameSource. Does NOT open
files, NOT touch MongoDB session documents, NOT delete temp files. Those
responsibilities live with the caller (run_job for the upload flow, or
incident_analyzer for marks).
"""
from __future__ import annotations

import asyncio
import gc
import logging
import re
from pathlib import Path as _Path
from typing import Callable

import cv2
import numpy as np
import torch

from .association import TrajectoryAssociator
from .config import (
    FRAME_STRIDE,
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
    PLATE_PAD,
    TOP_K_FRAMES,
    VEHICLE_CLASSES,
)
from .database import upload_image as _storage_upload
from .frame_source import FrameSource
from .gates import is_sharp
from .models import ModelBundle, multiframe_ocr_infer, ocr_batch, preprocess_plate
from .quality_scorer import quality_score
from .tracker import WebTrackletManager
from .video_processor import (
    crop_vehicle as _crop_vehicle,
    draw_annotated_frame as _draw_annotated_frame,
)

logger = logging.getLogger(__name__)

_PLATE_TRACKER_CFG = str(
    _Path(__file__).resolve().parents[2] / "configs/tracking/bytetrack_plate.yaml"
)

_VN_PLATE_RE = re.compile(
    r"^(?:"
    r"\d{2}[A-Z]{1,2}-\d{5}"
    r"|\d{2}-(?:[A-Z]\d|[A-Z]{2})-\d{5}"
    r"|\d{2}[A-Z]-\d{4}"
    r"|\d{2}-[A-Z]\d-\d{4}"
    r")$"
)


def _plate_valid(char_probs: list[tuple[str, float]]) -> bool:
    plate = "".join(c for c, _ in char_probs)
    return bool(_VN_PLATE_RE.match(plate))


def _safe_put(q: asyncio.Queue, item: object) -> None:
    if not q.full():
        q.put_nowait(item)


def _run_multiframe_ocr(
    tid: int,
    tracker: WebTrackletManager,
    models: ModelBundle,
    emit: Callable[[dict], None],
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    record_save: Callable | None,
) -> None:
    crops, scores = tracker._buffers[tid].top_k(k=TOP_K_FRAMES)
    if not crops:
        return

    vote_summary: dict[str, int] = {}

    if models.multiframe_ocr is not None:
        tensors = torch.stack([preprocess_plate(c) for c in crops]).unsqueeze(0)
        quality = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        char_probs = multiframe_ocr_infer(models.multiframe_ocr, tensors, quality, models.device)
        ocr_method = "multiframe"
    else:
        tensors = torch.stack([preprocess_plate(c) for c in crops]).to(models.device)
        ocr_results = ocr_batch(models.ocr, tensors, models.device)
        prob_lists = [chars for chars, _ in ocr_results]
        for pl in prob_lists:
            text = "".join(c for c, _ in pl)
            if text:
                vote_summary[text] = vote_summary.get(text, 0) + 1
        char_probs = WebTrackletManager._segment_vote(prob_lists)
        if char_probs is not None:
            ocr_method = "segment_vote"
        else:
            char_probs = WebTrackletManager._prob_vote(prob_lists)
            ocr_method = "prob_vote"

    if not _plate_valid(char_probs):
        if crops and scores:
            best_idx = scores.index(max(scores))
            tracker.update_plate_img(tid, crops[best_idx], char_probs)

        rejected_plate = "".join(c for c, _ in char_probs)
        rejected_chars = [[c, round(p, 3)] for c, p in char_probs]
        emit({
            "type": "rejected_vehicle",
            "id": tid,
            "cls": tracker._cls.get(tid, ""),
            "plate": rejected_plate,
            "chars": rejected_chars,
            "plate_b64": tracker.plate_b64(tid),
            "vehicle_b64": tracker.vehicle_b64(tid),
            "ocr_frames": len(crops),
            "vote_summary": vote_summary,
        })
        return

    tracker.update(tid, char_probs, all_confident=True)
    tracker._done[tid] = True

    if crops and scores:
        best_idx = scores.index(max(scores))
        tracker.update_plate_img(tid, crops[best_idx], char_probs)

    if tracker.plate_changed(tid):
        emit({
            "type": "vehicle",
            "id": tid,
            "cls": tracker._cls.get(tid, ""),
            "plate": tracker.display_text(tid),
            "chars": tracker.chars_json(tid),
            "done": True,
            "plate_b64": tracker.plate_b64(tid),
            "vehicle_b64": tracker.vehicle_b64(tid),
            "ocr_frames": tracker.ocr_frames(tid),
        })

    if session_id and loop is not None and record_save is not None:
        record_save(session_id, tid, tracker, char_probs, ocr_method, vote_summary, loop)


def process_frames(
    source: FrameSource,
    emit: Callable[[dict], None],
    models: ModelBundle,
    *,
    session_id: str = "",
    loop: asyncio.AbstractEventLoop | None = None,
    mjpeg_queue: asyncio.Queue | None = None,
    record_save: Callable | None = None,
) -> dict:
    """Run the full ALPR pipeline on a FrameSource.

    Returns: {total_vehicles, processed_frames}.
    """
    def emit_frame(jpg: bytes) -> None:
        if mjpeg_queue is not None and loop is not None:
            loop.call_soon_threadsafe(_safe_put, mjpeg_queue, jpg)

    tracker = WebTrackletManager()
    associator = TrajectoryAssociator(match_frames=5, agreement_ratio=0.6)
    models.vehicle_tracker.reset()

    total = source.total_frames or 0
    previously_tracked: set[int] = set()
    frame_idx = 0

    for src_idx, frame, _ts in source.iter_frames():
        frame_idx = src_idx + 1  # 1-based to match legacy run_job

        v_pred = models.vehicle.predict(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        if v_pred.boxes is not None and len(v_pred.boxes) > 0:
            xyxy = v_pred.boxes.xyxy.cpu().numpy()
            conf = v_pred.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = v_pred.boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)

        boxes, ids, classes = models.vehicle_tracker.track(dets, frame)

        tracked: list[dict] = []
        currently_tracked: set[int] = set()
        for box, tid, cid in zip(boxes, ids, classes):
            tid = int(tid)
            tracker._cls[tid] = models.vehicle.names[int(cid)]
            tracked.append({"id": tid, "box": box.tolist()})
            currently_tracked.add(tid)
            if tid in tracker._lost_count:
                tracker.reset_lost(tid)

        if frame_idx % 10 == 0 or frame_idx == total:
            emit({
                "type": "progress",
                "frame": frame_idx,
                "total": total or frame_idx,
                "pct": round(frame_idx / max(total, frame_idx) * 100, 1),
            })

        if frame_idx % FRAME_STRIDE != 0:
            previously_tracked = currently_tracked
            continue

        for tid in previously_tracked - currently_tracked:
            if (
                tracker.should_ocr(tid)
                and tracker.mark_lost(tid)
                and tracker.ready_for_multiframe_ocr(tid)
            ):
                _run_multiframe_ocr(tid, tracker, models, emit, session_id, loop, record_save)

        p_res = models.plate.track(frame, persist=True, tracker=_PLATE_TRACKER_CFG, verbose=False)[0]

        active_tids: set[int] = set()
        matched: list[tuple[int, np.ndarray, np.ndarray]] = []
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
                plate_crop = frame[ry : ry + rh, rx : rx + rw]
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
            if v_box is not None:
                vehicle_crop = _crop_vehicle(frame, v_box)
                matched.append((v_tid, p["crop"], vehicle_crop))

        for tid, plate_crop, vehicle_crop in matched:
            if not tracker.should_ocr(tid):
                continue
            q = quality_score(plate_crop)
            tracker.buffer_crop(tid, plate_crop, q, frame_idx)
            tracker.update_vehicle_img(tid, vehicle_crop, q)
            active_tids.add(tid)

        if mjpeg_queue is not None:
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

    # ── Finalise remaining buffered tracks ────────────────────────────────────
    for tid in list(tracker._buffers):
        if tracker.should_ocr(tid) and tracker.ready_for_multiframe_ocr(tid):
            _run_multiframe_ocr(tid, tracker, models, emit, session_id, loop, record_save)

    # ── Final snapshot ────────────────────────────────────────────────────────
    for tid in sorted(tracker._best):
        emit({
            "type": "vehicle",
            "id": tid,
            "cls": tracker._cls.get(tid, ""),
            "plate": tracker.display_text(tid),
            "chars": tracker.chars_json(tid),
            "done": tracker._done.get(tid, False),
            "plate_b64": tracker.plate_b64(tid),
            "vehicle_b64": tracker.vehicle_b64(tid),
            "ocr_frames": tracker.ocr_frames(tid),
            "final": True,
        })

    return {
        "total_vehicles": len(tracker._best),
        "processed_frames": frame_idx,
    }
```

- [ ] **Step 3: Refactor `run_job` in `pipeline.py` to delegate**

Replace the body of `run_job` in `api/core/pipeline.py` (keep imports and the `_record_save`, `_session_create`, `_session_update` helpers as-is). Replace from the line `def run_job(` to the end of the file with:

```python
def run_job(
    video_path: str,
    job_id: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    models: ModelBundle,
    jobs: dict,
    filename: str = "video.mp4",
    mjpeg_queue: asyncio.Queue | None = None,
) -> None:
    """Legacy upload-and-process-whole-video entry point. Thin wrapper around
    pipeline_core.process_frames; owns video file lifecycle + session row."""
    from .frame_source import FileFrameSource
    from .pipeline_core import process_frames

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    _session_create(job_id, filename, loop)

    try:
        source = FileFrameSource(video_path)
        summary = process_frames(
            source,
            emit=emit,
            models=models,
            session_id=job_id,
            loop=loop,
            mjpeg_queue=mjpeg_queue,
            record_save=_record_save,
        )
        emit({"type": "complete", "total_vehicles": summary["total_vehicles"]})
        _session_update(job_id, {
            "status": "completed",
            "total_records": summary["total_vehicles"],
            "processed_frames": summary["processed_frames"],
        }, loop)

    except Exception as exc:
        import traceback

        emit({"type": "error", "message": str(exc), "detail": traceback.format_exc()})
        _session_update(job_id, {"status": "failed", "error_message": str(exc)}, loop)

    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass
        jobs.pop(job_id, None)
        if mjpeg_queue is not None:
            loop.call_soon_threadsafe(_safe_put, mjpeg_queue, None)
```

Also delete the now-duplicate `_run_multiframe_ocr` and `_safe_put` definitions from `pipeline.py` (they now live in `pipeline_core.py`). Keep `_record_save`, `_session_create`, `_session_update` in `pipeline.py`.

- [ ] **Step 4: Run the parity test**

```bash
pytest tests/test_pipeline_core_parity.py -v
```

Expected: 1 PASS. If it fails, diff the captured vs golden event lists to find the divergence and fix `process_frames` before moving on. Do NOT regenerate the golden file — the whole point of the test is to detect drift.

- [ ] **Step 5: Run the whole test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Delete the capture script (one-off, no longer needed)**

```bash
rm tests/_capture_golden.py
```

- [ ] **Step 7: Commit**

```bash
git add api/core/pipeline_core.py api/core/pipeline.py tests/_capture_golden.py
git commit -m "refactor(core): extract process_frames; run_job delegates"
```

---

## Phase 3 — MediaMTX integration

### Task 3.1: Add MediaMTX service & config

**Files:**
- Create: `configs/mediamtx.yml`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Write the MediaMTX config**

Write to `configs/mediamtx.yml`:

```yaml
# MediaMTX runtime config for ALPR Incident Monitor.
# All RTSP camera paths are added dynamically by FastAPI via the HTTP API.

api: yes
apiAddress: :9997

webrtc: yes
webrtcAddress: :8889
webrtcAllowOrigin: '*'

rtspAddress: :8554
rtmp: no
hls: no

logLevel: info

paths: {}
```

- [ ] **Step 2: Add the service to docker-compose**

Open `docker-compose.yml`, locate the `services:` block, and add:

```yaml
  mediamtx:
    image: bluenviron/mediamtx:latest
    container_name: alpr-mediamtx
    ports:
      - "8554:8554"
      - "8889:8889"
      - "8189:8189/udp"
      - "9997:9997"
    volumes:
      - ./configs/mediamtx.yml:/mediamtx.yml
    restart: unless-stopped
```

- [ ] **Step 3: Verify MediaMTX starts and the API responds**

```bash
docker compose up -d mediamtx
curl -s http://localhost:9997/v3/paths/list | head -c 200
docker compose logs --tail=20 mediamtx
```

Expected: JSON list response (likely `{"itemCount":0,"pageCount":1,"items":[]}`). Logs should show "[RTSP] listener opened on :8554" and "[WebRTC] listener opened on :8889".

- [ ] **Step 4: Commit**

```bash
git add configs/mediamtx.yml docker-compose.yml
git commit -m "feat(infra): add MediaMTX service for RTSP↔WebRTC bridging"
```

### Task 3.2: Add MediaMTX HTTP client

**Files:**
- Create: `api/core/mediamtx_client.py`
- Create: `tests/test_mediamtx_client.py`

- [ ] **Step 1: Write the failing tests**

Write to `tests/test_mediamtx_client.py`:

```python
"""Tests for api/core/mediamtx_client.py."""
from __future__ import annotations

import httpx
import pytest

from api.core import mediamtx_client


@pytest.fixture
def mock_api(monkeypatch):
    """Patch the module-level httpx.Client with a MockTransport."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and "/v3/config/paths/add/" in str(request.url):
            return httpx.Response(200, json={"ok": True})
        if request.method == "DELETE" and "/v3/config/paths/delete/" in str(request.url):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://mediamtx:9997")
    monkeypatch.setattr(mediamtx_client, "_client", client)
    return requests


@pytest.mark.unit
def test_add_path_posts_correct_json(mock_api):
    mediamtx_client.add_path("live_abc", "rtsp://10.0.0.5/main")
    assert len(mock_api) == 1
    req = mock_api[0]
    assert req.method == "POST"
    assert str(req.url).endswith("/v3/config/paths/add/live_abc")
    import json as _json
    body = _json.loads(req.content.decode())
    assert body == {"source": "rtsp://10.0.0.5/main"}


@pytest.mark.unit
def test_remove_path_sends_delete(mock_api):
    mediamtx_client.remove_path("live_abc")
    assert len(mock_api) == 1
    assert mock_api[0].method == "DELETE"


@pytest.mark.unit
def test_remove_path_is_idempotent_on_404(monkeypatch):
    def handler(request):
        return httpx.Response(404, json={"error": "not found"})
    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://mediamtx:9997"
    )
    monkeypatch.setattr(mediamtx_client, "_client", client)
    mediamtx_client.remove_path("nonexistent")  # must NOT raise


@pytest.mark.unit
def test_add_path_raises_on_5xx(monkeypatch):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})
    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://mediamtx:9997"
    )
    monkeypatch.setattr(mediamtx_client, "_client", client)
    with pytest.raises(mediamtx_client.MediaMTXError):
        mediamtx_client.add_path("x", "rtsp://x/y")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_mediamtx_client.py -v
```

Expected: import error.

- [ ] **Step 3: Write the implementation**

Write to `api/core/mediamtx_client.py`:

```python
"""Thin HTTP client for the MediaMTX control API.

The MediaMTX API lets us add/remove paths at runtime, which is how we make
each Incident-Monitor session a separately-addressable stream.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://mediamtx:9997"
_API_URL = os.environ.get("MEDIAMTX_API_URL", _DEFAULT_API_URL)
_TIMEOUT = httpx.Timeout(5.0)

_client = httpx.Client(base_url=_API_URL, timeout=_TIMEOUT)


class MediaMTXError(RuntimeError):
    """Raised when the MediaMTX API returns an unexpected status."""


def add_path(name: str, source: str) -> None:
    """Register a new path that pulls from `source` (an RTSP URL)."""
    resp = _client.post(f"/v3/config/paths/add/{name}", json={"source": source})
    if resp.status_code >= 300:
        # Mask credentials in the logged URL
        safe = source.split("@")[-1] if "@" in source else source
        raise MediaMTXError(
            f"add_path({name}) failed: HTTP {resp.status_code} — source=…@{safe}"
        )


def remove_path(name: str) -> None:
    """Remove a path. Idempotent: 404 is ignored."""
    resp = _client.delete(f"/v3/config/paths/delete/{name}")
    if resp.status_code == 404:
        logger.debug("mediamtx: remove_path(%s) returned 404 (already gone)", name)
        return
    if resp.status_code >= 300:
        raise MediaMTXError(f"remove_path({name}) failed: HTTP {resp.status_code}")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_mediamtx_client.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/core/mediamtx_client.py tests/test_mediamtx_client.py
git commit -m "feat(core): add mediamtx_client (add_path / remove_path)"
```

---

## Phase 4 — Live session & rolling buffer

### Task 4.1: Implement `LiveSession`

**Files:**
- Create: `api/core/live_session.py`
- Create: `tests/test_live_session.py`

- [ ] **Step 1: Write the failing tests**

Write to `tests/test_live_session.py`:

```python
"""Tests for api/core/live_session.py — covers rolling buffer logic only.
Actual RTSP decoding is integration-tested manually (see E2E checklist)."""
from __future__ import annotations

import numpy as np
import pytest

from api.core.live_session import LiveSession


@pytest.mark.unit
def test_buffer_evicts_old_frames_at_maxlen():
    sess = LiveSession(session_id="t1", mediamtx_path="t1")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=1.0)  # maxlen=30
    for i in range(50):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    assert len(sess.frame_buffer) == 30
    # Oldest remaining frame should be index 20 (50 pushed, 30 retained)
    assert sess.frame_buffer[0][0] == 20


@pytest.mark.unit
def test_snapshot_returns_chronological_copy():
    sess = LiveSession(session_id="t2", mediamtx_path="t2")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=10.0)
    for i in range(10):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    snap = sess.snapshot_window(seconds=10.0)
    assert len(snap) == 10
    assert [f[0] for f in snap] == list(range(10))


@pytest.mark.unit
def test_snapshot_clamps_to_available():
    sess = LiveSession(session_id="t3", mediamtx_path="t3")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=10.0)
    for i in range(5):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    snap = sess.snapshot_window(seconds=10.0)
    assert len(snap) == 5  # only 5 available


@pytest.mark.unit
def test_snapshot_is_decoupled_from_buffer():
    sess = LiveSession(session_id="t4", mediamtx_path="t4")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=10.0)
    for i in range(10):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    snap = sess.snapshot_window(seconds=10.0)
    # Mutating snap must not affect the live buffer or vice versa
    snap.clear()
    assert len(sess.frame_buffer) == 10
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_live_session.py -v
```

Expected: import error.

- [ ] **Step 3: Write the implementation**

Write to `api/core/live_session.py`:

```python
"""LiveSession — per-monitor-session RTSP decoder + rolling frame buffer.

One decoder thread per session. Frames are appended to a bounded deque so the
last N seconds are always available. The same frames are JPEG-encoded and
pushed into an asyncio.Queue for the MJPEG fallback endpoint.
"""
from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Callable

import cv2
import numpy as np

from . import mediamtx_client

logger = logging.getLogger(__name__)

_INTERNAL_RTSP_BASE = os.environ.get("MEDIAMTX_INTERNAL_RTSP_BASE", "rtsp://mediamtx:8554")
_BUFFER_SECONDS = 10.0
_RECONNECT_RETRIES = 3
_RECONNECT_BACKOFF_SEC = 1.0


class LiveSession:
    """One live monitoring session. Owns the MediaMTX path and decoder thread."""

    def __init__(self, session_id: str, mediamtx_path: str) -> None:
        self.session_id = session_id
        self.mediamtx_path = mediamtx_path
        self.fps: float = 30.0
        self.frame_size: tuple[int, int] = (0, 0)
        self.frame_buffer: collections.deque = collections.deque()
        self.mjpeg_queue = None         # set on start()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_error: Callable[[str], None] | None = None

    # ── Test hooks ──────────────────────────────────────────────────────────
    def _init_buffer(self, fps: float, frame_size: tuple[int, int], seconds: float = _BUFFER_SECONDS) -> None:
        self.fps = fps
        self.frame_size = frame_size
        maxlen = max(1, int(fps * seconds))
        self.frame_buffer = collections.deque(maxlen=maxlen)

    def _push_frame(self, idx: int, frame: np.ndarray, ts: float) -> None:
        self.frame_buffer.append((idx, frame, ts))

    # ── Public API ──────────────────────────────────────────────────────────
    def start(self, rtsp_url: str, mjpeg_queue, on_error: Callable[[str], None] | None = None) -> None:
        """Register the MediaMTX path and spawn the decoder thread."""
        import asyncio  # local import — only needed at runtime

        mediamtx_client.add_path(self.mediamtx_path, rtsp_url)
        self.mjpeg_queue = mjpeg_queue
        self._on_error = on_error
        self._thread = threading.Thread(target=self._decoder_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        try:
            mediamtx_client.remove_path(self.mediamtx_path)
        except Exception:
            logger.exception("mediamtx remove_path failed for %s", self.mediamtx_path)

    def snapshot_window(self, seconds: float = _BUFFER_SECONDS) -> list[tuple[int, np.ndarray, float]]:
        """Return a chronologically-ordered shallow copy of the last `seconds` of buffered frames."""
        wanted = int(self.fps * seconds)
        snap = list(self.frame_buffer)
        if wanted < len(snap):
            snap = snap[-wanted:]
        return snap

    # ── Internal ────────────────────────────────────────────────────────────
    def _decoder_loop(self) -> None:
        url = f"{_INTERNAL_RTSP_BASE}/{self.mediamtx_path}"
        attempts = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                attempts += 1
                if attempts > _RECONNECT_RETRIES:
                    self._fail(f"Cannot open RTSP source via MediaMTX: {url}")
                    return
                time.sleep(_RECONNECT_BACKOFF_SEC)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            self._init_buffer(fps=fps, frame_size=(w, h), seconds=_BUFFER_SECONDS)

            idx = 0
            attempts = 0
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                ts = idx / fps
                self.frame_buffer.append((idx, frame, ts))
                if self.mjpeg_queue is not None:
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok and not self.mjpeg_queue.full():
                        try:
                            self.mjpeg_queue.put_nowait(bytes(jpg))
                        except Exception:
                            pass
                idx += 1

            cap.release()
            # Loop back to retry connection unless stopped
            attempts += 1
            if attempts > _RECONNECT_RETRIES:
                self._fail("RTSP stream lost (retry budget exhausted)")
                return
            time.sleep(_RECONNECT_BACKOFF_SEC)

    def _fail(self, msg: str) -> None:
        logger.error("LiveSession[%s]: %s", self.session_id, msg)
        if self._on_error is not None:
            try:
                self._on_error(msg)
            except Exception:
                logger.exception("LiveSession error callback raised")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_live_session.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/core/live_session.py tests/test_live_session.py
git commit -m "feat(core): add LiveSession with rolling 10s frame buffer"
```

---

## Phase 5 — Incident analyzer + persistence

### Task 5.1: Add `Incident` and `IncidentVehicle` Pydantic models

**Files:**
- Modify: `api/database/models.py` (append classes)
- Create: `tests/test_incident_models.py`

- [ ] **Step 1: Write the failing tests**

Write to `tests/test_incident_models.py`:

```python
"""Tests for the Incident / IncidentVehicle Pydantic models."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from api.database.models import Incident, IncidentVehicle


@pytest.mark.unit
def test_incident_vehicle_minimal_construction():
    v = IncidentVehicle(
        track_id=7,
        plate_text="30A-12345",
        plate_text_confidence=0.94,
        chars=[("3", 0.99), ("0", 0.97)],
        vehicle_class="car",
        plate_image_url=None,
        vehicle_image_url=None,
        ocr_method="multiframe",
        ocr_frames=18,
        first_seen_frame=4,
        last_seen_frame=142,
    )
    assert v.track_id == 7
    assert v.ocr_method == "multiframe"


@pytest.mark.unit
def test_incident_default_status_and_lists():
    now = datetime.now(timezone.utc)
    i = Incident(
        incident_id="inc_abc",
        session_id="ses_xyz",
        source_type="live",
        source_ref="rtsp://10.0.0.5/main",
        marked_at=now,
        window_start_sec=0.0,
        window_end_sec=10.0,
        duration_sec=10.0,
        status="processing",
        created_at=now,
        updated_at=now,
    )
    assert i.vehicles == []
    assert i.total_vehicles == 0
    assert i.error_message is None


@pytest.mark.unit
def test_incident_rejects_bad_source_type():
    now = datetime.now(timezone.utc)
    with pytest.raises(Exception):
        Incident(
            incident_id="inc_abc",
            session_id="ses_xyz",
            source_type="invalid",  # not "live" or "upload"
            source_ref="x",
            marked_at=now,
            window_start_sec=0.0,
            window_end_sec=1.0,
            duration_sec=1.0,
            status="processing",
            created_at=now,
            updated_at=now,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_incident_models.py -v
```

Expected: ImportError.

- [ ] **Step 3: Append the models to `api/database/models.py`**

Append at the end of the existing file:

```python
# ── Incident-Monitor models ──────────────────────────────────────────────────


class IncidentVehicle(BaseModel):
    """One vehicle detected within an incident's analysis window."""

    track_id: int
    plate_text: str
    plate_text_confidence: float
    chars: list[tuple[str, float]]
    vehicle_class: str
    plate_image_url: str | None = None
    vehicle_image_url: str | None = None
    ocr_method: Literal["multiframe", "segment_vote", "prob_vote"]
    ocr_frames: int
    first_seen_frame: int
    last_seen_frame: int


class Incident(BaseModel):
    """A user-marked incident: a short window pulled out of a live stream
    or uploaded video for fast ALPR analysis."""

    incident_id: str
    session_id: str
    source_type: Literal["live", "upload"]
    source_ref: str
    marked_at: datetime
    window_start_sec: float
    window_end_sec: float
    duration_sec: float
    status: Literal["processing", "completed", "failed"]
    vehicles: list[IncidentVehicle] = Field(default_factory=list)
    total_vehicles: int = 0
    processing_ms: int | None = None
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
```

If `Literal` and `Field` are not already imported at the top of the file, add:

```python
from typing import Literal
from pydantic import Field
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_incident_models.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/database/models.py tests/test_incident_models.py
git commit -m "feat(db): add Incident and IncidentVehicle Pydantic models"
```

### Task 5.2: Add MongoDB CRUD for incidents

**Files:**
- Modify: `api/database/mongodb.py`
- Create: `tests/test_incident_crud.py` (skipped if no Mongo configured)

- [ ] **Step 1: Write the failing tests (Mongo-dependent, mark as integration)**

Write to `tests/test_incident_crud.py`:

```python
"""Integration tests for incidents collection CRUD.

Skipped automatically when MONGODB_URI is unset."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from api.database import mongodb
from api.database.models import Incident

pytestmark = pytest.mark.skipif(
    "MONGODB_URI" not in os.environ, reason="MONGODB_URI not set"
)


@pytest.fixture(scope="module")
async def db_initialised():
    await mongodb.init_db(os.environ["MONGODB_URI"], "alpr_test")
    yield
    await mongodb.close_db()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_and_get_incident(db_initialised):
    now = datetime.now(timezone.utc)
    inc = Incident(
        incident_id="inc_test_1",
        session_id="ses_test",
        source_type="live",
        source_ref="rtsp://localhost/test",
        marked_at=now,
        window_start_sec=0.0,
        window_end_sec=10.0,
        duration_sec=10.0,
        status="processing",
        created_at=now,
        updated_at=now,
    )
    await mongodb.upsert_incident(inc)
    fetched = await mongodb.get_incident("inc_test_1")
    assert fetched is not None
    assert fetched.incident_id == "inc_test_1"
    assert fetched.status == "processing"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_incidents_filters(db_initialised):
    items = await mongodb.list_incidents(source_type="live", limit=10)
    assert all(i.source_type == "live" for i in items)
```

- [ ] **Step 2: Run the tests (skipped is fine if Mongo unavailable)**

```bash
pytest tests/test_incident_crud.py -v
```

Expected: 2 skipped (without MONGODB_URI) OR import error (without `upsert_incident`).

- [ ] **Step 3: Extend `api/database/mongodb.py`**

Add the `INCIDENTS_COL` constant near the existing collection constants:

```python
INCIDENTS_COL = "incidents"
```

In `_ensure_indexes`, add (after the existing `create_indexes` calls):

```python
    await db[INCIDENTS_COL].create_indexes([
        IndexModel([("incident_id", ASCENDING)], unique=True, name="uq_incident_id"),
        IndexModel([("session_id", ASCENDING)], name="ix_session_id"),
        IndexModel([("marked_at", DESCENDING)], name="ix_marked_at_desc"),
        IndexModel([("status", ASCENDING)], name="ix_status"),
        IndexModel([("source_type", ASCENDING)], name="ix_source_type"),
    ])
```

Update the import line at the top of the file:

```python
from .models import Incident, RecognitionRecord, RecognitionSession
```

Append these CRUD functions at the bottom of the file:

```python
# ── Incident CRUD ─────────────────────────────────────────────────────────────


async def upsert_incident(incident: Incident) -> None:
    """Insert or replace an incident document, matched by incident_id."""
    from datetime import datetime, timezone

    db = get_db()
    doc = incident.model_dump(by_alias=True)
    doc["updated_at"] = datetime.now(timezone.utc)
    await db[INCIDENTS_COL].update_one(
        {"incident_id": incident.incident_id},
        {"$set": doc},
        upsert=True,
    )


async def get_incident(incident_id: str) -> Incident | None:
    db = get_db()
    doc = await db[INCIDENTS_COL].find_one({"incident_id": incident_id})
    return Incident.model_validate(doc) if doc else None


async def list_incidents(
    *,
    session_id: str | None = None,
    source_type: str | None = None,
    limit: int = 50,
) -> list[Incident]:
    db = get_db()
    query: dict = {}
    if session_id is not None:
        query["session_id"] = session_id
    if source_type is not None:
        query["source_type"] = source_type
    cursor = db[INCIDENTS_COL].find(query).sort("marked_at", DESCENDING).limit(limit)
    return [Incident.model_validate(doc) async for doc in cursor]
```

- [ ] **Step 4: Run the tests (still skipped if no Mongo, but no import errors now)**

```bash
pytest tests/test_incident_crud.py -v
```

Expected: 2 skipped if `MONGODB_URI` is unset; 2 PASS if it is set.

- [ ] **Step 5: Commit**

```bash
git add api/database/mongodb.py tests/test_incident_crud.py
git commit -m "feat(db): add incidents collection CRUD + indexes"
```

### Task 5.3: Implement `incident_analyzer.run_incident`

**Files:**
- Create: `api/core/incident_analyzer.py`
- Create: `tests/test_incident_analyzer.py`

- [ ] **Step 1: Write the failing tests**

Write to `tests/test_incident_analyzer.py`:

```python
"""Tests for api/core/incident_analyzer.py.

Uses a fake ModelBundle + the short_clip fixture so the analyzer's event
translation can be verified end-to-end without real GPU inference."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from api.core.frame_source import FileFrameSource


@pytest.mark.unit
def test_run_incident_emits_started_and_complete(monkeypatch):
    """run_incident wraps process_frames; verify it emits incident_started
    and incident_complete with the correct incident_id, regardless of
    intermediate events."""
    from api.core import incident_analyzer

    events: list[dict] = []
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def fake_process_frames(source, emit, models, **kwargs):
        # Mimic the real pipeline_core: emit a vehicle event then return a summary
        emit({"type": "vehicle", "id": 7, "plate": "30A-12345",
              "chars": [["3", 0.99]], "cls": "car",
              "plate_b64": "", "vehicle_b64": "", "ocr_frames": 5})
        return {"total_vehicles": 1, "processed_frames": 30}

    monkeypatch.setattr(incident_analyzer, "process_frames", fake_process_frames)
    monkeypatch.setattr(incident_analyzer, "_persist_incident", lambda *a, **kw: None)

    async def drain() -> None:
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.2)
                events.append(ev)
            except asyncio.TimeoutError:
                return

    incident_analyzer.run_incident(
        incident_id="inc_test",
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
    assert "incident_started" in types
    assert "incident_vehicle" in types
    assert "incident_complete" in types
    assert all(e.get("incident_id") == "inc_test" for e in events
               if e["type"].startswith("incident_"))


@pytest.mark.unit
def test_run_incident_translates_vehicle_event(monkeypatch):
    from api.core import incident_analyzer

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

    monkeypatch.setattr(incident_analyzer, "process_frames", fake_process_frames)
    monkeypatch.setattr(incident_analyzer, "_persist_incident", lambda *a, **kw: None)

    incident_analyzer.run_incident(
        incident_id="inc_xlate",
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

    veh = [e for e in events if e["type"] == "incident_vehicle"]
    rej = [e for e in events if e["type"] == "incident_rejected_vehicle"]
    assert len(veh) == 1 and veh[0]["plate"] == "30A-12345"
    assert len(rej) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_incident_analyzer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the implementation**

Write to `api/core/incident_analyzer.py`:

```python
"""incident_analyzer — orchestrates a single mark→analysis job.

Wraps pipeline_core.process_frames, translates its event types into
incident_* events so the SSE consumer can route them to the right card,
and persists the result to the `incidents` MongoDB collection.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Literal

import cv2

from .frame_source import FrameSource
from .models import ModelBundle
from .pipeline_core import process_frames

logger = logging.getLogger(__name__)


_EVENT_MAP = {
    "vehicle": "incident_vehicle",
    "rejected_vehicle": "incident_rejected_vehicle",
    "progress": "incident_progress",
}


def _persist_incident(
    *,
    incident_id: str,
    session_id: str,
    source_type: str,
    source_ref: str,
    window_start_sec: float,
    window_end_sec: float,
    status: str,
    vehicles: list[dict],
    rejected: list[dict],
    processing_ms: int,
    error_message: str | None,
    marked_at: datetime,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Build an Incident document and fire-and-forget save it."""
    try:
        from ..database import mongodb
        from ..database.models import Incident, IncidentVehicle
        from . import database as core_db

        if not mongodb.is_db_configured():
            return

        # Promote vehicle events into IncidentVehicle docs, uploading any
        # base64 thumbs to Supabase Storage if present.
        vehicle_docs: list[IncidentVehicle] = []
        for v in vehicles:
            plate_url = None
            vehicle_url = None
            if v.get("plate_b64"):
                import base64
                plate_url = core_db.upload_image(
                    "evidence",
                    f"incidents/{incident_id}/plate_{v['id']}.jpg",
                    base64.b64decode(v["plate_b64"]),
                )
            if v.get("vehicle_b64"):
                import base64
                vehicle_url = core_db.upload_image(
                    "evidence",
                    f"incidents/{incident_id}/vehicle_{v['id']}.jpg",
                    base64.b64decode(v["vehicle_b64"]),
                )
            conf = (
                sum(p for _, p in v.get("chars", [])) / max(1, len(v.get("chars", [])))
                if v.get("chars") else 0.0
            )
            vehicle_docs.append(IncidentVehicle(
                track_id=int(v["id"]),
                plate_text=v["plate"],
                plate_text_confidence=round(conf, 4),
                chars=[(c, float(p)) for c, p in v.get("chars", [])],
                vehicle_class=v.get("cls", "vehicle"),
                plate_image_url=plate_url,
                vehicle_image_url=vehicle_url,
                ocr_method="multiframe",
                ocr_frames=int(v.get("ocr_frames", 0)),
                first_seen_frame=0,
                last_seen_frame=int(v.get("ocr_frames", 0)),
            ))

        now = datetime.now(timezone.utc)
        incident = Incident(
            incident_id=incident_id,
            session_id=session_id,
            source_type=source_type,
            source_ref=source_ref,
            marked_at=marked_at,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            duration_sec=window_end_sec - window_start_sec,
            status=status,
            vehicles=vehicle_docs,
            total_vehicles=len(vehicle_docs),
            processing_ms=processing_ms,
            created_at=marked_at,
            updated_at=now,
            error_message=error_message,
        )
        asyncio.run_coroutine_threadsafe(mongodb.upsert_incident(incident), loop)
    except Exception:
        logger.exception("Failed to persist incident %s", incident_id)


def run_incident(
    *,
    incident_id: str,
    session_id: str,
    source: FrameSource,
    source_type: Literal["live", "upload"],
    source_ref: str,
    window_start_sec: float,
    window_end_sec: float,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    models: ModelBundle,
) -> None:
    """Run a single incident analysis. Thread-pool entry point."""
    marked_at = datetime.now(timezone.utc)
    started = time.monotonic()

    vehicles: list[dict] = []
    rejected: list[dict] = []

    def emit_raw(ev: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    emit_raw({
        "type": "incident_started",
        "incident_id": incident_id,
        "session_id": session_id,
        "source_type": source_type,
        "window_start_sec": window_start_sec,
        "window_end_sec": window_end_sec,
        "frames_count": source.total_frames or 0,
    })

    def emit_translated(ev: dict) -> None:
        t = ev.get("type", "")
        new_type = _EVENT_MAP.get(t, t)
        out = {**ev, "type": new_type, "incident_id": incident_id}
        if t == "vehicle":
            vehicles.append(ev)
        elif t == "rejected_vehicle":
            rejected.append(ev)
        emit_raw(out)

    try:
        summary = process_frames(
            source,
            emit=emit_translated,
            models=models,
            session_id="",   # we persist via _persist_incident, not _record_save
            loop=None,
        )
        dur_ms = int((time.monotonic() - started) * 1000)
        _persist_incident(
            incident_id=incident_id,
            session_id=session_id,
            source_type=source_type,
            source_ref=source_ref,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            status="completed",
            vehicles=vehicles,
            rejected=rejected,
            processing_ms=dur_ms,
            error_message=None,
            marked_at=marked_at,
            loop=loop,
        )
        emit_raw({
            "type": "incident_complete",
            "incident_id": incident_id,
            "total_vehicles": summary["total_vehicles"],
            "duration_ms": dur_ms,
            "status": "completed",
        })

    except Exception as exc:
        logger.exception("incident %s failed", incident_id)
        dur_ms = int((time.monotonic() - started) * 1000)
        _persist_incident(
            incident_id=incident_id,
            session_id=session_id,
            source_type=source_type,
            source_ref=source_ref,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            status="failed",
            vehicles=vehicles,
            rejected=rejected,
            processing_ms=dur_ms,
            error_message=str(exc),
            marked_at=marked_at,
            loop=loop,
        )
        emit_raw({
            "type": "incident_error",
            "incident_id": incident_id,
            "message": str(exc),
        })
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_incident_analyzer.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/core/incident_analyzer.py tests/test_incident_analyzer.py
git commit -m "feat(core): add incident_analyzer (mark→analysis→persist)"
```

---

## Phase 6 — HTTP routes

### Task 6.1: Add monitor router module + upload route

**Files:**
- Create: `api/routes_monitor.py`
- Modify: `api/main.py` (mount the router)
- Create: `tests/test_monitor_routes.py`

- [ ] **Step 1: Write the failing tests**

Write to `tests/test_monitor_routes.py`:

```python
"""Tests for /monitor/* HTTP routes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Spin up the FastAPI app with models mocked out so we don't load YOLO/etc."""
    from api.main import app
    return TestClient(app)


@pytest.mark.integration
def test_monitor_upload_returns_session_id(client, tmp_path):
    fixture = Path("tests/fixtures/short_clip.mp4")
    with open(fixture, "rb") as f:
        resp = client.post(
            "/monitor/upload",
            files={"file": ("short_clip.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert "video_url" in body
    assert body["video_url"].startswith("/monitor/upload/")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_monitor_routes.py::test_monitor_upload_returns_session_id -v
```

Expected: 404 (route doesn't exist).

- [ ] **Step 3: Implement the upload route**

Write to `api/routes_monitor.py`:

```python
"""HTTP routes for the Incident Monitor feature.

Mounted under /monitor and /incidents by api/main.py.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level registries; main.py initialises them.
monitor_sessions: dict[str, dict] = {}   # session_id → {"kind", "path"|"live_session", ...}
incident_queues: dict[str, asyncio.Queue] = {}   # session_id → SSE queue


def _new_session_id() -> str:
    return f"mon_{uuid.uuid4().hex[:10]}"


def _new_incident_id() -> str:
    return f"inc_{uuid.uuid4().hex[:10]}"


@router.post("/monitor/upload")
async def monitor_upload(file: UploadFile = File(...)) -> dict:
    """Accept a video file for monitor-mode playback + mark-driven analysis."""
    session_id = _new_session_id()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        tmp_path = f.name

    monitor_sessions[session_id] = {
        "kind": "upload",
        "path": tmp_path,
        "filename": file.filename or "video.mp4",
    }
    incident_queues[session_id] = asyncio.Queue()
    return {
        "session_id": session_id,
        "video_url": f"/monitor/upload/{session_id}/video",
    }


@router.get("/monitor/upload/{session_id}/video")
async def monitor_upload_video(session_id: str) -> FileResponse:
    sess = monitor_sessions.get(session_id)
    if sess is None or sess["kind"] != "upload":
        raise HTTPException(status_code=404, detail="Session not found")
    return FileResponse(sess["path"], media_type="video/mp4")
```

In `api/main.py`, add at the top with the other imports:

```python
from api import routes_monitor
```

And just before the `if DIST_DIR.exists():` block, add:

```python
app.include_router(routes_monitor.router)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_monitor_routes.py::test_monitor_upload_returns_session_id -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes_monitor.py api/main.py tests/test_monitor_routes.py
git commit -m "feat(api): add /monitor/upload route + session registry"
```

### Task 6.2: Add live-connect / disconnect routes

**Files:**
- Modify: `api/routes_monitor.py`
- Modify: `tests/test_monitor_routes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_monitor_routes.py`:

```python
@pytest.mark.integration
def test_monitor_live_connect_rejects_non_rtsp_scheme(client):
    resp = client.post("/monitor/live/connect", json={"rtsp_url": "http://evil/path"})
    assert resp.status_code == 400


@pytest.mark.integration
def test_monitor_live_connect_returns_urls(client, monkeypatch):
    """Mock LiveSession.start so we don't actually hit a camera."""
    from api import routes_monitor

    def fake_start(self, rtsp_url, mjpeg_queue, on_error=None):
        pass

    from api.core.live_session import LiveSession
    monkeypatch.setattr(LiveSession, "start", fake_start)

    resp = client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://10.0.0.5/main"})
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert "whep_url" in body
    assert "mjpeg_url" in body
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_monitor_routes.py -v -k live_connect
```

Expected: 404.

- [ ] **Step 3: Implement live-connect / disconnect**

Append to `api/routes_monitor.py`:

```python
import os
from urllib.parse import urlparse

from pydantic import BaseModel

from api.core.live_session import LiveSession

_WEBRTC_PUBLIC_BASE = os.environ.get("MEDIAMTX_PUBLIC_WEBRTC_BASE", "http://localhost:8889")
_MJPEG_PUBLIC_BASE = os.environ.get("MEDIAMTX_PUBLIC_MJPEG_BASE", "")  # relative if blank


class ConnectBody(BaseModel):
    rtsp_url: str


@router.post("/monitor/live/connect")
async def monitor_live_connect(body: ConnectBody) -> dict:
    parsed = urlparse(body.rtsp_url)
    if parsed.scheme not in ("rtsp", "rtsps"):
        raise HTTPException(status_code=400, detail="URL must be rtsp:// or rtsps://")

    session_id = _new_session_id()
    path = f"live_{session_id[4:]}"  # MediaMTX path name
    mjpeg_q: asyncio.Queue = asyncio.Queue(maxsize=60)

    sess = LiveSession(session_id=session_id, mediamtx_path=path)
    try:
        sess.start(body.rtsp_url, mjpeg_queue=mjpeg_q)
    except Exception as exc:
        logger.exception("Live connect failed")
        raise HTTPException(status_code=502, detail=f"Could not connect: {exc}")

    monitor_sessions[session_id] = {
        "kind": "live",
        "live_session": sess,
        "mediamtx_path": path,
        "mjpeg_queue": mjpeg_q,
        "rtsp_url": body.rtsp_url,
    }
    incident_queues[session_id] = asyncio.Queue()

    whep_url = f"{_WEBRTC_PUBLIC_BASE}/{path}/whep"
    mjpeg_url = f"{_MJPEG_PUBLIC_BASE}/monitor/live/{session_id}/mjpeg"
    return {"session_id": session_id, "whep_url": whep_url, "mjpeg_url": mjpeg_url}


@router.delete("/monitor/live/{session_id}")
async def monitor_live_disconnect(session_id: str) -> dict:
    sess = monitor_sessions.pop(session_id, None)
    incident_queues.pop(session_id, None)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess["kind"] != "live":
        raise HTTPException(status_code=400, detail="Not a live session")
    sess["live_session"].stop()
    return {"ok": True}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_monitor_routes.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes_monitor.py tests/test_monitor_routes.py
git commit -m "feat(api): add /monitor/live/connect and disconnect routes"
```

### Task 6.3: Add MJPEG-fallback route

**Files:**
- Modify: `api/routes_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_monitor_routes.py`:

```python
@pytest.mark.integration
def test_monitor_live_mjpeg_returns_multipart(client, monkeypatch):
    from api import routes_monitor
    from api.core.live_session import LiveSession

    monkeypatch.setattr(LiveSession, "start", lambda self, *a, **kw: None)

    # Connect, then push a single JPEG into the queue
    resp = client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://x/y"})
    sid = resp.json()["session_id"]
    routes_monitor.monitor_sessions[sid]["mjpeg_queue"].put_nowait(b"\xff\xd8fake")

    with client.stream("GET", f"/monitor/live/{sid}/mjpeg") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        # Just read one chunk and bail
        for chunk in r.iter_bytes():
            assert b"image/jpeg" in chunk
            break
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_monitor_routes.py -v -k mjpeg
```

Expected: 404.

- [ ] **Step 3: Implement the MJPEG route**

Append to `api/routes_monitor.py`:

```python
@router.get("/monitor/live/{session_id}/mjpeg")
async def monitor_live_mjpeg(session_id: str) -> StreamingResponse:
    sess = monitor_sessions.get(session_id)
    if sess is None or sess["kind"] != "live":
        raise HTTPException(status_code=404, detail="Session not found")
    mjpeg_q: asyncio.Queue = sess["mjpeg_queue"]

    async def gen():
        while True:
            try:
                frame = await asyncio.wait_for(mjpeg_q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            if frame is None:
                break
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/test_monitor_routes.py -v -k mjpeg
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes_monitor.py tests/test_monitor_routes.py
git commit -m "feat(api): add /monitor/live/{id}/mjpeg fallback endpoint"
```

### Task 6.4: Add `/monitor/{session_id}/mark` route

**Files:**
- Modify: `api/routes_monitor.py`
- Modify: `tests/test_monitor_routes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_monitor_routes.py`:

```python
@pytest.mark.integration
def test_mark_upload_validates_window_too_long(client):
    resp = client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4","rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = client.post(f"/monitor/{sid}/mark",
        json={"mode": "upload", "t_start": 0.0, "t_end": 999.0})
    assert resp.status_code == 400


@pytest.mark.integration
def test_mark_upload_accepts_valid_window(client, monkeypatch):
    from api import routes_monitor

    monkeypatch.setattr(routes_monitor, "_dispatch_incident", lambda *a, **kw: None)

    resp = client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4","rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = client.post(f"/monitor/{sid}/mark",
        json={"mode": "upload", "t_start": 0.0, "t_end": 1.0})
    assert resp.status_code == 200
    assert "incident_id" in resp.json()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_monitor_routes.py -v -k mark
```

Expected: 404.

- [ ] **Step 3: Implement the mark route**

Append to `api/routes_monitor.py`:

```python
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

# Single-worker pool — only one mark analyzes at a time to avoid GPU contention.
_incident_executor = ThreadPoolExecutor(max_workers=1)

MAX_INTERVAL_SEC = 30.0


class MarkBody(BaseModel):
    mode: Literal["live", "upload"]
    t_start: float | None = None
    t_end: float | None = None


def _dispatch_incident(
    *,
    incident_id: str,
    session_id: str,
    sess: dict,
    body: MarkBody,
    queue: asyncio.Queue,
    request: Request,
) -> None:
    """Build a FrameSource and submit run_incident to the worker pool."""
    from api.core.frame_source import FileFrameSource, LiveBufferFrameSource
    from api.core.incident_analyzer import run_incident

    models = request.app.state.models if hasattr(request.app.state, "models") else None
    # main.py sets app.state.models in lifespan — see Task 7.1

    loop = asyncio.get_event_loop()

    if body.mode == "upload":
        source = FileFrameSource(sess["path"], t_start=body.t_start, t_end=body.t_end)
        source_ref = sess["filename"]
        ws, we = body.t_start, body.t_end
    else:
        live_sess = sess["live_session"]
        snap = live_sess.snapshot_window(seconds=10.0)
        if len(snap) < int(live_sess.fps * 1.0):
            raise HTTPException(status_code=409, detail="Buffer still warming up — wait 1–2s")
        source = LiveBufferFrameSource(snap, fps=live_sess.fps, frame_size=live_sess.frame_size)
        source_ref = sess["rtsp_url"]
        ws = snap[0][2]
        we = snap[-1][2]

    _incident_executor.submit(
        run_incident,
        incident_id=incident_id,
        session_id=session_id,
        source=source,
        source_type=body.mode,
        source_ref=source_ref,
        window_start_sec=ws,
        window_end_sec=we,
        queue=queue,
        loop=loop,
        models=models,
    )


@router.post("/monitor/{session_id}/mark")
async def monitor_mark(session_id: str, body: MarkBody, request: Request) -> dict:
    sess = monitor_sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if body.mode == "upload":
        if sess["kind"] != "upload":
            raise HTTPException(status_code=400, detail="Session is live; expected upload mark")
        if body.t_start is None or body.t_end is None:
            raise HTTPException(status_code=400, detail="t_start and t_end required for upload mark")
        if not (0.0 <= body.t_start < body.t_end):
            raise HTTPException(status_code=400, detail="Invalid interval")
        if body.t_end - body.t_start > MAX_INTERVAL_SEC:
            raise HTTPException(status_code=400, detail=f"Interval exceeds {MAX_INTERVAL_SEC}s max")
    else:
        if sess["kind"] != "live":
            raise HTTPException(status_code=400, detail="Session is upload; expected live mark")

    incident_id = _new_incident_id()
    queue = incident_queues[session_id]
    _dispatch_incident(
        incident_id=incident_id,
        session_id=session_id,
        sess=sess,
        body=body,
        queue=queue,
        request=request,
    )
    return {"incident_id": incident_id}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_monitor_routes.py -v -k mark
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes_monitor.py tests/test_monitor_routes.py
git commit -m "feat(api): add /monitor/{id}/mark route with window validation"
```

### Task 6.5: Add incident SSE stream + GET endpoints

**Files:**
- Modify: `api/routes_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_monitor_routes.py`:

```python
@pytest.mark.integration
def test_incident_stream_yields_text_event_stream(client, monkeypatch):
    from api import routes_monitor

    monkeypatch.setattr(routes_monitor, "_dispatch_incident", lambda *a, **kw: None)

    resp = client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4","rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    # Push a fake event into the queue
    routes_monitor.incident_queues[sid].put_nowait({"type": "incident_progress", "pct": 50})

    with client.stream("GET", f"/monitor/{sid}/incidents/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for chunk in r.iter_bytes():
            assert b"incident_progress" in chunk
            break
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_monitor_routes.py -v -k incident_stream
```

Expected: 404.

- [ ] **Step 3: Implement the SSE + GET routes**

Append to `api/routes_monitor.py`:

```python
import json


@router.get("/monitor/{session_id}/incidents/stream")
async def monitor_incidents_stream(session_id: str) -> StreamingResponse:
    queue = incident_queues.get(session_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def gen():
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=60.0)
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/incidents/{incident_id}")
async def get_incident_route(incident_id: str) -> dict:
    from api.database.mongodb import get_incident, is_db_configured

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    inc = await get_incident(incident_id)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc.model_dump(mode="json")


@router.get("/incidents")
async def list_incidents_route(
    source: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
) -> dict:
    from api.database.mongodb import list_incidents, is_db_configured

    if not is_db_configured():
        return {"items": []}
    items = await list_incidents(session_id=session_id, source_type=source, limit=limit)
    return {"items": [i.model_dump(mode="json") for i in items]}
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_monitor_routes.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routes_monitor.py tests/test_monitor_routes.py
git commit -m "feat(api): add incident SSE stream + GET /incidents endpoints"
```

### Task 6.6: Wire `models` into `app.state`

**Files:**
- Modify: `api/main.py`

- [ ] **Step 1: Replace the lifespan + the in-module global**

Find the `lifespan` function in `api/main.py` and the line `_models: ModelBundle | None = None`. Update so models are exposed via `app.state.models`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
    if MONGODB_URI:
        await init_db(MONGODB_URI, MONGODB_DB_NAME)
    else:
        logger.warning("MONGODB_URI not set — database persistence disabled.")
    yield
    await close_db()
```

Remove the now-unused module-level `_models` variable. Update the existing `/upload` route's `run_job` dispatch to use `request.app.state.models` instead (add `request: Request` param). Concretely, the existing `/upload` becomes:

```python
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    job_id      = uuid.uuid4().hex[:8]
    queue       = asyncio.Queue()
    mjpeg_queue = asyncio.Queue(maxsize=60)
    _jobs[job_id]         = queue
    _mjpeg_queues[job_id] = mjpeg_queue

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        tmp = f.name

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, run_job, tmp, job_id, queue, loop, request.app.state.models, _jobs,
        file.filename or "video.mp4", mjpeg_queue,
    )
    return {"job_id": job_id}
```

Also add `Request` to the existing imports: `from fastapi import FastAPI, File, HTTPException, Request, UploadFile`.

- [ ] **Step 2: Run the existing tests**

```bash
pytest tests/ -v
```

Expected: all PASS, including legacy upload-flow tests.

- [ ] **Step 3: Commit**

```bash
git add api/main.py
git commit -m "refactor(api): expose ModelBundle via app.state for route reuse"
```

---

## Phase 7 — Frontend: mode switch & page skeleton

### Task 7.1: Add a `mode` switch in App.jsx

**Files:**
- Modify: `web/src/App.jsx`

- [ ] **Step 1: Wrap the current page body in a mode-driven conditional**

At the top of `App.jsx`, add the `mode` state:

```jsx
const [mode, setMode] = useState('process')   // 'process' | 'monitor'
```

Add a two-tab nav in the header (replace the existing brand block content with):

```jsx
<div className="ml-6 flex items-center gap-1">
  <button
    onClick={() => setMode('process')}
    className={`text-xs px-3 py-1.5 rounded-md transition-colors ${
      mode === 'process' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
    }`}
  >
    Xử lý video
  </button>
  <button
    onClick={() => setMode('monitor')}
    className={`text-xs px-3 py-1.5 rounded-md transition-colors ${
      mode === 'monitor' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
    }`}
  >
    Giám sát sự cố
  </button>
</div>
```

Then wrap the existing main two-column block in `{mode === 'process' && ( … )}` and add `{mode === 'monitor' && <MonitorPage />}` next to it. Add the import:

```jsx
import MonitorPage from './components/monitor/MonitorPage'
```

- [ ] **Step 2: Add a placeholder `MonitorPage`**

Create `web/src/components/monitor/MonitorPage.jsx`:

```jsx
export default function MonitorPage() {
  return (
    <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5 text-slate-300">
      <p className="text-sm">Incident Monitor — coming up in next tasks.</p>
    </div>
  )
}
```

- [ ] **Step 3: Verify the dev server loads both tabs**

```bash
cd web && npm run dev
```

Open the app, click `Giám sát sự cố` — should render the placeholder. Click back — should restore the upload flow.

- [ ] **Step 4: Commit**

```bash
git add web/src/App.jsx web/src/components/monitor/MonitorPage.jsx
git commit -m "feat(web): add mode switch (Xử lý video ↔ Giám sát sự cố)"
```

### Task 7.2: Source selector + state lifting in MonitorPage

**Files:**
- Modify: `web/src/components/monitor/MonitorPage.jsx`
- Create: `web/src/components/monitor/SourceSelector.jsx`

- [ ] **Step 1: Implement `SourceSelector`**

Write to `web/src/components/monitor/SourceSelector.jsx`:

```jsx
import { useState } from 'react'
import DropZone from '../DropZone'

export default function SourceSelector({ onConnectLive, onSelectFile }) {
  const [tab, setTab] = useState('rtsp')
  const [url, setUrl] = useState('')

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
      <div className="flex items-center gap-1 mb-3">
        <button
          onClick={() => setTab('rtsp')}
          className={`text-xs px-3 py-1.5 rounded ${
            tab === 'rtsp' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
          }`}
        >
          RTSP camera
        </button>
        <button
          onClick={() => setTab('upload')}
          className={`text-xs px-3 py-1.5 rounded ${
            tab === 'upload' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
          }`}
        >
          Upload video
        </button>
      </div>

      {tab === 'rtsp' ? (
        <form
          onSubmit={(e) => { e.preventDefault(); if (url.trim()) onConnectLive(url.trim()) }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="rtsp://10.0.0.5:554/main"
            className="flex-1 bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm
                       focus:border-blue-500 focus:outline-none text-white"
          />
          <button
            type="submit"
            className="text-xs px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium"
          >
            Kết nối
          </button>
        </form>
      ) : (
        <DropZone onFileSelect={onSelectFile} dark />
      )}
    </div>
  )
}
```

- [ ] **Step 2: Lift session state into `MonitorPage`**

Replace `web/src/components/monitor/MonitorPage.jsx` with:

```jsx
import { useState } from 'react'
import SourceSelector from './SourceSelector'

export default function MonitorPage() {
  const [session, setSession] = useState(null)
  // session shape:
  //   live:   { mode: 'live',   sessionId, whepUrl, mjpegUrl, rtspUrl }
  //   upload: { mode: 'upload', sessionId, videoUrl, file }

  const handleConnectLive = async (rtspUrl) => {
    const resp = await fetch('/monitor/live/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rtsp_url: rtspUrl }),
    })
    if (!resp.ok) {
      alert('Could not connect: ' + (await resp.text()))
      return
    }
    const data = await resp.json()
    setSession({
      mode: 'live',
      sessionId: data.session_id,
      whepUrl: data.whep_url,
      mjpegUrl: data.mjpeg_url,
      rtspUrl,
    })
  }

  const handleSelectFile = async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    const resp = await fetch('/monitor/upload', { method: 'POST', body: fd })
    if (!resp.ok) { alert('Upload failed'); return }
    const data = await resp.json()
    setSession({
      mode: 'upload',
      sessionId: data.session_id,
      videoUrl: data.video_url,
      file,
    })
  }

  return (
    <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5">
      {!session ? (
        <SourceSelector
          onConnectLive={handleConnectLive}
          onSelectFile={handleSelectFile}
        />
      ) : (
        <div className="text-slate-300 text-sm">
          Session ready: {session.mode} / {session.sessionId}
          {/* viewer + incidents panel come in next tasks */}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Verify in browser**

`npm run dev`, switch to Monitor tab, paste an RTSP URL and click Kết nối — expect either a successful response (if a camera is reachable) or a clear error alert. The Upload tab should accept a video and transition to the "Session ready" message.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/monitor/MonitorPage.jsx web/src/components/monitor/SourceSelector.jsx
git commit -m "feat(web): add SourceSelector and session lifting in MonitorPage"
```

### Task 7.3: WebRTC (WHEP) hook

**Files:**
- Create: `web/src/hooks/monitor/useWebRTC.js`

- [ ] **Step 1: Implement the hook**

Write to `web/src/hooks/monitor/useWebRTC.js`:

```javascript
import { useEffect, useRef, useState } from 'react'

/**
 * useWebRTC — minimal WHEP client.
 * Returns { videoRef, status, error } and attaches the inbound track
 * to videoRef.current automatically.
 */
export default function useWebRTC(whepUrl) {
  const videoRef = useRef(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!whepUrl) return undefined
    let pc = new RTCPeerConnection()
    let cancelled = false

    pc.addTransceiver('video', { direction: 'recvonly' })
    pc.addTransceiver('audio', { direction: 'recvonly' })

    pc.ontrack = (e) => {
      if (videoRef.current) videoRef.current.srcObject = e.streams[0]
    }
    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'connected') setStatus('live')
      if (pc.iceConnectionState === 'failed') {
        setStatus('error')
        setError('WebRTC ICE failed')
      }
    }

    setStatus('connecting')
    ;(async () => {
      try {
        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        const resp = await fetch(whepUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: offer.sdp,
        })
        if (!resp.ok) throw new Error('WHEP POST failed: ' + resp.status)
        const answerSdp = await resp.text()
        if (cancelled) return
        await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp })
      } catch (e) {
        if (!cancelled) {
          setStatus('error')
          setError(e.message)
        }
      }
    })()

    return () => {
      cancelled = true
      pc.close()
      pc = null
    }
  }, [whepUrl])

  return { videoRef, status, error }
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/hooks/monitor/useWebRTC.js
git commit -m "feat(web): add useWebRTC (WHEP) hook"
```

### Task 7.4: `LiveViewer` with MJPEG fallback

**Files:**
- Create: `web/src/components/monitor/LiveViewer.jsx`

- [ ] **Step 1: Implement the viewer**

Write to `web/src/components/monitor/LiveViewer.jsx`:

```jsx
import useWebRTC from '../../hooks/monitor/useWebRTC'

export default function LiveViewer({ whepUrl, mjpegUrl }) {
  const { videoRef, status, error } = useWebRTC(whepUrl)

  return (
    <div className="bg-black rounded-lg overflow-hidden relative aspect-video">
      {status !== 'error' ? (
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-contain"
        />
      ) : (
        <>
          <img src={mjpegUrl} alt="live" className="w-full h-full object-contain" />
          <div className="absolute top-2 left-2 text-xs bg-amber-900/80 text-amber-100 px-2 py-1 rounded">
            WebRTC unavailable — using MJPEG fallback ({error})
          </div>
        </>
      )}
      {status === 'connecting' && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
          Đang kết nối…
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/monitor/LiveViewer.jsx
git commit -m "feat(web): add LiveViewer with WebRTC primary + MJPEG fallback"
```

### Task 7.5: `UploadViewer` + `IntervalPicker`

**Files:**
- Create: `web/src/components/monitor/UploadViewer.jsx`
- Create: `web/src/components/monitor/IntervalPicker.jsx`

- [ ] **Step 1: Implement `IntervalPicker`**

Write to `web/src/components/monitor/IntervalPicker.jsx`:

```jsx
import { useState, useEffect } from 'react'

const MAX_INTERVAL = 30.0  // seconds

function fmt(t) {
  if (!Number.isFinite(t)) return '0:00'
  const m = Math.floor(t / 60), s = Math.floor(t % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

export default function IntervalPicker({
  duration, initialStart, initialEnd, onSeek, onAnalyze, onCancel,
}) {
  const [start, setStart] = useState(initialStart)
  const [end,   setEnd]   = useState(initialEnd)

  const delta = end - start
  const tooLong = delta > MAX_INTERVAL
  const valid = delta > 0 && !tooLong

  useEffect(() => { onSeek(start) }, [start])  // preview start
  useEffect(() => { onSeek(end) },   [end])    // preview end

  return (
    <div className="bg-slate-800/70 border border-slate-700 rounded-lg p-4 mt-3">
      <div className="text-xs text-slate-400 mb-2">Timeline</div>
      <div className="flex items-center gap-3 text-xs">
        <span>{fmt(0)}</span>
        <input
          type="range" min={0} max={duration} step={0.1}
          value={start}
          onChange={(e) => setStart(Math.min(parseFloat(e.target.value), end - 0.1))}
          className="flex-1"
        />
        <input
          type="range" min={0} max={duration} step={0.1}
          value={end}
          onChange={(e) => setEnd(Math.max(parseFloat(e.target.value), start + 0.1))}
          className="flex-1"
        />
        <span>{fmt(duration)}</span>
      </div>
      <div className="text-xs text-slate-400 mt-2">
        {fmt(start)} — {fmt(end)}  ·  Δ = {delta.toFixed(1)}s
        {tooLong && <span className="text-red-400 ml-2">(tối đa {MAX_INTERVAL}s)</span>}
      </div>
      <div className="flex gap-2 mt-3">
        <button
          onClick={() => onAnalyze(start, end)}
          disabled={!valid}
          className={`text-xs px-4 py-2 rounded font-medium ${
            valid ? 'bg-blue-600 hover:bg-blue-500 text-white'
                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
          }`}
        >
          Phân tích
        </button>
        <button
          onClick={onCancel}
          className="text-xs px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-slate-300"
        >
          Hủy
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Implement `UploadViewer`**

Write to `web/src/components/monitor/UploadViewer.jsx`:

```jsx
import { useEffect, useRef, useState } from 'react'
import IntervalPicker from './IntervalPicker'

export default function UploadViewer({ videoUrl, onMark }) {
  const videoRef = useRef(null)
  const [duration, setDuration] = useState(0)
  const [picking, setPicking]   = useState(false)
  const [initialRange, setInitialRange] = useState([0, 1])

  const handleStartPicking = () => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    const t = v.currentTime
    const start = Math.max(0, t - 10)
    const end   = Math.min(duration, t + 5)
    setInitialRange([start, end])
    setPicking(true)
  }

  const handleSeek = (t) => {
    if (videoRef.current) videoRef.current.currentTime = t
  }

  return (
    <div className="bg-black rounded-lg overflow-hidden">
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        className="w-full aspect-video object-contain"
        onLoadedMetadata={(e) => setDuration(e.target.duration)}
      />
      {!picking ? (
        <div className="bg-slate-800/70 border-t border-slate-700 p-3">
          <button
            onClick={handleStartPicking}
            className="text-xs px-4 py-2 rounded bg-red-600 hover:bg-red-500 text-white font-medium"
          >
            🚩 Mark Interval
          </button>
        </div>
      ) : (
        <IntervalPicker
          duration={duration}
          initialStart={initialRange[0]}
          initialEnd={initialRange[1]}
          onSeek={handleSeek}
          onAnalyze={(start, end) => { setPicking(false); onMark(start, end) }}
          onCancel={() => setPicking(false)}
        />
      )}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add web/src/components/monitor/UploadViewer.jsx web/src/components/monitor/IntervalPicker.jsx
git commit -m "feat(web): add UploadViewer + IntervalPicker (max 30s window)"
```

### Task 7.6: Mark hook & incident SSE consumer

**Files:**
- Create: `web/src/hooks/monitor/useMark.js`
- Create: `web/src/hooks/monitor/useIncidentStream.js`

- [ ] **Step 1: Implement `useMark`**

Write to `web/src/hooks/monitor/useMark.js`:

```javascript
export async function postMark(sessionId, body) {
  const resp = await fetch(`/monitor/${sessionId}/mark`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new Error(await resp.text())
  return (await resp.json()).incident_id
}
```

- [ ] **Step 2: Implement `useIncidentStream`**

Write to `web/src/hooks/monitor/useIncidentStream.js`:

```javascript
import { useEffect } from 'react'

/**
 * Subscribes to /monitor/{sessionId}/incidents/stream and calls onEvent
 * for each event received. Closes the EventSource on unmount.
 */
export default function useIncidentStream(sessionId, onEvent) {
  useEffect(() => {
    if (!sessionId) return undefined
    const es = new EventSource(`/monitor/${sessionId}/incidents/stream`)
    es.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data)
        if (ev.type === 'ping') return
        onEvent(ev)
      } catch (e) {
        console.error('SSE parse error', e)
      }
    }
    es.onerror = () => { /* EventSource auto-reconnects */ }
    return () => es.close()
  }, [sessionId, onEvent])
}
```

- [ ] **Step 3: Commit**

```bash
git add web/src/hooks/monitor/useMark.js web/src/hooks/monitor/useIncidentStream.js
git commit -m "feat(web): add useMark + useIncidentStream hooks"
```

### Task 7.7: `IncidentsPanel`, `IncidentCard`, `IncidentDetail`

**Files:**
- Create: `web/src/components/monitor/IncidentsPanel.jsx`
- Create: `web/src/components/monitor/IncidentCard.jsx`
- Create: `web/src/components/monitor/IncidentDetail.jsx`

- [ ] **Step 1: Implement `IncidentCard`**

Write to `web/src/components/monitor/IncidentCard.jsx`:

```jsx
import { useState } from 'react'
import IncidentDetail from './IncidentDetail'

function fmtTime(iso) {
  if (!iso) return '--:--'
  return new Date(iso).toLocaleTimeString()
}

export default function IncidentCard({ incident }) {
  const [expanded, setExpanded] = useState(false)
  const { id, status, markedAt, windowStartSec, windowEndSec, vehicles, pct, error } = incident
  const vehArr = Object.values(vehicles || {})
  const primary = vehArr[0]

  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-3 mb-2">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>{id.slice(-6)}</span>
        <span>{fmtTime(markedAt)}</span>
      </div>
      <div className="text-xs text-slate-400 mt-1">
        Δ = {(windowEndSec - windowStartSec).toFixed(1)}s · {vehArr.length} xe
      </div>

      {status === 'pending' || status === 'processing' ? (
        <div className="mt-2 text-xs text-blue-300">
          Đang phân tích… {pct ? `${pct}%` : ''}
        </div>
      ) : status === 'failed' ? (
        <div className="mt-2 text-xs text-red-400">Lỗi: {error}</div>
      ) : (
        <>
          {primary && (
            <div className="mt-2 text-sm font-bold text-emerald-400">
              {primary.plate}
            </div>
          )}
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 text-xs text-slate-400 hover:text-white"
          >
            {expanded ? 'Ẩn' : 'Chi tiết'}
          </button>
          {expanded && <IncidentDetail incident={incident} />}
        </>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Implement `IncidentDetail`**

Write to `web/src/components/monitor/IncidentDetail.jsx`:

```jsx
export default function IncidentDetail({ incident }) {
  const vehArr = Object.values(incident.vehicles || {})
  return (
    <div className="mt-2 space-y-2">
      {vehArr.map((v) => (
        <div key={v.track_id ?? v.id} className="bg-slate-900/50 rounded p-2">
          {v.plate_b64 && (
            <img
              src={`data:image/jpeg;base64,${v.plate_b64}`}
              alt={v.plate}
              className="w-full max-h-16 object-contain bg-black rounded mb-1"
            />
          )}
          <div className="text-sm font-mono text-emerald-300">{v.plate}</div>
          <div className="text-[10px] text-slate-500">{v.cls} · {v.ocr_frames} frames</div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Implement `IncidentsPanel`**

Write to `web/src/components/monitor/IncidentsPanel.jsx`:

```jsx
import IncidentCard from './IncidentCard'

export default function IncidentsPanel({ incidents }) {
  const list = Object.values(incidents).sort(
    (a, b) => new Date(b.markedAt) - new Date(a.markedAt),
  )
  return (
    <div className="w-80 flex-shrink-0 bg-slate-900/30 rounded-lg p-3 overflow-y-auto"
         style={{ height: 'calc(100vh - 200px)' }}>
      <div className="text-xs text-slate-400 mb-3">
        Sự cố ({list.length})
      </div>
      {list.length === 0 ? (
        <div className="text-xs text-slate-500 text-center py-8">
          Chưa có sự cố nào.
        </div>
      ) : (
        list.map((inc) => <IncidentCard key={inc.id} incident={inc} />)
      )}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add web/src/components/monitor/IncidentsPanel.jsx web/src/components/monitor/IncidentCard.jsx web/src/components/monitor/IncidentDetail.jsx
git commit -m "feat(web): add IncidentsPanel + IncidentCard + IncidentDetail"
```

### Task 7.8: Wire everything together in MonitorPage

**Files:**
- Modify: `web/src/components/monitor/MonitorPage.jsx`

- [ ] **Step 1: Wire viewer + mark + SSE + panel**

Replace `web/src/components/monitor/MonitorPage.jsx` with:

```jsx
import { useCallback, useEffect, useState } from 'react'

import SourceSelector  from './SourceSelector'
import LiveViewer      from './LiveViewer'
import UploadViewer    from './UploadViewer'
import IncidentsPanel  from './IncidentsPanel'
import useIncidentStream from '../../hooks/monitor/useIncidentStream'
import { postMark }    from '../../hooks/monitor/useMark'

export default function MonitorPage() {
  const [session,  setSession]  = useState(null)
  const [incidents, setIncidents] = useState({})

  // SSE handler
  const handleEvent = useCallback((ev) => {
    const id = ev.incident_id
    if (!id) return

    setIncidents((prev) => {
      const cur = prev[id] || {
        id, status: 'pending', vehicles: {}, markedAt: new Date().toISOString(),
        windowStartSec: 0, windowEndSec: 0,
      }
      switch (ev.type) {
        case 'incident_started':
          return { ...prev, [id]: {
            ...cur,
            status: 'processing',
            sourceType: ev.source_type,
            windowStartSec: ev.window_start_sec,
            windowEndSec:   ev.window_end_sec,
            framesCount:    ev.frames_count,
          }}
        case 'incident_progress':
          return { ...prev, [id]: { ...cur, pct: ev.pct } }
        case 'incident_vehicle':
          return { ...prev, [id]: { ...cur,
            vehicles: { ...cur.vehicles, [ev.id]: ev }
          }}
        case 'incident_rejected_vehicle':
          return { ...prev, [id]: { ...cur,
            rejected: { ...(cur.rejected || {}), [ev.id]: ev }
          }}
        case 'incident_complete':
          return { ...prev, [id]: { ...cur, status: 'completed',
            durationMs: ev.duration_ms, totalVehicles: ev.total_vehicles }}
        case 'incident_error':
          return { ...prev, [id]: { ...cur, status: 'failed', error: ev.message } }
        default:
          return prev
      }
    })
  }, [])

  useIncidentStream(session?.sessionId, handleEvent)

  // Tear down live session on unmount or new-session
  useEffect(() => {
    return () => {
      if (session?.mode === 'live') {
        fetch(`/monitor/live/${session.sessionId}`, { method: 'DELETE' }).catch(() => {})
      }
    }
  }, [session?.sessionId])

  // ── Actions ────────────────────────────────────────────────────────────
  const handleConnectLive = async (rtspUrl) => {
    const resp = await fetch('/monitor/live/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rtsp_url: rtspUrl }),
    })
    if (!resp.ok) { alert('Could not connect: ' + (await resp.text())); return }
    const data = await resp.json()
    setSession({ mode: 'live', sessionId: data.session_id, whepUrl: data.whep_url, mjpegUrl: data.mjpeg_url, rtspUrl })
    setIncidents({})
  }

  const handleSelectFile = async (file) => {
    const fd = new FormData(); fd.append('file', file)
    const resp = await fetch('/monitor/upload', { method: 'POST', body: fd })
    if (!resp.ok) { alert('Upload failed'); return }
    const data = await resp.json()
    setSession({ mode: 'upload', sessionId: data.session_id, videoUrl: data.video_url, file })
    setIncidents({})
  }

  const handleMarkLive = async () => {
    try {
      const id = await postMark(session.sessionId, { mode: 'live' })
      setIncidents((p) => ({ ...p, [id]: {
        id, status: 'pending', markedAt: new Date().toISOString(),
        windowStartSec: 0, windowEndSec: 0, vehicles: {},
      }}))
    } catch (e) { alert('Mark failed: ' + e.message) }
  }

  const handleMarkUpload = async (tStart, tEnd) => {
    try {
      const id = await postMark(session.sessionId, { mode: 'upload', t_start: tStart, t_end: tEnd })
      setIncidents((p) => ({ ...p, [id]: {
        id, status: 'pending', markedAt: new Date().toISOString(),
        windowStartSec: tStart, windowEndSec: tEnd, vehicles: {},
      }}))
    } catch (e) { alert('Mark failed: ' + e.message) }
  }

  // ── Layout ─────────────────────────────────────────────────────────────
  if (!session) {
    return (
      <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5">
        <SourceSelector onConnectLive={handleConnectLive} onSelectFile={handleSelectFile} />
      </div>
    )
  }

  return (
    <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5 flex gap-4">
      <div className="flex-1 min-w-0">
        {session.mode === 'live' ? (
          <>
            <LiveViewer whepUrl={session.whepUrl} mjpegUrl={session.mjpegUrl} />
            <div className="mt-3">
              <button
                onClick={handleMarkLive}
                className="text-sm px-5 py-3 rounded-lg bg-red-600 hover:bg-red-500
                           text-white font-bold w-full"
              >
                🚩 Mark Now (10s)
              </button>
            </div>
          </>
        ) : (
          <UploadViewer videoUrl={session.videoUrl} onMark={handleMarkUpload} />
        )}
      </div>
      <IncidentsPanel incidents={incidents} />
    </div>
  )
}
```

- [ ] **Step 2: Manual E2E sanity check**

```bash
docker compose up -d mediamtx mongo
python -m uvicorn api.main:app --reload &
cd web && npm run dev
```

Open the browser, go to the Monitor tab:
- Upload mode: drop a sample video → click 🚩 Mark Interval → drag handles to a 5s range → click Phân tích → verify a card appears, transitions pending → completed → shows any plates.
- Live mode (if you have an RTSP source): paste URL → connect → click 🚩 Mark Now → verify the 10s rolling buffer is analyzed and a card appears.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/monitor/MonitorPage.jsx
git commit -m "feat(web): wire mark → SSE → IncidentsPanel in MonitorPage"
```

---

## Phase 8 — Documentation & cleanup

### Task 8.1: Update `.env.example` and CLAUDE.md

**Files:**
- Modify: `api/.env.example` (create if missing)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the new env vars**

Append to `api/.env.example` (create the file if it doesn't exist):

```
# MediaMTX integration for Incident Monitor
MEDIAMTX_API_URL=http://mediamtx:9997
MEDIAMTX_INTERNAL_RTSP_BASE=rtsp://mediamtx:8554
MEDIAMTX_PUBLIC_WEBRTC_BASE=http://localhost:8889
MEDIAMTX_PUBLIC_MJPEG_BASE=
```

- [ ] **Step 2: Document the new architecture in `CLAUDE.md`**

In `ALPR_Vietnamese/CLAUDE.md`, under the "Architecture" section, add a new subsection:

```markdown
### Incident Monitor (mark-driven analysis)

A second mode of operation (the "Giám sát sự cố" tab). Operators observe a
live RTSP camera (via MediaMTX→WebRTC) or play back an uploaded video, then
mark an incident: a short window is pulled out and analyzed in isolation.

Key modules:
- `api/core/pipeline_core.py` — shared inference loop used by both the legacy
  upload flow and the incident analyzer.
- `api/core/frame_source.py` — `FrameSource` Protocol with `FileFrameSource`
  and `LiveBufferFrameSource` implementations.
- `api/core/live_session.py` — per-session RTSP decoder thread + rolling 10s
  buffer + MJPEG-fallback queue.
- `api/core/mediamtx_client.py` — thin HTTP client for adding/removing
  MediaMTX paths dynamically.
- `api/core/incident_analyzer.py` — orchestrates a single mark → analysis →
  persistence job.
- `api/routes_monitor.py` — HTTP + SSE endpoints under `/monitor/*` and
  `/incidents/*`.

The MediaMTX container (added to docker-compose) is the single RTSP consumer
of the camera; Python reads MediaMTX's re-published RTSP, and the browser
gets WebRTC via MediaMTX's WHEP endpoint.

Spec: `docs/superpowers/specs/2026-05-20-incident-monitor-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add api/.env.example CLAUDE.md
git commit -m "docs: document Incident Monitor architecture + env vars"
```

### Task 8.2: Run the full test suite + coverage

**Files:** (no edits — verification only)

- [ ] **Step 1: Run pytest with coverage**

```bash
pytest tests/ -v --cov=api/core --cov=api/routes_monitor --cov-report=term-missing
```

Expected: all tests PASS. Coverage on the new modules (`pipeline_core`, `frame_source`, `live_session`, `incident_analyzer`, `mediamtx_client`, `routes_monitor`) should be ≥80%.

- [ ] **Step 2: If coverage is below 80%, add targeted tests**

Use the missing-line report to write additional unit tests for uncovered branches. Typical gaps after the above tasks:
- error-path branches in `LiveSession._decoder_loop` (RTSP open failure)
- `_persist_incident` running when `is_db_configured() == False`
- 409 buffer-warmup branch in `/mark` (test by submitting an empty live session)

Add tests until coverage clears 80%, then commit:

```bash
git add tests/
git commit -m "test: increase coverage on incident-monitor modules to ≥80%"
```

### Task 8.3: Manual E2E checklist

**Files:**
- Create: `docs/superpowers/plans/2026-05-20-incident-monitor-e2e-checklist.md`

- [ ] **Step 1: Write the manual E2E checklist**

Write to `docs/superpowers/plans/2026-05-20-incident-monitor-e2e-checklist.md`:

```markdown
# Incident Monitor — Manual E2E Checklist

Run these after `docker compose up -d` and `npm run dev`.

## Upload mode

- [ ] Switch to "Giám sát sự cố" tab.
- [ ] Upload `tests/fixtures/short_clip.mp4`.
- [ ] Video player appears, controls work, no console errors.
- [ ] Click "🚩 Mark Interval" → video pauses, IntervalPicker overlay appears.
- [ ] Drag the two handles to a 5s sub-range.
- [ ] Confirm Δ readout updates in real time.
- [ ] Click "Phân tích" → an IncidentCard appears in pending state.
- [ ] Card transitions to "completed" after analysis.
- [ ] Card shows zero vehicles for the fixture (it has no plates).
- [ ] Mark a SECOND interval — both cards remain visible.
- [ ] Refresh page → cards disappear (in-memory only); incident still in Mongo.

## Live mode (requires a reachable RTSP source)

- [ ] Switch to "Giám sát sự cố" tab → choose "RTSP camera".
- [ ] Paste an RTSP URL (any public test stream or local camera).
- [ ] Click "Kết nối" → LiveViewer renders the WebRTC stream within ~2s.
- [ ] If WebRTC fails, MJPEG fallback banner appears and `<img>` shows frames.
- [ ] Click "🚩 Mark Now" → card appears immediately as pending.
- [ ] Card transitions to completed once analysis finishes.
- [ ] Click "🚩 Mark Now" within 1s of connecting → expect 409 "Buffer warming up".
- [ ] Refresh page → LiveSession is torn down (MediaMTX path removed via `curl http://localhost:9997/v3/paths/list`).

## Error paths

- [ ] Submit interval > 30s → "Interval exceeds 30s max" alert.
- [ ] Submit invalid RTSP URL (http://...) → "URL must be rtsp:// or rtsps://".
- [ ] Stop MediaMTX container mid-live-session → eventually LiveSession reports failure (check server logs).

## Regression on legacy flow

- [ ] Switch back to "Xử lý video" → upload a video → full processing works exactly as before.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-05-20-incident-monitor-e2e-checklist.md
git commit -m "docs: add manual E2E checklist for incident-monitor"
```

---

## Self-Review (done by plan author)

### Spec coverage

- ✅ Section 1 (problem/goal) → Phase 1–8 cover the full feature.
- ✅ Section 2 in/out of scope → all "in scope" items have tasks; out-of-scope items have no tasks.
- ✅ Section 3 architectural decisions → enforced by module structure in Phases 1–6.
- ✅ Section 4 topology → Task 3.1 (docker-compose + mediamtx.yml).
- ✅ Section 5 module layout → Tasks 1.1 / 2.2 / 2.3 / 3.2 / 4.1 / 5.3 / 6.1.
- ✅ Section 6 abstractions → Tasks 1.1 (FrameSource), 2.3 (process_frames), 4.1 (LiveSession), 3.2 (mediamtx_client), 5.3 (run_incident).
- ✅ Section 7 routes → Tasks 6.1 / 6.2 / 6.3 / 6.4 / 6.5.
- ✅ Section 8 SSE contracts → Task 5.3 implements `_EVENT_MAP`; Task 6.5 streams them.
- ✅ Section 9 MongoDB schema → Tasks 5.1 + 5.2.
- ✅ Section 10 frontend layout → Tasks 7.1 – 7.8.
- ✅ Section 11 visual layouts → realized via Tasks 7.4 / 7.5 / 7.7.
- ✅ Section 12 config → Tasks 3.1 + 8.1.
- ✅ Section 13 failure handling → Task 4.1 (retry), Task 6.4 (409 buffer warmup, 400 window too long), Task 7.4 (MJPEG fallback).
- ✅ Section 14 testing → Tasks 0.1, 1.1, 3.2, 4.1, 5.1, 5.2, 5.3, 6.* + 8.2 coverage gate.
- ✅ Section 15 risks → parity test (Task 2.1), single-worker executor (Task 6.4), retries (Task 4.1).
- ✅ Section 16 security → URL scheme validation (Task 6.2), credential masking in `mediamtx_client.add_path`.

### Placeholder scan

- No "TBD", "TODO", or "fill in later" in any task. Every step has either code, exact commands, or both.
- All function names referenced across tasks are consistent: `process_frames`, `run_incident`, `_persist_incident`, `LiveSession.snapshot_window`, `FileFrameSource`, `LiveBufferFrameSource`, `mediamtx_client.add_path / remove_path`.

### Type consistency

- `Incident` and `IncidentVehicle` field names align with the SSE `incident_vehicle` event keys (`track_id`/`id` translation handled in `_persist_incident`).
- `MarkBody` accepts both modes; the route validates `t_start`/`t_end` only for upload.
- `app.state.models` is set in Task 6.6 BEFORE it is referenced in Task 6.4's `_dispatch_incident` — order matters: Task 6.6 lands before any real run, but the route file is created in 6.1. The `_dispatch_incident` is exercised only in Task 6.4's tests where `_dispatch_incident` is monkeypatched, so type ordering is safe.

No issues found that aren't addressed in line.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-20-incident-monitor.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
