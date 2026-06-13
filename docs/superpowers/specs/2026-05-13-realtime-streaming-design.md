# Real-Time Multi-Source Streaming — Design Spec

**Date:** 2026-05-13
**Status:** Approved
**Scope:** Add live RTSP stream processing alongside the existing offline upload flow.

---

## Problem

The current ALPR pipeline (`api/core/pipeline.py`) triggers OCR only on two events:
track loss (vehicle disappears) or video end. Both require waiting — neither works for a
live stream that runs indefinitely. A new strategy is needed for real-time sources.

---

## Requirements

- Support multiple concurrent live RTSP streams (N cameras simultaneously).
- Support both live stream sessions and offline upload sessions running at the same time.
- Live sessions are persistent — no defined end; operator stops manually.
- OCR trigger for live streams: crop-count threshold (fire as soon as MIN_FRAMES_FOR_OCR
  high-quality crops are buffered per track). Quality over speed — no artificial rush.
- Best-effort quality: after OCR fires and the track is marked done, do not re-run even
  if the vehicle stays in frame longer.

---

## Architecture

### New files

```
api/
├── core/
│   ├── pipeline.py          # UNCHANGED — offline upload flow
│   ├── live_pipeline.py     # NEW — run_live_stream()
│   └── stream_manager.py    # NEW — LiveSession dataclass, StreamManager class
└── main.py                  # EXTENDED — /streams endpoints + WebSocket handler
```

### Shared modules (untouched)

`tracker.py`, `association.py`, `models.py`, `gates.py`, `quality_scorer.py`,
`video_processor.py`, and all OCR helpers in `pipeline.py` are imported by both paths.
`_run_multiframe_ocr()`, `_session_create()`, `_session_update()`, `_record_save()`
from `pipeline.py` are re-exported and used directly by `live_pipeline.py`.

### Two parallel session tracks

```
Upload flow (existing)
  POST /upload         → run_job() in thread-pool
  _jobs dict           → job_id → asyncio.Queue (SSE events)
  GET /stream/{id}/events     SSE
  GET /stream/{id}/mjpeg      MJPEG

Live stream flow (new)
  POST /streams        → StreamManager.start()
  StreamManager        → stream_id → LiveSession
  WS  /streams/{id}/ws        WebSocket (bidirectional)
  GET /streams/{id}/mjpeg     MJPEG (same pattern as upload)
  GET /streams                list all sessions
  DELETE /streams/{id}        stop session
```

Both tracks produce the same SSE/WS event shapes for `vehicle` and `rejected_vehicle`,
so frontend rendering components are shared.

---

## Live Pipeline (`live_pipeline.py`)

### Function signature

```python
def run_live_stream(
    url: str,
    stream_id: str,
    event_queue: asyncio.Queue,
    mjpeg_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    models: ModelBundle,
    stop_event: threading.Event,
) -> None:
```

### OCR trigger logic

```
Per-frame loop:
  ├── vehicle tracking (BotSORT, every frame)
  ├── skip plate detection on non-stride frames
  ├── (stride frames) plate detection + TrajectoryAssociator
  ├── buffer_crop(tid, crop, quality, frame_idx)
  │
  ├── TRIGGER (new — live only):
  │     if ready_for_multiframe_ocr(tid) and should_ocr(tid):
  │         _run_multiframe_ocr(...)   ← fires while vehicle is still in frame
  │         # sets _done[tid] = True → no re-trigger
  │
  └── TRACK LOSS (fallback — same as offline):
        if tid disappeared and not _done[tid] and ready_for_multiframe_ocr(tid):
            _run_multiframe_ocr(...)   ← catches short-stay vehicles
```

`MIN_FRAMES_FOR_OCR = 3` (from `config.py`) is the threshold.
`MAX_RECONNECT_ATTEMPTS = 10` is added to `config.py` alongside other pipeline constants.

### RTSP reconnection

```
connect()
  └── success → frame loop
        └── cap.read() returns False / timeout
              └── emit {"type": "reconnecting", "attempt": N}
              └── sleep min(2^N, 30) seconds  ← exponential backoff, cap 30s
              └── check stop_event between retries
              └── retry up to MAX_RECONNECT_ATTEMPTS (default: 10)
                    └── exhausted → emit {"type": "error"} → exit thread
```

### Stop signal

`stop_event.is_set()` is checked:
- At the top of every frame loop iteration
- During reconnect sleep (wake immediately if set)

On clean stop: emit `{"type": "stopped"}`, release `cap`, exit.

### Progress events

No `total` / `pct` (stream is infinite). Emits `frame_count` every 10 frames for
monitoring:
```json
{"type": "heartbeat", "frame_count": 340, "active_tracks": 2}
```

### Differences from `pipeline.py` summary

| Aspect | Offline (`pipeline.py`) | Live (`live_pipeline.py`) |
|---|---|---|
| Source | File path | RTSP URL |
| OCR trigger | Track loss + video end | Crop count threshold |
| Video end flush | Yes | No |
| Stop mechanism | `cap.read()` returns False | `stop_event` |
| Progress event | `progress` with pct | `heartbeat` with frame_count |
| Reconnection | N/A | Exponential backoff |
| Terminal event | `complete` | `stopped` or `error` |

---

## StreamManager (`stream_manager.py`)

### LiveSession dataclass

```python
@dataclass
class LiveSession:
    stream_id: str
    url: str
    name: str
    status: Literal["connecting", "running", "reconnecting", "stopped", "error"]
    event_queue: asyncio.Queue
    mjpeg_queue: asyncio.Queue       # maxsize=60
    stop_event: threading.Event
    thread: threading.Thread
    started_at: datetime
    vehicle_count: int = 0
```

### StreamManager methods

```python
class StreamManager:
    def start(self, stream_id: str, url: str, name: str) -> LiveSession
    def stop(self, stream_id: str) -> None          # sets stop_event, joins thread
    def list(self) -> list[dict]                     # status snapshot
    def get(self, stream_id: str) -> LiveSession | None
```

`start()` creates the session, spawns a daemon thread running `run_live_stream()`,
stores it in an internal dict, and returns immediately. The WebSocket client receives
`{"type": "connected"}` once `cap.isOpened()` succeeds inside the thread.

### Concurrency model

Each live session: one thread. Events pushed via
`loop.call_soon_threadsafe(queue.put_nowait, event)` — identical to the existing
offline pattern. No new threading primitives.

The `_models: ModelBundle` singleton loaded at startup is passed to `StreamManager`
at construction and shared across all sessions (models are stateless for inference).

---

## API Endpoints

### New endpoints

```
POST   /streams
       Request:  {"url": "rtsp://...", "name": "Cam 1"}
       Response: {"stream_id": "a1b2c3d4"}
       Effect:   registers stream, starts background thread

DELETE /streams/{stream_id}
       Response: 204 No Content
       Effect:   sets stop_event, waits for thread join (timeout 5s)

GET    /streams
       Response: [{stream_id, name, url, status, started_at, vehicle_count}]

WS     /streams/{stream_id}/ws
       Server → Client messages:
         {"type": "connected"}
         {"type": "heartbeat", "frame_count": N, "active_tracks": N}
         {"type": "vehicle", "id": N, "cls": "...", "plate": "...", ...}
         {"type": "rejected_vehicle", ...}
         {"type": "reconnecting", "attempt": N}
         {"type": "stopped"}
         {"type": "error", "message": "..."}
         {"type": "ping"}
       Client → Server messages:
         {"type": "stop"}   ← triggers graceful shutdown

GET    /streams/{stream_id}/mjpeg
       multipart/x-mixed-replace (same implementation as /stream/{job_id}/mjpeg)
```

### WebSocket reconnect from client side

If the WebSocket connection drops (network blip), the client reconnects to the same
`/streams/{stream_id}/ws`. The pipeline thread continues running. On reconnect, the
server resumes forwarding events from the queue.

---

## Frontend Changes

### New components

| Component | Purpose |
|---|---|
| `LiveStreamTab.jsx` | Tab container: stream list, add-stream form, session cards |
| `useStreamSession.js` | WebSocket hook for live sessions |
| `StreamCard.jsx` | Per-session status card (status badge, vehicle count, stop button) |

### Reused unchanged

`LiveFrame.jsx` (MJPEG viewer), vehicle sidebar, plate card components, `useStream.js`
(SSE hook for upload sessions).

### UI structure

```
[Upload Video]  [Live Streams]          ← top-level tab switcher

Live Streams tab:
  ┌── Add Stream form ──────────────┐
  │ URL:  [rtsp://...             ] │
  │ Name: [Cam 1      ]    [Start] │
  └─────────────────────────────────┘

  Session cards (one per active stream):
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Cam 1   │  │ Cam 2   │  │ Cam 3   │
  │ running │  │ error   │  │ stopped │
  │ 12 cars │  │ [retry] │  │ [start] │
  └─────────┘  └─────────┘  └─────────┘

  Selected stream view:
  ┌─────────────────────┬──────────────┐
  │  MJPEG preview      │ Vehicle list │
  │  (LiveFrame.jsx)    │ (sidebar)    │
  └─────────────────────┴──────────────┘
```

### `useStreamSession.js` behaviour

- Opens `WebSocket` to `/streams/{stream_id}/ws`
- Handles `vehicle`, `rejected_vehicle` → appends to local plate list
- Handles `reconnecting` → shows reconnecting badge on session card
- Handles `error` / `stopped` → marks session card accordingly
- Sends `{"type": "stop"}` when operator clicks Stop
- Sends ping-pong keepalive every 30s

---

## Database

Live sessions use the same `RecognitionSession` MongoDB document with two additional
fields:
- `source_type: "live_stream"` (offline sessions: `"upload"`)
- `stream_url: str`

Status progression: `"processing"` → `"stopped"` (clean) or `"failed"` (error).
No `"completed"` status for live sessions. `processed_frames` is written once on
session stop/error (not on every heartbeat — avoids N continuous DB writes).

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| RTSP URL unreachable on start | Exponential backoff retry; `reconnecting` events; `error` after max retries |
| Stream drops mid-session | Same backoff loop; UI shows "Reconnecting…" on session card |
| Client WS disconnects | Pipeline thread keeps running; client reconnects to same stream_id |
| OCR inference error | Log exception, skip that track's OCR attempt, continue pipeline |
| MongoDB unavailable | Log warning, skip persistence, continue recognition |
| `stop_event` set during sleep | Wake immediately via `stop_event.wait(timeout=delay)` |
| `DELETE /streams/{id}` on unknown id | 404 Not Found |

---

## Testing

### Unit tests
- `test_live_pipeline.py`: OCR trigger fires after MIN_FRAMES_FOR_OCR crops; does not re-fire after `_done`; track loss fallback triggers when vehicle leaves before threshold
- `test_stream_manager.py`: `start()` spawns thread; `stop()` sets event and joins; `list()` returns correct status

### Integration tests
- Mock RTSP source (local video file opened as RTSP via `cv2.VideoCapture`)
- Verify WebSocket messages received in order: `connected` → `vehicle` events → `stopped`
- Verify MongoDB records created per recognized plate

### Not covered by automated tests (manual)
- Real RTSP camera reconnection behaviour
- Multi-stream concurrency at scale
- Frontend session card status transitions
