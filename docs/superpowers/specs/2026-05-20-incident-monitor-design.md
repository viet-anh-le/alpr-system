# Incident Monitor — Design Spec

**Date:** 2026-05-20
**Status:** Draft, awaiting user review
**Owner:** vietanhle3012bn@gmail.com

## 1. Problem & Goal

Operators continuously observe a video stream (live IP camera or a previously-uploaded long video). When they notice an incident — a collision, a traffic violation, a suspicious vehicle — they need to **mark the moment** and have the system identify the involved license plates **within seconds**, not minutes.

The existing system only supports "upload a whole video → wait for full processing". This spec adds an **Incident Monitor** mode that decouples observation from analysis: observation is cheap (raw playback only), analysis is targeted (a single short window per mark).

### Success criteria

- Operator can connect to an RTSP camera or open a long uploaded video and watch it with low (≤2s) end-to-end latency.
- Marking an incident in **live mode** is a single click; results begin streaming back within ~5s for a 10s window.
- Marking an incident in **upload mode** lets the operator drag-select an interval up to 30s; results stream back within ~10s.
- Multiple incidents can be marked in one session; all are persisted to MongoDB and reviewable later.
- The existing `Xử lý video` (whole-video upload) flow remains unchanged and passes all existing tests.

## 2. Scope

### In scope

- New page `Incident Monitor` (Vietnamese: `Giám sát sự cố`) with two-tab header navigation.
- RTSP camera ingest via MediaMTX (single source of truth for the camera).
- Browser video via WebRTC (WHEP), with MJPEG fallback.
- Rolling 10s raw-frame buffer per live session (native resolution).
- Uploaded-video mode with HTML5 `<video>` playback + interval-picker overlay (max 30s window).
- Shared inference pipeline (`pipeline_core.process_frames`) used by live, upload, and the legacy `run_job`.
- New `incidents` MongoDB collection; images in Supabase Storage under `incidents/<id>/`.
- WebRTC failure detection with MJPEG fallback.

### Out of scope (v1)

- Long-term recording (>10s rolling) — would need MediaMTX `record: yes` and HLS segments.
- Browser webcam ingest (WebRTC publish).
- Authorization on incident records (any session can read any incident).
- Streaming results between concurrent operators.
- Live detection bounding boxes during observation.

## 3. Architectural Decisions (with chosen options)

| Decision | Choice | Why |
|---|---|---|
| Modes | Both live RTSP + uploaded video | Operators have both CCTV and archived footage. |
| Mode coupling | Separate page; existing flow untouched | Lowest regression risk; one mental model per page. |
| Live mark UX | One-click "Mark Now" → last 10s | Operators react in seconds; no time to fiddle with handles. |
| Upload mark UX | Pause + two-handle interval picker (max 30s) | Operators want precision when scrubbing recorded footage. |
| Default upload window | `[currentTime - 10s, currentTime + 5s]`, clamped | Pre-fills a useful window; user adjusts as needed. |
| Camera ingest | MediaMTX (RTSP from camera) | Battle-tested; one TCP session to camera; handles reconnect, jitter, NAT. |
| Browser video | WebRTC via WHEP (primary), MJPEG fallback | <500ms latency primary; HTTP MJPEG works through firewalls. |
| MediaMTX paths | Dynamically added/removed per session via MediaMTX HTTP API | Avoids static config; clean teardown. |
| Observation pipeline | Raw decode only — no inference | Minimize steady-state CPU/GPU. All processing is mark-driven. |
| Analysis pipeline | Extract pure `pipeline_core.process_frames` shared by all entry points | One source of truth for detect→track→OCR→vote logic. |
| Rolling buffer | `collections.deque(maxlen=fps*10)` of native-resolution frames | Trades RAM (~1.8GB per session @ 1080p30) for analysis quality. |
| Concurrency | Single-worker thread pool for incident analyzer; ≤2 concurrent live sessions | Avoids GPU contention; cap RAM at ~3.6GB for buffers. |
| Persistence | New `incidents` collection | Clean separation from `recognition_records`. |
| Frontend state | Plain `useState` keyed by incident_id (no Redux/Zustand) | Matches existing codebase style. |

## 4. Topology

```
                    ┌────────────────────────────────────────┐
                    │   IP Camera                            │
                    └──────────────┬─────────────────────────┘
                                   │ RTSP
                                   ▼
                    ┌────────────────────────────────────────┐
                    │   MediaMTX (Docker container)          │
                    │   path: live_<session_id>              │
                    │                                        │
                    │   Exposes:  :8554 RTSP                 │
                    │             :8889 WebRTC (WHEP)        │
                    │             :9997 HTTP API             │
                    └──────────┬─────────────────┬───────────┘
                               │                 │
                       RTSP    │                 │  WebRTC (WHEP)
                       (republish)               │  via :8889
                               ▼                 ▼
                    ┌────────────────────┐ ┌──────────────────┐
                    │  FastAPI / Python  │ │     Browser      │
                    │  cv2.VideoCapture  │ │  RTCPeer +       │
                    │  → rolling buffer  │ │  <video>         │
                    └────────────────────┘ └──────────────────┘
```

For upload mode, MediaMTX is not involved: the browser plays the file with HTML5 `<video>` (local object URL), and FastAPI reads the same file via `cv2.VideoCapture` when a mark arrives.

## 5. Backend Module Layout

```
api/core/
├── pipeline.py              ← existing; refactored to delegate to pipeline_core
├── pipeline_core.py         ← NEW: pure inference loop (detect → track → buffer → OCR → emit)
├── frame_source.py          ← NEW: FrameSource Protocol + FileFrameSource + LiveBufferFrameSource
├── live_session.py          ← NEW: per-session RTSP decoder + rolling buffer + MJPEG queue
├── mediamtx_client.py       ← NEW: thin HTTP client (add_path, remove_path)
├── incident_analyzer.py     ← NEW: orchestrates a single mark → analysis job
└── ... (rest unchanged)

api/
├── main.py                  ← adds new routes (Section 7)
└── database/
    ├── models.py            ← adds Incident, IncidentVehicle Pydantic models
    └── mongodb.py           ← adds upsert_incident, get_incident, list_incidents

configs/
└── mediamtx.yml             ← NEW: MediaMTX config (api: yes, webrtc: yes, no static paths)

docker-compose.yml           ← adds mediamtx service
```

## 6. Key Abstractions

### `FrameSource` Protocol (`api/core/frame_source.py`)

```python
class FrameSource(Protocol):
    fps: float
    frame_size: tuple[int, int]     # (w, h)
    total_frames: int | None        # None for live snapshot

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        """Yield (frame_idx, frame_bgr, timestamp_sec) until exhausted."""
```

Implementations:

- `FileFrameSource(path: str, t_start: float = 0.0, t_end: float | None = None)` — opens `cv2.VideoCapture`, seeks to `t_start` via `CAP_PROP_POS_MSEC`, yields frames until video time ≥ `t_end`. `t_end=None` means "until end of file".
- `LiveBufferFrameSource(frames: list[tuple[int, np.ndarray, float]], fps: float, frame_size: tuple[int, int])` — wraps a list snapshotted from a `LiveSession`'s rolling buffer. Frames are pre-decoded; iteration is a no-op pass-through.

### `pipeline_core.process_frames` (extracted from existing `run_job`)

```python
def process_frames(
    source: FrameSource,
    emit: Callable[[dict], None],
    models: ModelBundle,
    *,
    session_id: str = "",
    loop: asyncio.AbstractEventLoop | None = None,
    mjpeg_queue: asyncio.Queue | None = None,
) -> dict:
    """
    Runs the full detect → track → buffer → OCR → vote → emit pipeline.
    Returns a summary dict: {total_vehicles, processed_frames, duration_ms}.
    """
```

The existing `run_job` is rewritten to:
```python
def run_job(video_path, job_id, queue, loop, models, jobs, filename, mjpeg_queue):
    source = FileFrameSource(video_path, t_start=0.0, t_end=None)
    summary = process_frames(source, emit=..., models=models, session_id=job_id, ...)
    emit({"type": "complete", **summary})
```

This is **the regression-risk surface**, guarded by `test_pipeline_core_parity`.

### `LiveSession` (`api/core/live_session.py`)

```python
class LiveSession:
    session_id: str
    mediamtx_path: str
    fps: float
    frame_size: tuple[int, int]
    frame_buffer: collections.deque   # maxlen = int(fps * 10)
    mjpeg_queue: asyncio.Queue        # maxsize=60 — for MJPEG fallback only
    started_at: float                 # monotonic

    def start(self, rtsp_url: str) -> None: ...
    def stop(self) -> None: ...
    def snapshot_window(self, seconds: float = 10.0) -> list[tuple[int, np.ndarray, float]]: ...
```

One decoder thread per session:
1. Calls `mediamtx_client.add_path(path, source=rtsp_url)`.
2. Opens `cv2.VideoCapture(f"rtsp://mediamtx:8554/{path}")`.
3. Reads frames in a tight loop:
   - Append `(idx, frame, ts)` to `frame_buffer` (older frames auto-evicted by `maxlen`).
   - JPEG-encode and `put_nowait` into `mjpeg_queue` (drop if full — MJPEG fallback is best-effort).
4. On `cap.read() == False`: retry 3× with 1s backoff, then emit error and tear down.
5. On `stop()`: calls `mediamtx_client.remove_path(path)`.

### `mediamtx_client` (`api/core/mediamtx_client.py`)

```python
def add_path(name: str, source: str) -> None:
    """POST /v3/config/paths/add/{name} with {source: <rtsp_url>}.
       Raises MediaMTXError on non-2xx."""

def remove_path(name: str) -> None:
    """DELETE /v3/config/paths/delete/{name}. Idempotent — 404 is ignored."""
```

Uses `httpx.Client` against `MEDIAMTX_API_URL` (from env). Timeout 5s.

### `incident_analyzer.IncidentJob` (`api/core/incident_analyzer.py`)

```python
def run_incident(
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
    """
    Runs process_frames on the given source, translates pipeline events into
    incident_* events, persists the final Incident document to MongoDB.
    """
```

Event translation: subscribes to the pipeline's emit callback and rewrites event types:
- `vehicle` → `incident_vehicle`
- `rejected_vehicle` → `incident_rejected_vehicle`
- `progress` → `incident_progress`
- `complete` → `incident_complete`
- `error` → `incident_error`

All carrying `incident_id` so the SSE consumer can route to the right card.

## 7. HTTP Routes

```
POST   /monitor/live/connect          {rtsp_url}                    → {session_id, whep_url, mjpeg_url}
DELETE /monitor/live/{session_id}                                   ← disconnect; removes MediaMTX path
GET    /monitor/live/{session_id}/mjpeg                             ← MJPEG fallback stream

POST   /monitor/upload                (multipart file)              → {session_id, video_url}
GET    /monitor/upload/{session_id}/video                           ← streams the file for HTML5 <video>

POST   /monitor/{session_id}/mark
       Body (live):   {mode: "live"}
       Body (upload): {mode: "upload", t_start: float, t_end: float}
                                                                     → {incident_id}
GET    /monitor/{session_id}/incidents/stream                       ← SSE: incident_* events

GET    /incidents/{incident_id}                                     ← fetch persisted incident
GET    /incidents?source=&from=&to=&limit=                          ← list (for History view)
```

**Validation:**
- `/mark` with `mode=upload`: `0 ≤ t_start < t_end`, `t_end - t_start ≤ 30.0`, both within video bounds.
- `/mark` with `mode=live`: rejects if session has no buffered frames yet (returns 409).
- `/monitor/live/connect`: validates RTSP URL with `urllib.parse` — scheme must be `rtsp://` or `rtsps://`. **No shell interpolation** — URL is passed only to MediaMTX API as JSON.

## 8. SSE Event Contracts (server → browser)

```typescript
{ type: "incident_started",
  incident_id, session_id, source_type, window_start_sec, window_end_sec, frames_count }

{ type: "incident_progress",
  incident_id, pct, current_frame, total_frames }

{ type: "incident_vehicle",
  incident_id, track_id, plate, chars, cls, plate_b64, vehicle_b64, ocr_frames, confidence }

{ type: "incident_rejected_vehicle",
  incident_id, track_id, plate, chars, cls, plate_b64, vehicle_b64, ocr_frames, reason }

{ type: "incident_complete",
  incident_id, total_vehicles, duration_ms, status: "completed" }

{ type: "incident_error",
  incident_id, message }
```

These mirror the existing `vehicle`/`progress`/`complete` shapes — frontend rendering components for vehicles are reused with minimal mapping.

## 9. MongoDB Schema

### `incidents` collection

```json
{
  "_id": "inc_a1b2c3d4",
  "session_id": "ses_xyz",
  "source_type": "live",
  "source_ref": "rtsp://10.0.0.5:554/main",
  "marked_at": "2026-05-20T10:24:51.331Z",
  "window_start_sec": 612.4,
  "window_end_sec": 622.4,
  "duration_sec": 10.0,
  "status": "completed",
  "vehicles": [
    {
      "track_id": 7,
      "plate_text": "30A-12345",
      "plate_text_confidence": 0.94,
      "chars": [["3", 0.99], ["0", 0.97]],
      "vehicle_class": "car",
      "plate_image_url": "https://...supabase.../incidents/inc_a1b2c3d4/plate_7.jpg",
      "vehicle_image_url": "https://...supabase.../incidents/inc_a1b2c3d4/vehicle_7.jpg",
      "ocr_method": "multiframe",
      "ocr_frames": 18,
      "first_seen_frame": 4,
      "last_seen_frame": 142
    }
  ],
  "total_vehicles": 1,
  "processing_ms": 4321,
  "created_at": "2026-05-20T10:24:51.331Z",
  "updated_at": "2026-05-20T10:24:55.652Z",
  "error_message": null
}
```

**Indexes:**
- `session_id`
- `marked_at` (descending — History view sort)
- `status`
- `source_type`

### Pydantic models (`api/database/models.py`)

```python
class IncidentVehicle(BaseModel):
    track_id: int
    plate_text: str
    plate_text_confidence: float
    chars: list[tuple[str, float]]
    vehicle_class: str
    plate_image_url: str | None
    vehicle_image_url: str | None
    ocr_method: Literal["multiframe", "segment_vote", "prob_vote"]
    ocr_frames: int
    first_seen_frame: int
    last_seen_frame: int

class Incident(BaseModel):
    incident_id: str
    session_id: str
    source_type: Literal["live", "upload"]
    source_ref: str
    marked_at: datetime
    window_start_sec: float
    window_end_sec: float
    duration_sec: float
    status: Literal["processing", "completed", "failed"]
    vehicles: list[IncidentVehicle] = []
    total_vehicles: int = 0
    processing_ms: int | None = None
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
```

## 10. Frontend Module Layout

```
web/src/
├── App.jsx                       ← mode switch: 'process' | 'monitor'
├── components/
│   ├── (existing)                ← unchanged
│   └── monitor/
│       ├── MonitorPage.jsx       ← top-level for the new page
│       ├── SourceSelector.jsx    ← RTSP URL input | Upload tab
│       ├── LiveViewer.jsx        ← WebRTC <video> + MJPEG fallback
│       ├── UploadViewer.jsx      ← HTML5 <video> + timeline + interval picker
│       ├── MarkBar.jsx           ← bottom controls (mode-aware)
│       ├── IntervalPicker.jsx    ← two-handle timeline overlay (upload mode)
│       ├── IncidentsPanel.jsx    ← right-side scrollable list
│       ├── IncidentCard.jsx      ← single incident summary (pending/done/failed)
│       └── IncidentDetail.jsx    ← expanded card / drawer
└── hooks/
    └── monitor/
        ├── useLiveSession.js     ← POST /connect, lifecycle, DELETE on unmount
        ├── useWebRTC.js          ← WHEP client (~60 lines, no library)
        ├── useIncidentStream.js  ← SSE consumer for incident_* events
        └── useMark.js            ← POST /mark, returns pending incident id
```

### Component responsibilities

- `MonitorPage` owns: `sessionId`, `mode` ('live'|'upload'), `incidents` map. Composes the layout.
- `LiveViewer` owns: WebRTC connection state, fallback logic. Renders `<video autoplay muted playsinline>`.
- `UploadViewer` owns: `videoRef`, currentTime tracking, interval-pick mode toggle.
- `IntervalPicker` is a controlled component — `[start, end]`, `onChange`, `onAnalyze`. Drags update player `currentTime` for preview.
- `IncidentsPanel` is dumb — renders whatever `incidents` it receives, sorted by `markedAt desc`.
- `IncidentCard` has three visual states: `pending` (spinner + elapsed timer), `completed` (plate text + thumb), `failed` (error + retry).

### WebRTC client (`useWebRTC.js`)

```javascript
useWebRTC(whepUrl) → { videoRef, status: 'connecting'|'live'|'error', error }

1. const pc = new RTCPeerConnection()
2. pc.addTransceiver('video', {direction:'recvonly'})
3. pc.addTransceiver('audio', {direction:'recvonly'})
4. const offer = await pc.createOffer()
5. await pc.setLocalDescription(offer)
6. const resp = await fetch(whepUrl, {
     method:'POST',
     headers:{'Content-Type':'application/sdp'},
     body: offer.sdp,
   })
7. const answerSdp = await resp.text()
8. await pc.setRemoteDescription({type:'answer', sdp:answerSdp})
9. pc.ontrack = e => { videoRef.current.srcObject = e.streams[0] }
10. cleanup: pc.close() in useEffect return
```

On `iceconnectionstatechange === 'failed'`, status flips to `'error'`. `LiveViewer` swaps in `<img src={mjpegUrl}>` and surfaces a banner: *"WebRTC unavailable — using MJPEG fallback (higher latency)"*.

### Incident state (in `MonitorPage`)

```javascript
incidents: {
  [incident_id]: {
    id: string,
    status: 'pending' | 'processing' | 'completed' | 'failed',
    sourceType: 'live' | 'upload',
    markedAt: ISO8601,
    windowStartSec: number,
    windowEndSec: number,
    pct: number,
    elapsedMs: number,
    vehicles: { [track_id]: {...} },
    rejected: { [track_id]: {...} },
    error: string | null,
  }
}
```

## 11. Visual Layouts

### Monitor page — live mode

```
┌─ Incident Monitor ────────────────────────────────────────────────────┐
│  Source: [RTSP ▼] rtsp://10.0.0.5:554/main  [Connected ●]            │
├───────────────────────────────────────────────────┬───────────────────┤
│                                                   │  Incidents (3)    │
│   ┌─────────────────────────────────────────┐    │ ┌───────────────┐ │
│   │           VIDEO PLAYER (WebRTC)          │    │ │ #3  10:24:51  │ │
│   │       Fallback: MJPEG <img>             │    │ │ 10s · 2 plates│ │
│   └─────────────────────────────────────────┘    │ │ 30A-12345 ... │ │
│                                                   │ └───────────────┘ │
│   ● Live · 00:12:43         Buffer: ▓▓▓▓▓░  10s  │ ┌───────────────┐ │
│                                                   │ │ #2  pending…  │ │
│   ┌─────────────────────────────────────────┐    │ └───────────────┘ │
│   │      [ 🚩 Mark Now ]                     │    │ ┌───────────────┐ │
│   └─────────────────────────────────────────┘    │ │ #1  done      │ │
│                                                   │ └───────────────┘ │
└───────────────────────────────────────────────────┴───────────────────┘
```

### Monitor page — upload mode (interval picker overlay)

```
┌─ Incident Monitor ────────────────────────────────────────────────────┐
│  Source: [Upload ▼]  long-video.mp4 (28:14 total)                    │
├───────────────────────────────────────────────────┬───────────────────┤
│                                                   │  Incidents (1)    │
│   ┌─────────────────────────────────────────┐    │                   │
│   │     VIDEO PLAYER (paused at 14:32)       │    │ ┌───────────────┐ │
│   └─────────────────────────────────────────┘    │ │ #1  pending…  │ │
│                                                   │ └───────────────┘ │
│   ┌─────────────────────────────────────────┐    │                   │
│   │ Timeline:                                │    │                   │
│   │ 00:00 ──────●═════════●────── 28:14      │    │                   │
│   │          14:24       14:39   Δ = 15.0 s  │    │                   │
│   │ [ Analyze ] (≤30s)  [ Cancel ]           │    │                   │
│   └─────────────────────────────────────────┘    │                   │
└───────────────────────────────────────────────────┴───────────────────┘
```

## 12. Configuration

### `configs/mediamtx.yml` (new)

```yaml
api: yes
apiAddress: :9997
webrtc: yes
webrtcAddress: :8889
rtspAddress: :8554
paths: {}        # no static paths — all added dynamically by FastAPI
```

### `docker-compose.yml` (additions)

```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:latest
    container_name: alpr-mediamtx
    ports:
      - "8554:8554"          # RTSP (FastAPI reads from here)
      - "8889:8889"          # WebRTC (browser reads from here)
      - "8189:8189/udp"      # WebRTC ICE
      - "9997:9997"          # HTTP API (FastAPI manages paths here)
    volumes:
      - ./configs/mediamtx.yml:/mediamtx.yml
    restart: unless-stopped
```

### `api/.env` (additions)

```
MEDIAMTX_API_URL=http://mediamtx:9997
MEDIAMTX_INTERNAL_RTSP_BASE=rtsp://mediamtx:8554
MEDIAMTX_PUBLIC_WEBRTC_BASE=http://localhost:8889
MEDIAMTX_PUBLIC_MJPEG_BASE=http://localhost:8000
```

The `_PUBLIC_*` URLs are what the browser receives in API responses. The `_INTERNAL_*` URL is what FastAPI uses inside the docker network.

## 13. Failure Handling

| Failure | Detection | User-facing behavior |
|---|---|---|
| MediaMTX path-add fails | `mediamtx_client.add_path` raises | 502 from `/connect` with reason; UI: "Could not connect — check URL/credentials" |
| RTSP camera disconnects | `cv2.VideoCapture.read()` returns False ×3 | Session terminates; SSE error event; UI: "Stream lost — reconnect?" |
| WebRTC ICE failure | `pc.iceconnectionstatechange === 'failed'` | Auto-swap to MJPEG fallback; banner: "WebRTC unavailable — using MJPEG (higher latency)" |
| Mark while buffer empty | Live session has fewer than `fps * 1.0` buffered frames (<1s of footage) | 409 from `/mark`; UI: "Buffer still warming up — wait 1–2s" |
| Mark interval > 30s | Server-side validation in `/mark` | 400; UI: "Window too long (max 30s)" — also disabled in IntervalPicker |
| GPU OOM during analysis | Caught in `run_incident` | `incident_error` SSE event; card shows red "Failed" + retry |
| Concurrent mark on busy GPU | Single-worker pool queues | Card shows "Queued" then "Analyzing" |

## 14. Testing Strategy

### Unit tests (mocked models)

- `tests/test_frame_source.py`
  - `FileFrameSource` seeks to `t_start` correctly, stops at `t_end`, reports `fps`.
  - Edge cases: `t_start=0`, `t_end > duration`, `t_start==t_end`.
- `tests/test_live_session.py`
  - Rolling buffer evicts old frames at `maxlen`.
  - `snapshot_window` returns chronologically-ordered frames.
  - `snapshot_window(seconds=10)` clamps to `maxlen`.
- `tests/test_mediamtx_client.py`
  - `add_path` POSTs the right JSON; surfaces errors.
  - `remove_path` is idempotent on 404.
  - Mocked with `httpx.MockTransport`.
- `tests/test_incident_analyzer.py`
  - Given fixed frames + mocked `ModelBundle`, emits the correct event sequence in the right order with the correct `incident_id`.

### Integration tests

- `tests/test_monitor_routes.py`
  - `POST /monitor/upload` with a fixture mp4 → returns session_id.
  - `POST /mark` with `mode=upload, t_start=2, t_end=7` → returns incident_id.
  - `GET /incidents/stream` yields started/progress/complete events.
  - Pipeline mocked to a tiny stub.
- `tests/test_pipeline_core_parity.py` — **regression guard**
  - Runs `run_job(fixture_video)` (legacy entry point).
  - Runs `process_frames(FileFrameSource(fixture_video, 0, None))` directly.
  - Asserts both produce identical sequences of `vehicle` events (same `track_id`, same `plate`, same `chars`).

### E2E (manual, documented)

- Connect to a public RTSP test stream → mark → verify incident card appears.
- Upload a sample video → drag interval → analyze → verify plates match the offline pipeline.
- Kill MediaMTX mid-session → verify UI shows clear error and offers MJPEG fallback path.

### Coverage target

- 80% on new modules: `pipeline_core`, `frame_source`, `live_session`, `incident_analyzer`, `mediamtx_client`.
- Existing modules retain their current coverage (verified by `pytest --cov`).

## 15. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Regression in upload flow during `run_job` refactor | `test_pipeline_core_parity` asserts identical vehicle outputs before/after. |
| Two simultaneous marks contending for GPU | `concurrent.futures.ThreadPoolExecutor(max_workers=1)` for incident analyzer. Marks return `incident_id` immediately; SSE surfaces queue position if needed. |
| RTSP camera disconnects mid-session | `LiveSession` decoder retries 3× with 1s backoff on `cap.read() == False`, then emits `incident_error` and tears down. |
| WebRTC fails across NAT/firewall | MJPEG fallback wired into `LiveViewer`. |
| Rolling buffer memory at native resolution | Cap at 2 concurrent live sessions. ~3.6GB RAM total worst case at 1080p30. Document in deployment notes. |
| MediaMTX container restart loses dynamic paths | Acceptable for v1: UI shows "Stream lost"; user reconnects. Future: persist active paths in Redis/Mongo if needed. |
| Vietnamese plate regex rejects valid plates | Reuse existing `_VN_PLATE_RE` and `_run_multiframe_ocr` rejection path; rejected vehicles still emitted as `incident_rejected_vehicle` so the operator can inspect crops. |

## 16. Security Considerations

- **RTSP URL handling:** URL is validated for scheme (`rtsp://` or `rtsps://`) and passed only as JSON to MediaMTX API. Never interpolated into shell commands. MediaMTX itself dials the camera; we never `exec ffmpeg`.
- **Credentials in URLs:** If the user supplies `rtsp://user:pass@host/...`, MediaMTX stores it in memory; we never log the full URL — only the host/path portion. Document in code comments.
- **CORS for WHEP:** MediaMTX's WebRTC endpoint must allow the browser origin. Configure `webrtcAllowOrigin: '*'` for development; tighten for production.
- **Upload size:** Existing `/upload` already accepts arbitrary file sizes — reuse the same limit (none yet enforced; add `max_upload_size` config in future hardening).
- **Incident persistence:** No authorization on incident records in v1. Add session-scoped access control later if multi-tenant.

## 17. Open Questions

1. **Long-term recording:** Should we enable MediaMTX `record: yes` (HLS segments to disk) to support marks beyond the 10s rolling buffer? Out of scope for v1; revisit if operators report 10s isn't enough.
2. **Live detection bounding boxes:** Operators may want a hint of where vehicles are without OCR running. Out of scope for v1; can layer on later by running detection-only at 5fps with separate SSE event.
3. **Cross-session incident view:** A "global" history of all incidents across all sessions. Already supported by `GET /incidents` route, but UI surface not designed yet — likely added to existing `HistoryModal`.

## 18. References

- MediaMTX: https://github.com/bluenviron/mediamtx
- WHEP spec (WebRTC-HTTP Egress Protocol): https://datatracker.ietf.org/doc/draft-murillo-whep/
- Existing realtime streaming design: `docs/superpowers/specs/2026-05-13-realtime-streaming-design.md`
- Existing pipeline: `api/core/pipeline.py`, `api/core/tracker.py`, `api/core/tracker_adapter.py`
