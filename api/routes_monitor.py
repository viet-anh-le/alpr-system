"""HTTP routes for the Event Monitor feature.

Mounted under /monitor and /events by api/main.py.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from api.core.chunk_upload import ChunkUploadStore
from api.core.config import MAX_UPLOAD_MB, normalize_ocr_backend
from api.core.live_session import LiveSession, internal_mediamtx_path
from api.core.preprocessing import normalize_preprocess_mode

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level registries; main.py initialises them.
monitor_sessions: dict[str, dict] = {}   # session_id → {"kind", "path"|"live_session", ...}
event_queues: dict[str, asyncio.Queue] = {}   # session_id → SSE queue
_sessions_lock = threading.Lock()

_WEBRTC_PUBLIC_BASE = os.environ.get("MEDIAMTX_PUBLIC_WEBRTC_BASE", "http://localhost:8889")
_MJPEG_PUBLIC_BASE = os.environ.get("MEDIAMTX_PUBLIC_MJPEG_BASE", "")  # relative if blank
# Small MJPEG buffer keeps live latency low: a full queue drops new frames
# rather than accumulating a multi-second backlog. ~5 frames ≈ <0.5s at 12 fps.
_MJPEG_QUEUE_SIZE = int(os.environ.get("MJPEG_QUEUE_SIZE", "5"))

# Single-worker pool — only one mark analyzes at a time to avoid GPU contention.
_event_executor = ThreadPoolExecutor(max_workers=1)
_cleanup_task: asyncio.Task | None = None

MAX_INTERVAL_SEC = 30.0
MONITOR_UPLOAD_TTL_SEC = float(os.environ.get("MONITOR_UPLOAD_TTL_SEC", "3600"))
MONITOR_CLEANUP_INTERVAL_SEC = float(os.environ.get("MONITOR_CLEANUP_INTERVAL_SEC", "300"))
MONITOR_UPLOAD_DIR = Path(
    os.environ.get(
        "MONITOR_UPLOAD_DIR",
        str(Path(tempfile.gettempdir()) / "alpr_monitor_uploads"),
    )
)
MONITOR_UPLOAD_PREFIX = "monitor_upload_"


def _new_session_id() -> str:
    return f"mon_{uuid.uuid4().hex[:10]}"


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:10]}"


def _now() -> float:
    return time.time()


def _touch_session(sess: dict) -> None:
    sess["last_access_at"] = _now()


def _runtime_ocr_backend(value: str) -> str:
    normalized = normalize_ocr_backend(value)
    return "default" if value.strip().lower() == "default" else normalized


def _delete_file(path: str | os.PathLike[str] | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temp file %s", path)


def cleanup_upload_session(session_id: str, *, force: bool = False) -> dict:
    """Delete an upload session, deferring if an event still needs the file."""
    path: str | None = None
    with _sessions_lock:
        sess = monitor_sessions.get(session_id)
        if sess is None:
            return {"ok": True, "cleanup": "missing"}
        if sess.get("kind") != "upload":
            raise ValueError("Not an upload session")
        if int(sess.get("active_events", 0)) > 0 and not force:
            sess["cleanup_requested"] = True
            _touch_session(sess)
            return {"ok": True, "cleanup": "deferred"}

        path = sess.get("path")
        monitor_sessions.pop(session_id, None)
        event_queues.pop(session_id, None)

    _delete_file(path)
    return {"ok": True, "cleanup": "deleted"}


def _retain_upload_event(session_id: str) -> None:
    with _sessions_lock:
        sess = monitor_sessions.get(session_id)
        if sess is None or sess.get("kind") != "upload":
            raise RuntimeError("Upload session no longer exists")
        sess["active_events"] = int(sess.get("active_events", 0)) + 1
        _touch_session(sess)


def _release_upload_event(session_id: str) -> None:
    should_cleanup = False
    with _sessions_lock:
        sess = monitor_sessions.get(session_id)
        if sess is None or sess.get("kind") != "upload":
            return
        sess["active_events"] = max(0, int(sess.get("active_events", 0)) - 1)
        _touch_session(sess)
        should_cleanup = sess["active_events"] == 0 and bool(sess.get("cleanup_requested"))

    if should_cleanup:
        cleanup_upload_session(session_id)


def _run_event_with_upload_lifecycle(
    *,
    upload_session_id: str,
    run_event_fn,
    **kwargs,
) -> None:
    try:
        run_event_fn(**kwargs)
    except Exception as exc:
        logger.exception("Event %s failed before analyzer could emit an error", kwargs.get("event_id"))
        loop = kwargs.get("loop")
        queue = kwargs.get("queue")
        if loop is not None and queue is not None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "event_error",
                    "event_id": kwargs.get("event_id"),
                    "message": str(exc),
                },
            )
    finally:
        _release_upload_event(upload_session_id)


def cleanup_expired_upload_sessions(*, now: float | None = None) -> list[str]:
    current = _now() if now is None else now
    expired: list[str] = []
    with _sessions_lock:
        for session_id, sess in list(monitor_sessions.items()):
            if sess.get("kind") != "upload":
                continue
            if int(sess.get("active_events", 0)) > 0:
                continue
            last_access = float(sess.get("last_access_at", sess.get("created_at", current)))
            if current - last_access >= MONITOR_UPLOAD_TTL_SEC:
                expired.append(session_id)

    removed: list[str] = []
    for session_id in expired:
        result = cleanup_upload_session(session_id)
        if result.get("cleanup") == "deleted":
            removed.append(session_id)
    return removed


def cleanup_stale_upload_files(*, now: float | None = None) -> list[Path]:
    current = _now() if now is None else now
    if not MONITOR_UPLOAD_DIR.exists():
        return []

    with _sessions_lock:
        tracked_paths = {
            Path(sess["path"]).resolve()
            for sess in monitor_sessions.values()
            if sess.get("kind") == "upload" and sess.get("path")
        }

    removed: list[Path] = []
    for path in sorted(MONITOR_UPLOAD_DIR.glob(f"{MONITOR_UPLOAD_PREFIX}*")):
        try:
            if path.resolve() in tracked_paths:
                continue
            if current - path.stat().st_mtime < MONITOR_UPLOAD_TTL_SEC:
                continue
            path.unlink(missing_ok=True)
            removed.append(path)
        except OSError:
            logger.warning("Could not remove stale monitor upload %s", path)
    return removed


async def _monitor_upload_cleanup_loop() -> None:
    while True:
        cleanup_expired_upload_sessions()
        cleanup_stale_upload_files()
        _monitor_chunk_store.cleanup_expired()
        await asyncio.sleep(MONITOR_CLEANUP_INTERVAL_SEC)


def start_monitor_cleanup_task() -> asyncio.Task | None:
    global _cleanup_task
    if MONITOR_CLEANUP_INTERVAL_SEC <= 0:
        return None
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_monitor_upload_cleanup_loop())
    return _cleanup_task


async def stop_monitor_cleanup_task() -> None:
    global _cleanup_task
    task = _cleanup_task
    _cleanup_task = None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def cleanup_all_upload_sessions() -> None:
    with _sessions_lock:
        upload_session_ids = [
            session_id
            for session_id, sess in monitor_sessions.items()
            if sess.get("kind") == "upload"
        ]
    for session_id in upload_session_ids:
        cleanup_upload_session(session_id, force=True)
    _monitor_chunk_store.cleanup_all()


# ── Upload mode ───────────────────────────────────────────────────────────────


# Chunk parts for large monitor uploads (single POST is capped by Cloudflare's
# ~100 MB body limit). Reassembled into MONITOR_UPLOAD_DIR, then treated exactly
# like a normal monitor upload — so the file is deleted on session disconnect /
# TTL cleanup just like the single-POST path.
_ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_monitor_chunk_store = ChunkUploadStore(
    MONITOR_UPLOAD_DIR / "chunks", ttl_sec=MONITOR_UPLOAD_TTL_SEC
)


def _video_suffix(filename: str | None) -> str:
    raw = Path(filename or "").suffix.lower()
    return raw if raw in _ALLOWED_VIDEO_SUFFIXES else ".mp4"


def _safe_remove(path: str | None) -> None:
    if path:
        Path(path).unlink(missing_ok=True)


def _register_upload_session(
    tmp_path: str,
    filename: str,
    normalized_mode: str,
    normalized_ocr_backend: str,
) -> dict:
    """Register a monitor upload session for an already-written video file."""
    session_id = _new_session_id()
    now = _now()
    monitor_sessions[session_id] = {
        "kind": "upload",
        "path": tmp_path,
        "filename": filename or "video.mp4",
        "preprocess_mode": normalized_mode,
        "ocr_backend": normalized_ocr_backend,
        "created_at": now,
        "last_access_at": now,
        "active_events": 0,
        "cleanup_requested": False,
    }
    event_queues[session_id] = asyncio.Queue()
    return {
        "session_id": session_id,
        "video_url": f"/monitor/upload/{session_id}/video",
        "preprocess_mode": normalized_mode,
        "ocr_backend": normalized_ocr_backend,
    }


@router.post("/monitor/upload")
async def monitor_upload(
    file: UploadFile = File(...),
    preprocess_mode: str = Form("none"),
    ocr_backend: str = Form("default"),
) -> dict:
    """Accept a video file for monitor-mode playback + mark-driven analysis."""
    try:
        normalized_mode = normalize_preprocess_mode(preprocess_mode)
        normalized_ocr_backend = _runtime_ocr_backend(ocr_backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = _video_suffix(file.filename)
    MONITOR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=MONITOR_UPLOAD_PREFIX,
        dir=MONITOR_UPLOAD_DIR,
    ) as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
        tmp_path = f.name

    return _register_upload_session(
        tmp_path, file.filename or "video.mp4", normalized_mode, normalized_ocr_backend
    )


@router.post("/monitor/upload/chunk")
async def monitor_upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form("video.mp4"),
    chunk: UploadFile = File(...),
) -> dict:
    """Receive one chunk of a large monitor-mode video (see _monitor_chunk_store)."""
    _monitor_chunk_store.cleanup_expired()
    err = _monitor_chunk_store.validate_params(upload_id, chunk_index, total_chunks)
    if err:
        raise HTTPException(status_code=400, detail=err)

    meta = _monitor_chunk_store.begin_or_get(
        upload_id, owner="", filename=filename or "video.mp4", suffix=_video_suffix(filename)
    )
    data = await chunk.read()
    if not _monitor_chunk_store.write_chunk(meta, chunk_index, data, MAX_UPLOAD_MB * 1024 * 1024):
        _monitor_chunk_store.discard(upload_id)
        raise HTTPException(
            status_code=413, detail=f"Video vượt quá giới hạn {MAX_UPLOAD_MB} MB"
        )
    return {
        "upload_id": upload_id,
        "received": _monitor_chunk_store.received_count(meta),
        "total_chunks": total_chunks,
    }


@router.post("/monitor/upload/complete")
async def monitor_upload_complete(
    upload_id: str = Form(...),
    total_chunks: int = Form(...),
    preprocess_mode: str = Form("none"),
    ocr_backend: str = Form("default"),
) -> dict:
    """Reassemble monitor upload chunks and open the playback session."""
    try:
        normalized_mode = normalize_preprocess_mode(preprocess_mode)
        normalized_ocr_backend = _runtime_ocr_backend(ocr_backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    meta = _monitor_chunk_store.get(upload_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Upload không tồn tại hoặc đã hết hạn")

    missing = _monitor_chunk_store.missing_chunks(meta, total_chunks)
    if missing:
        _monitor_chunk_store.discard(upload_id)
        raise HTTPException(
            status_code=400,
            detail=f"Thiếu {len(missing)} mảnh (vd chunk {missing[0]}) — hãy tải lại",
        )

    MONITOR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=meta["suffix"], prefix=MONITOR_UPLOAD_PREFIX,
            dir=MONITOR_UPLOAD_DIR,
        ) as f:
            tmp_path = f.name
            written = _monitor_chunk_store.assemble_into(meta, total_chunks, f, max_bytes)
        if written == 0:
            raise HTTPException(status_code=400, detail="Video rỗng")
    except ValueError:
        _safe_remove(tmp_path)
        _monitor_chunk_store.discard(upload_id)
        raise HTTPException(
            status_code=413, detail=f"Video vượt quá giới hạn {MAX_UPLOAD_MB} MB"
        )
    except BaseException:
        _safe_remove(tmp_path)
        _monitor_chunk_store.discard(upload_id)
        raise

    _monitor_chunk_store.discard(upload_id)
    return _register_upload_session(
        tmp_path, meta["filename"], normalized_mode, normalized_ocr_backend
    )


@router.delete("/monitor/upload/chunk/{upload_id}")
async def monitor_abort_chunk_upload(upload_id: str) -> dict:
    """Explicitly discard an in-progress monitor chunk upload."""
    _monitor_chunk_store.discard(upload_id)
    return {"ok": True}


@router.delete("/monitor/upload/{session_id}")
async def monitor_upload_disconnect(session_id: str) -> dict:
    """Delete an upload session and remove its temp file."""
    sess = monitor_sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess["kind"] != "upload":
        raise HTTPException(status_code=400, detail="Not an upload session")
    return cleanup_upload_session(session_id)


@router.get("/monitor/upload/{session_id}/video")
async def monitor_upload_video(session_id: str) -> FileResponse:
    sess = monitor_sessions.get(session_id)
    if sess is None or sess["kind"] != "upload":
        raise HTTPException(status_code=404, detail="Session not found")
    _touch_session(sess)
    return FileResponse(sess["path"], media_type="video/mp4")


# ── Live mode ─────────────────────────────────────────────────────────────────


class ConnectBody(BaseModel):
    rtsp_url: str
    ocr_backend: str = "default"


@router.post("/monitor/live/connect")
async def monitor_live_connect(body: ConnectBody) -> dict:
    parsed = urlparse(body.rtsp_url)
    if parsed.scheme not in ("rtsp", "rtsps"):
        raise HTTPException(status_code=400, detail="URL must be rtsp:// or rtsps://")
    try:
        normalized_ocr_backend = _runtime_ocr_backend(body.ocr_backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = _new_session_id()
    existing_path = internal_mediamtx_path(body.rtsp_url)
    path = existing_path or f"live_{session_id[4:]}"  # MediaMTX path name
    owns_mediamtx_path = existing_path is None
    mjpeg_q: asyncio.Queue = asyncio.Queue(maxsize=_MJPEG_QUEUE_SIZE)

    sess = LiveSession(
        session_id=session_id,
        mediamtx_path=path,
        owns_mediamtx_path=owns_mediamtx_path,
    )
    try:
        sess.start(body.rtsp_url, mjpeg_queue=mjpeg_q)
    except Exception as exc:
        logger.exception("Live connect failed")
        raise HTTPException(status_code=502, detail=f"Could not connect: {exc}")

    monitor_sessions[session_id] = {
        "kind": "live",
        "live_session": sess,
        "mediamtx_path": path,
        "owns_mediamtx_path": owns_mediamtx_path,
        "mjpeg_queue": mjpeg_q,
        "rtsp_url": body.rtsp_url,
        "ocr_backend": normalized_ocr_backend,
    }
    event_queues[session_id] = asyncio.Queue()

    whep_url = f"{_WEBRTC_PUBLIC_BASE}/{path}/whep"
    mjpeg_url = f"{_MJPEG_PUBLIC_BASE}/monitor/live/{session_id}/mjpeg"
    return {"session_id": session_id, "whep_url": whep_url, "mjpeg_url": mjpeg_url}


@router.delete("/monitor/live/{session_id}")
async def monitor_live_disconnect(session_id: str) -> dict:
    sess = monitor_sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess["kind"] != "live":
        raise HTTPException(status_code=400, detail="Not a live session")
    monitor_sessions.pop(session_id, None)
    event_queues.pop(session_id, None)
    sess["live_session"].stop()
    return {"ok": True}


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


# ── Mark route ────────────────────────────────────────────────────────────────


class MarkBody(BaseModel):
    mode: Literal["live", "upload"]
    t_start: float | None = None
    t_end: float | None = None


def _dispatch_event(
    *,
    event_id: str,
    session_id: str,
    sess: dict,
    body: MarkBody,
    queue: asyncio.Queue,
    request: Request,
) -> None:
    """Build a FrameSource and submit run_event to the worker pool."""
    from api.core.frame_source import FileFrameSource, LiveBufferFrameSource
    from api.core.event_analyzer import run_event

    models = request.app.state.models if hasattr(request.app.state, "models") else None

    loop = asyncio.get_running_loop()
    retained_upload = False

    if body.mode == "upload":
        _retain_upload_event(session_id)
        retained_upload = True
        try:
            source = FileFrameSource(sess["path"], t_start=body.t_start, t_end=body.t_end)
            preprocess_mode = normalize_preprocess_mode(sess.get("preprocess_mode"))
            source_ref = sess["filename"]
            ws, we = body.t_start, body.t_end
        except Exception:
            _release_upload_event(session_id)
            raise
    else:
        live_sess = sess["live_session"]
        snap = live_sess.snapshot_window(seconds=10.0)
        if len(snap) < int(live_sess.fps * 1.0):
            raise HTTPException(status_code=409, detail="Buffer still warming up — wait 1–2s")
        source = LiveBufferFrameSource(snap, fps=live_sess.fps, frame_size=live_sess.frame_size)
        source_ref = sess["rtsp_url"]
        ws = snap[0][2]
        we = snap[-1][2]

    kwargs = {
        "event_id": event_id,
        "session_id": session_id,
        "source": source,
        "source_type": body.mode,
        "source_ref": source_ref,
        "window_start_sec": ws,
        "window_end_sec": we,
        "queue": queue,
        "loop": loop,
        "models": models,
        "ocr_backend": sess.get("ocr_backend", "default"),
        "preprocess_mode": normalize_preprocess_mode(sess.get("preprocess_mode")),
    }

    if body.mode == "upload":
        try:
            _event_executor.submit(
                _run_event_with_upload_lifecycle,
                upload_session_id=session_id,
                run_event_fn=run_event,
                **kwargs,
            )
        except Exception:
            if retained_upload:
                _release_upload_event(session_id)
            raise
    else:
        _event_executor.submit(run_event, **kwargs)


@router.post("/monitor/{session_id}/mark")
async def monitor_mark(session_id: str, body: MarkBody, request: Request) -> dict:
    sess = monitor_sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _touch_session(sess)

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

    event_id = _new_event_id()
    queue = event_queues[session_id]
    _dispatch_event(
        event_id=event_id,
        session_id=session_id,
        sess=sess,
        body=body,
        queue=queue,
        request=request,
    )
    return {"event_id": event_id}


# ── Event SSE + GET endpoints ──────────────────────────────────────────────


@router.get("/monitor/{session_id}/events/stream")
async def monitor_events_stream(session_id: str) -> StreamingResponse:
    queue = event_queues.get(session_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def gen():
        yield ": keep-alive\n\n"   # SSE comment — establishes stream immediately
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=60.0)
                if ev is None:     # sentinel — closes the stream gracefully
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/events/{event_id}")
async def get_event_route(event_id: str) -> dict:
    from api.database.mongodb import get_event, is_db_configured

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    event = await get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event.model_dump(mode="json")


@router.get("/events")
async def list_events_route(
    source: str | None = None,
    session_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    from api.database.mongodb import list_events, is_db_configured

    if not is_db_configured():
        return {"items": []}
    items = await list_events(session_id=session_id, source_type=source, limit=limit)
    return {"items": [i.model_dump(mode="json") for i in items]}
