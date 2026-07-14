"""
api/main.py — FastAPI routes only.

All heavy logic lives in core/:
  core/config.py   — constants & paths
  core/models.py   — ModelBundle, load_models, ocr_batch
  core/gates.py    — pre-OCR quality filters
  core/tracker.py  — Levenshtein fusion + temporal consistency
  core/pipeline.py — video processing job
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.auth import get_current_user, get_current_user_with_csrf, router as auth_router
from api.core.config import (
    MAX_UPLOAD_MB,
    MONGODB_DB_NAME,
    MONGODB_URI,
    WEB_ORIGIN,
    normalize_ocr_backend,
)
from api.core.chunk_upload import ChunkUploadStore
from api.core.models import load_models
from api.core import jobstore
from api.core.preprocessing import normalize_preprocess_mode
from api.database.models import User
from api.database.mongodb import close_db, init_db
import api.routes_monitor as routes_monitor

logger = logging.getLogger(__name__)

_mjpeg_queues: dict[str, asyncio.Queue] = {}   # MJPEG frame queues (live monitor)

DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".webm", ".mov", ".mkv"}

# ── Shared upload directory ───────────────────────────────────────────────────
# Assembled/received videos are handed to a separate worker container, so they
# must live on a volume both containers mount (not the process-local temp dir).
UPLOAD_DIR = Path(os.environ.get("ALPR_UPLOAD_DIR", tempfile.gettempdir())) / "alpr_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Chunked upload ────────────────────────────────────────────────────────────
# Proxies in front of the API cap request bodies (Cloudflare free = 100 MB), so a
# single POST /upload can't carry large videos. The client splits the file into
# sub-limit chunks; the server reassembles them on local disk, then runs the same
# job. The reassembled file is deleted by run_job's finally when the processing
# session ends; chunk parts are cleaned by the store (see api/core/chunk_upload).
_CHUNK_UPLOAD_TTL_SEC = int(os.environ.get("CHUNK_UPLOAD_TTL_SEC", str(60 * 60)))
_chunk_store = ChunkUploadStore(
    Path(os.environ.get("CHUNK_UPLOAD_DIR", tempfile.gettempdir())) / "alpr_chunk_uploads",
    ttl_sec=_CHUNK_UPLOAD_TTL_SEC,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Batch video jobs run in the worker; models are only needed here for the
    # live-monitor feature. Loading them is best-effort so the web API stays
    # available (uploads still enqueue) even if the GPU/models are unavailable.
    app.state.models = None
    if os.environ.get("ALPR_API_LOAD_MODELS", "true").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            app.state.models = load_models()
        except Exception:
            logger.exception("load_models failed — live-monitor disabled; uploads unaffected.")
    routes_monitor.start_monitor_cleanup_task()
    if MONGODB_URI:
        await init_db(MONGODB_URI, MONGODB_DB_NAME)
    else:
        logger.warning("MONGODB_URI not set — database persistence disabled.")
    try:
        yield
    finally:
        await routes_monitor.stop_monitor_cleanup_task()
        routes_monitor.cleanup_all_upload_sessions()
        _chunk_store.cleanup_all()
        await jobstore.close_redis()
        await close_db()
        routes_monitor._event_executor.shutdown(wait=False, cancel_futures=True)


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ALPR Web", lifespan=lifespan)
_cors_origins = [origin.strip() for origin in WEB_ORIGIN.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health probes ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    """Liveness: process is up. No dependencies — used to auto-restart a hung API."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    """Readiness: Redis (queue/SSE substrate) reachable. 503 if not, so the proxy
    stops routing traffic until dependencies recover."""
    if not await jobstore.ping():
        raise HTTPException(status_code=503, detail="redis unavailable")
    return {"status": "ready"}


# ── API routes ────────────────────────────────────────────────────────────────

def _serialize_model(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)


def _user_id(user: User) -> str:
    if user.id is None:
        raise HTTPException(status_code=401, detail="Invalid user")
    return str(user.id)


def _validate_video_meta(file: UploadFile) -> str:
    """Validate extension + content-type without reading the body."""
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ")

    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/")
        or content_type in {"application/octet-stream", "application/x-matroska"}
    ):
        raise HTTPException(status_code=400, detail="File upload phải là video")
    return suffix


async def _stream_upload_to_shared(file: UploadFile, suffix: str) -> str:
    """Stream an upload to the shared volume in bounded chunks (no full-file
    buffering in RAM), enforcing the size cap as bytes arrive."""
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir=str(UPLOAD_DIR))
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413, detail=f"Video vượt quá giới hạn {MAX_UPLOAD_MB} MB"
                    )
                out.write(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="Video rỗng")
    except BaseException:
        _safe_unlink(tmp)
        raise
    return tmp


def _safe_unlink(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _normalize_modes(preprocess_mode: str, ocr_backend: str) -> tuple[str, str]:
    """Normalize preprocess + OCR backend, mapping ValueError to HTTP 400."""
    try:
        normalized_mode = normalize_preprocess_mode(preprocess_mode)
        normalized_ocr_backend = normalize_ocr_backend(ocr_backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runtime_ocr_backend = (
        "default" if ocr_backend.strip().lower() == "default" else normalized_ocr_backend
    )
    return normalized_mode, runtime_ocr_backend


def _validate_video_suffix(filename: str | None) -> str:
    suffix = Path(filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ")
    return suffix


async def _enqueue_video_job(
    current_user: User,
    tmp: str,
    filename: str,
    normalized_mode: str,
    runtime_ocr_backend: str,
) -> dict:
    """Enqueue an already-assembled video for the GPU worker(s).

    The video lives on the shared upload volume; a worker reads it, runs the
    pipeline, and unlinks it when done. A per-user in-flight cap prevents one
    user from monopolising the workers — over the cap we reject with 429 and
    remove the temp file immediately. Unlike the old in-process semaphore, jobs
    under the cap always queue (never hard-rejected for server busyness), so a
    long video never blocks other users' requests.
    """
    user_id = _user_id(current_user)
    active = await jobstore.user_active_count(user_id)
    if active >= jobstore.MAX_INFLIGHT_PER_USER:
        _safe_unlink(tmp)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Bạn đang có {active} video trong hàng đợi/đang xử lý "
                f"(tối đa {jobstore.MAX_INFLIGHT_PER_USER}). Vui lòng chờ hoàn tất."
            ),
        )
    job_id = uuid.uuid4().hex[:8]
    await jobstore.enqueue_job(job_id, user_id, {
        "job_id": job_id,
        "video_path": tmp,
        "filename": filename,
        "preprocess_mode": normalized_mode,
        "ocr_backend": runtime_ocr_backend,
        "user_id": user_id,
    })
    return {
        "job_id": job_id,
        "preprocess_mode": normalized_mode,
        "ocr_backend": runtime_ocr_backend,
        "processed_video_expected": normalized_mode != "none",
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    preprocess_mode: str = Form("none"),
    ocr_backend: str = Form("default"),
    current_user: User = Depends(get_current_user_with_csrf),
) -> dict:
    normalized_mode, runtime_ocr_backend = _normalize_modes(preprocess_mode, ocr_backend)
    suffix = _validate_video_meta(file)
    tmp = await _stream_upload_to_shared(file, suffix)
    return await _enqueue_video_job(
        current_user, tmp, file.filename or "video.mp4",
        normalized_mode, runtime_ocr_backend,
    )


@app.post("/upload/chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form("video.mp4"),
    chunk: UploadFile = File(...),
    current_user: User = Depends(get_current_user_with_csrf),
) -> dict:
    """Receive one chunk of a large video and persist it on disk.

    The client sizes each chunk to stay under the fronting proxy's body limit
    (Cloudflare free = 100 MB). Parts are written per-index so retries are
    idempotent, then reassembled by /upload/complete.
    """
    _chunk_store.cleanup_expired()

    err = _chunk_store.validate_params(upload_id, chunk_index, total_chunks)
    if err:
        raise HTTPException(status_code=400, detail=err)

    owner = _user_id(current_user)
    existing = _chunk_store.get(upload_id)
    if existing is not None and existing["owner"] != owner:
        raise HTTPException(status_code=403, detail="Không có quyền với upload này")
    suffix = _validate_video_suffix(filename)
    meta = _chunk_store.begin_or_get(upload_id, owner, filename or "video.mp4", suffix)

    data = await chunk.read()
    if not _chunk_store.write_chunk(meta, chunk_index, data, MAX_UPLOAD_MB * 1024 * 1024):
        _chunk_store.discard(upload_id)
        raise HTTPException(
            status_code=413, detail=f"Video vượt quá giới hạn {MAX_UPLOAD_MB} MB"
        )

    return {
        "upload_id": upload_id,
        "received": _chunk_store.received_count(meta),
        "total_chunks": total_chunks,
    }


@app.post("/upload/complete")
async def upload_complete(
    request: Request,
    upload_id: str = Form(...),
    total_chunks: int = Form(...),
    preprocess_mode: str = Form("none"),
    ocr_backend: str = Form("default"),
    current_user: User = Depends(get_current_user_with_csrf),
) -> dict:
    """Reassemble uploaded chunks into one video on disk and start processing."""
    normalized_mode, runtime_ocr_backend = _normalize_modes(preprocess_mode, ocr_backend)

    meta = _chunk_store.get(upload_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Upload không tồn tại hoặc đã hết hạn")
    if meta["owner"] != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Không có quyền với upload này")

    missing = _chunk_store.missing_chunks(meta, total_chunks)
    if missing:
        _chunk_store.discard(upload_id)
        raise HTTPException(
            status_code=400,
            detail=f"Thiếu {len(missing)} mảnh (vd chunk {missing[0]}) — hãy tải lại",
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=meta["suffix"], dir=str(UPLOAD_DIR)
        ) as f:
            tmp = f.name
            written = _chunk_store.assemble_into(meta, total_chunks, f, max_bytes)
        if written == 0:
            raise HTTPException(status_code=400, detail="Video rỗng")
    except ValueError:
        _safe_unlink(tmp)
        _chunk_store.discard(upload_id)
        raise HTTPException(
            status_code=413, detail=f"Video vượt quá giới hạn {MAX_UPLOAD_MB} MB"
        )
    except BaseException:
        _safe_unlink(tmp)
        _chunk_store.discard(upload_id)
        raise

    # Parts are no longer needed once reassembled into `tmp`.
    _chunk_store.discard(upload_id)

    return await _enqueue_video_job(
        current_user, tmp, meta["filename"],
        normalized_mode, runtime_ocr_backend,
    )


@app.delete("/upload/chunk/{upload_id}")
async def abort_chunk_upload(
    upload_id: str,
    current_user: User = Depends(get_current_user_with_csrf),
) -> dict:
    """Explicitly discard an in-progress chunk upload (client cancels / leaves)."""
    meta = _chunk_store.get(upload_id)
    if meta is not None and meta["owner"] != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Không có quyền với upload này")
    _chunk_store.discard(upload_id)
    return {"ok": True}


@app.get("/records/{job_id}/{track_id}")
async def get_track_record(
    job_id: str,
    track_id: int,
    current_user: User = Depends(get_current_user),
) -> dict:
    from api.database.mongodb import get_record_by_track_for_user, is_db_configured

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    record = await get_record_by_track_for_user(job_id, track_id, _user_id(current_user))
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    return record.model_dump(mode="json")


@app.get("/jobs/{job_id}/preprocessed-video")
async def get_preprocessed_video(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    artifact = await jobstore.get_artifact(job_id, _user_id(current_user))
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(
        artifact["path"],
        media_type="video/mp4",
        filename=f"{job_id}-preprocessed.mp4",
    )


@app.get("/stream/{job_id}")
async def stream(job_id: str, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    owner = await jobstore.get_owner(job_id)
    if owner is None or owner != _user_id(current_user):
        return HTMLResponse("Job not found", status_code=404)

    async def gen():
        # jobstore.stream_events replays the full history (so a browser that
        # connects after the worker started still gets every event), then blocks
        # for live events, emitting pings on idle and stopping after complete/error.
        async for data in jobstore.stream_events(job_id):
            yield f"data: {data}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stream/{job_id}/mjpeg")
async def stream_mjpeg(job_id: str, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    mjpeg_queue = _mjpeg_queues.get(job_id)
    if mjpeg_queue is None:
        return HTMLResponse("Job not found", status_code=404)
    if await jobstore.get_owner(job_id) != _user_id(current_user):
        return HTMLResponse("Job not found", status_code=404)

    async def gen():
        try:
            while True:
                try:
                    frame_bytes = await asyncio.wait_for(mjpeg_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    break
                if frame_bytes is None:  # sentinel — pipeline finished
                    break
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
        finally:
            _mjpeg_queues.pop(job_id, None)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Auth and dashboard data routes ───────────────────────────────────────────
app.include_router(auth_router)


@app.get("/sessions")
async def list_sessions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
) -> dict:
    from api.database.mongodb import count_sessions_for_user, is_db_configured, list_sessions_for_user

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    user_id = _user_id(current_user)
    items = await list_sessions_for_user(user_id, limit=limit, offset=offset)
    total = await count_sessions_for_user(user_id)
    return {
        "items": [_serialize_model(item) for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


_HISTORY_FILTER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_VEHICLE_CLASS_PATTERN = re.compile(r"^[A-Za-z0-9_ -]{1,40}$")


def _validate_history_filter(name: str, value: str | None, pattern: re.Pattern[str]) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if not pattern.fullmatch(stripped):
        raise HTTPException(status_code=400, detail=f"Invalid {name}")
    return stripped


@app.get("/records")
async def list_records(
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session_id: str | None = Query(None, min_length=1, max_length=128),
    plate: str | None = Query(None, min_length=1, max_length=32),
    vehicle_class: str | None = Query(None, min_length=1, max_length=40),
    current_user: User = Depends(get_current_user),
) -> dict:
    from api.database.mongodb import (
        count_records_for_user,
        is_db_configured,
        list_records_for_user,
        summarize_records_for_user,
    )

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    safe_session_id = _validate_history_filter("session_id", session_id, _HISTORY_FILTER_PATTERN)
    safe_vehicle_class = _validate_history_filter(
        "vehicle_class",
        vehicle_class,
        _VEHICLE_CLASS_PATTERN,
    )
    safe_plate = plate.strip() if plate else None
    user_id = _user_id(current_user)
    filters = {
        "session_id": safe_session_id,
        "plate": safe_plate,
        "vehicle_class": safe_vehicle_class,
    }
    items = await list_records_for_user(
        user_id,
        limit=limit,
        offset=offset,
        **filters,
    )
    total = await count_records_for_user(user_id, **filters)
    summary = await summarize_records_for_user(user_id, **filters)
    return {
        "items": [_serialize_model(item) for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
        "summary": summary,
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, current_user: User = Depends(get_current_user)) -> dict:
    from api.database.mongodb import get_session_for_user, is_db_configured

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    session = await get_session_for_user(session_id, _user_id(current_user))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _serialize_model(session)


@app.get("/sessions/{session_id}/records")
async def get_session_records(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    from api.database.mongodb import (
        get_records_for_session_for_user,
        get_session_for_user,
        is_db_configured,
    )

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    if await get_session_for_user(session_id, _user_id(current_user)) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    records = await get_records_for_session_for_user(session_id, _user_id(current_user))
    return {"items": [_serialize_model(record) for record in records]}


# ── Mount monitor router ─────────────────────────────────────────────────────
app.include_router(routes_monitor.router)

# ── Serve React build (production) ───────────────────────────────────────────
# In development, Vite dev server proxies /upload and /stream to this backend.
# In production, run `npm run build` and FastAPI serves dist/ at root.

if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="spa")
else:
    @app.get("/")
    async def dev_hint() -> HTMLResponse:
        return HTMLResponse(
            "<h2>Backend running ✓</h2>"
            "<p>Start the React dev server: <code>cd web && npm run dev</code></p>",
            status_code=200,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
