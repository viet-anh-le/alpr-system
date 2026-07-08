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
from api.core.models import ModelBundle, load_models
from api.core.pipeline import run_job
from api.core.preprocessed_video import (
    clear_preprocessed_video_artifacts,
    cleanup_expired_preprocessed_video_artifacts,
    get_preprocessed_video_artifact,
    start_preprocessed_video_cleanup_task,
    stop_preprocessed_video_cleanup_task,
)
from api.core.preprocessing import normalize_preprocess_mode
from api.database.models import User
from api.database.mongodb import close_db, init_db
import api.routes_monitor as routes_monitor

logger = logging.getLogger(__name__)

_jobs:         dict[str, asyncio.Queue] = {}   # SSE event queues
_mjpeg_queues: dict[str, asyncio.Queue] = {}   # MJPEG frame queues
_job_owners:   dict[str, str] = {}              # job_id → user_id

DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".webm", ".mov", ".mkv"}

# ── GPU concurrency limiter ───────────────────────────────────────────────────
# Each ALPR video job consumes significant VRAM.  Limit concurrent jobs to
# avoid CUDA OOM on a single GPU (e.g. RunPod RTX 3090 / 24 GB).
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
_job_semaphore: asyncio.Semaphore | None = None  # created after event loop exists

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
    app.state.models = load_models()
    routes_monitor.start_monitor_cleanup_task()
    start_preprocessed_video_cleanup_task()
    if MONGODB_URI:
        await init_db(MONGODB_URI, MONGODB_DB_NAME)
    else:
        logger.warning("MONGODB_URI not set — database persistence disabled.")
    try:
        yield
    finally:
        await routes_monitor.stop_monitor_cleanup_task()
        await stop_preprocessed_video_cleanup_task()
        routes_monitor.cleanup_all_upload_sessions()
        clear_preprocessed_video_artifacts()
        _chunk_store.cleanup_all()
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


# ── API routes ────────────────────────────────────────────────────────────────

def _serialize_model(model) -> dict:
    return model.model_dump(mode="json", by_alias=True)


def _user_id(user: User) -> str:
    if user.id is None:
        raise HTTPException(status_code=401, detail="Invalid user")
    return str(user.id)


def _validate_video_file(file: UploadFile, data: bytes) -> str:
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ")

    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/")
        or content_type in {"application/octet-stream", "application/x-matroska"}
    ):
        raise HTTPException(status_code=400, detail="File upload phải là video")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Video vượt quá giới hạn {MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(status_code=400, detail="Video rỗng")
    return suffix


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


def _launch_video_job(
    request: Request,
    current_user: User,
    tmp: str,
    filename: str,
    normalized_mode: str,
    runtime_ocr_backend: str,
) -> dict:
    """Register + launch a processing job for an already-assembled temp video.

    On a queue-full rejection the temp file is removed immediately; otherwise
    run_job deletes it in its finally when the processing session ends.
    """
    global _job_semaphore
    if _job_semaphore is None:
        _job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    if _job_semaphore.locked():
        _safe_unlink(tmp)
        raise HTTPException(
            status_code=429,
            detail=f"Server đang xử lý tối đa {MAX_CONCURRENT_JOBS} video. Vui lòng thử lại sau.",
        )

    job_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue = asyncio.Queue()
    mjpeg_queue = None
    _jobs[job_id] = queue
    _job_owners[job_id] = _user_id(current_user)

    loop = asyncio.get_event_loop()

    async def _run_with_semaphore() -> None:
        async with _job_semaphore:
            await loop.run_in_executor(
                None, run_job, tmp, job_id, queue, loop, request.app.state.models, _jobs,
                filename, mjpeg_queue, normalized_mode, runtime_ocr_backend,
                _user_id(current_user), _job_owners,
            )

    asyncio.ensure_future(_run_with_semaphore())
    return {
        "job_id": job_id,
        "preprocess_mode": normalized_mode,
        "ocr_backend": runtime_ocr_backend,
        "processed_video_expected": normalized_mode != "none",
    }


@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    preprocess_mode: str = Form("none"),
    ocr_backend: str = Form("default"),
    current_user: User = Depends(get_current_user_with_csrf),
) -> dict:
    normalized_mode, runtime_ocr_backend = _normalize_modes(preprocess_mode, ocr_backend)
    cleanup_expired_preprocessed_video_artifacts()

    file_bytes = await file.read()
    suffix = _validate_video_file(file, file_bytes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        tmp = f.name

    return _launch_video_job(
        request, current_user, tmp, file.filename or "video.mp4",
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
    cleanup_expired_preprocessed_video_artifacts()

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=meta["suffix"]) as f:
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

    return _launch_video_job(
        request, current_user, tmp, meta["filename"],
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
    artifact = get_preprocessed_video_artifact(job_id, _user_id(current_user))
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(
        artifact.path,
        media_type="video/mp4",
        filename=f"{job_id}-preprocessed.mp4",
    )


@app.get("/stream/{job_id}")
async def stream(job_id: str, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    queue = _jobs.get(job_id)
    if queue is None:
        return HTMLResponse("Job not found", status_code=404)
    if _job_owners.get(job_id) != _user_id(current_user):
        return HTMLResponse("Job not found", status_code=404)

    async def gen():
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=60.0)
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev.get("type") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'

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
    if _job_owners.get(job_id) != _user_id(current_user):
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
    current_user: User = Depends(get_current_user),
) -> dict:
    from api.database.mongodb import is_db_configured, list_sessions_for_user

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    items = await list_sessions_for_user(_user_id(current_user), limit=limit)
    return {"items": [_serialize_model(item) for item in items]}


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
