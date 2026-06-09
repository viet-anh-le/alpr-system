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
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.core.config import ALPR_PREVIEW_FPS, MONGODB_DB_NAME, MONGODB_URI
from api.core.models import ModelBundle, load_models
from api.core.pipeline import run_job
from api.core.preprocessing import normalize_preprocess_mode
from api.database.mongodb import close_db, init_db
import api.routes_monitor as routes_monitor

logger = logging.getLogger(__name__)

_jobs:         dict[str, asyncio.Queue] = {}   # SSE event queues
_mjpeg_queues: dict[str, asyncio.Queue] = {}   # MJPEG frame queues

DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
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
        await close_db()
        routes_monitor._incident_executor.shutdown(wait=False, cancel_futures=True)


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ALPR Web", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API routes ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    preprocess_mode: str = Form("none"),
    ocr_backend: str = Form("default"),
) -> dict:
    try:
        normalized_mode = normalize_preprocess_mode(preprocess_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id      = uuid.uuid4().hex[:8]
    queue       = asyncio.Queue()
    mjpeg_queue = asyncio.Queue(maxsize=60) if ALPR_PREVIEW_FPS > 0 else None
    _jobs[job_id]         = queue
    if mjpeg_queue is not None:
        _mjpeg_queues[job_id] = mjpeg_queue

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await file.read())
        tmp = f.name

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, run_job, tmp, job_id, queue, loop, request.app.state.models, _jobs,
        file.filename or "video.mp4", mjpeg_queue, normalized_mode, ocr_backend
    )
    return {"job_id": job_id, "preprocess_mode": normalized_mode, "ocr_backend": ocr_backend}


@app.get("/records/{job_id}/{track_id}")
async def get_track_record(job_id: str, track_id: int) -> dict:
    from api.database.mongodb import get_record_by_track, is_db_configured

    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    record = await get_record_by_track(job_id, track_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    return record.model_dump(mode="json")


@app.get("/stream/{job_id}")
async def stream(job_id: str) -> StreamingResponse:
    queue = _jobs.get(job_id)
    if queue is None:
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
async def stream_mjpeg(job_id: str) -> StreamingResponse:
    mjpeg_queue = _mjpeg_queues.get(job_id)
    if mjpeg_queue is None:
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
