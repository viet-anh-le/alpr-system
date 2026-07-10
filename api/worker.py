"""api/worker.py — GPU inference worker.

Runs as its own process/container (`python -m api.worker`), separate from the
web API. It loads the ALPR models once, then consumes jobs from the Redis queue
and runs the *unchanged* ``run_job`` pipeline. Progress/frame/vehicle/complete
events are appended to a per-job Redis Stream that the API's SSE endpoint reads.

Why a separate process solves the stated goals:
  * availability — the worker has its own restart policy; a crash here never
    takes down the web API.
  * throughput / p95-p99 — heavy video inference no longer runs inside the API
    event loop, so light web requests stay fast under load.
  * fairness — a long video occupies a worker slot, but other jobs simply wait
    in the Redis queue (instead of being hard-rejected with 429) and are picked
    up as slots free. Per-user in-flight caps (enforced at enqueue) stop one
    user from monopolising every slot.

Scale by running more replicas of this service and/or raising WORKER_CONCURRENCY
(bounded by VRAM).
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import redis

from api.core import jobstore
from api.core.config import (
    ALPR_PREPROCESSED_VIDEO_DIR,
    ALPR_PREPROCESSED_VIDEO_TTL_SEC,
    MONGODB_DB_NAME,
    MONGODB_URI,
)
from api.core.models import load_models
from api.core.pipeline import run_job
from api.core.preprocessed_video import build_preprocessed_video_path
from api.database.mongodb import close_db, init_db

logger = logging.getLogger("alpr.worker")

# How many videos ONE worker processes concurrently. VRAM-bound; scale out with
# more replicas for more parallelism.
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "2"))
# Delete preprocessed-video files older than this from the shared dir.
_ARTIFACT_SWEEP_INTERVAL_SEC = int(os.environ.get("ALPR_ARTIFACT_SWEEP_INTERVAL_SEC", "600"))
# How often to scan for jobs orphaned by a crashed worker and reclaim them.
_RECLAIM_INTERVAL_SEC = int(os.environ.get("ALPR_RECLAIM_INTERVAL_SEC", "30"))
# How often a live worker refreshes its in-flight entries' idle time. Must be
# comfortably shorter than jobstore.RECLAIM_IDLE_MS to avoid self-reclaim.
_HEARTBEAT_SEC = int(os.environ.get("ALPR_HEARTBEAT_SEC", "20"))

# Stream entry-ids of jobs this worker is currently processing (heartbeated so
# they aren't reclaimed as orphans while genuinely running).
_inflight: set[str] = set()

_executor = ThreadPoolExecutor(max_workers=max(1, WORKER_CONCURRENCY))


class SyncStreamEmitter:
    """Queue-shim passed to ``run_job`` in place of the old ``asyncio.Queue``.

    ``run_job`` emits via ``loop.call_soon_threadsafe(queue.put_nowait, event)``,
    so ``put_nowait`` runs in the event-loop thread and must not await — we use a
    synchronous Redis client and append to the job's stream. XADD is sub-millisecond
    locally, keeping event ordering intact.
    """

    def __init__(self, job_id: str, sync_redis: "redis.Redis") -> None:
        self.job_id = job_id
        self._r = sync_redis
        self._key = jobstore.events_key(job_id)

    def put_nowait(self, event: dict) -> None:
        try:
            self._r.xadd(
                self._key,
                {"data": json.dumps(event, ensure_ascii=False)},
                maxlen=jobstore.EVENT_MAXLEN,
                approximate=True,
            )
            self._r.expire(self._key, jobstore.EVENT_TTL_SEC)
        except Exception:  # never let telemetry kill a job
            logger.exception("failed to publish event for job %s", self.job_id)


async def _handle(entry_id: str, job: dict, models, loop, sync_redis, sem: asyncio.Semaphore) -> None:
    job_id = job.get("job_id")
    user_id = job.get("user_id")
    mode = job.get("preprocess_mode", "none")
    video_path = job.get("video_path")
    _inflight.add(entry_id)
    try:
        # Idempotency for reclaimed (redelivered) jobs: run_job unlinks the video
        # in its finally, so a missing file means this job already finished before
        # the crash — ack + skip instead of reprocessing it into a failure.
        if not video_path or not os.path.exists(video_path):
            logger.warning("job %s: video missing (already processed?); skipping reprocess", job_id)
            return
        await jobstore.set_status(job_id, "processing")
        emitter = SyncStreamEmitter(job_id, sync_redis)
        # run_job owns the video-file lifecycle (unlinks it in its finally) and
        # writes session/records to MongoDB. It never re-raises pipeline errors —
        # it emits an "error" event and marks the Mongo session failed itself.
        await loop.run_in_executor(
            _executor,
            functools.partial(
                run_job,
                video_path,
                job_id,
                emitter,             # queue shim -> Redis stream
                loop,
                models,
                {},                  # jobs dict (unused across processes)
                job.get("filename", "video.mp4"),
                None,                # mjpeg_queue — batch flow has no MJPEG
                mode,
                job.get("ocr_backend", "default"),
                user_id,
                {},                  # job_owners dict (unused)
            ),
        )
        # Register the preprocessed-video artifact (if produced) so the API,
        # running in another container, can serve the download from the shared
        # volume. run_job's own in-process registration is process-local.
        if mode != "none" and user_id:
            path = build_preprocessed_video_path(job_id)
            if path.exists() and path.stat().st_size > 0:
                await jobstore.register_artifact(
                    job_id, user_id, str(path), ALPR_PREPROCESSED_VIDEO_TTL_SEC
                )
    except Exception:
        logger.exception("worker crash while handling job %s", job_id)
    finally:
        # Terminal status frees the user's in-flight slot regardless of outcome;
        # authoritative per-job status lives in Mongo + the event stream.
        try:
            await jobstore.set_status(job_id, "done")
        except Exception:
            logger.exception("failed to finalize status for job %s", job_id)
        # Ack only after the job is fully handled: an unacked entry survives a
        # crash and is reclaimed/redelivered by a live worker.
        try:
            await jobstore.ack(entry_id)
        except Exception:
            logger.exception("failed to ack entry %s (job %s)", entry_id, job_id)
        _inflight.discard(entry_id)
        sem.release()


async def _sweep_artifacts_loop() -> None:
    """Delete stale preprocessed-video files whose Redis registry TTL has lapsed."""
    directory = Path(ALPR_PREPROCESSED_VIDEO_DIR)
    ttl = max(60.0, float(ALPR_PREPROCESSED_VIDEO_TTL_SEC))
    while True:
        await asyncio.sleep(_ARTIFACT_SWEEP_INTERVAL_SEC)
        try:
            now = time.time()
            for f in directory.glob("*.mp4"):
                try:
                    if now - f.stat().st_mtime > ttl:
                        f.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception:
            logger.exception("artifact sweep failed")


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    loop = asyncio.get_running_loop()

    if MONGODB_URI:
        await init_db(MONGODB_URI, MONGODB_DB_NAME)
    else:
        logger.warning("MONGODB_URI not set — DB persistence disabled.")

    logger.info("loading models…")
    models = await loop.run_in_executor(None, load_models)
    logger.info("models loaded; worker concurrency=%d", WORKER_CONCURRENCY)

    sync_redis = redis.Redis.from_url(jobstore.REDIS_URL, decode_responses=True)
    sem = asyncio.Semaphore(WORKER_CONCURRENCY)
    stop = asyncio.Event()
    consumer = os.environ.get("HOSTNAME") or f"worker-{os.getpid()}"
    await jobstore.ensure_group()
    logger.info("consumer=%s group=%s", consumer, jobstore.CONSUMER_GROUP)

    def _request_stop(*_a) -> None:
        logger.info("shutdown signal received; draining…")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # pragma: no cover
            pass

    async def _dispatch(entry_id: str, job: dict) -> bool:
        """Validate + launch a job under the concurrency semaphore. Returns
        False (and releases nothing) only if the slot should be freed by caller."""
        if not job.get("job_id") or not job.get("video_path"):
            logger.warning("dropping malformed job (acking): %s", job)
            await jobstore.ack(entry_id)
            return False
        asyncio.create_task(_handle(entry_id, job, models, loop, sync_redis, sem))
        return True

    async def _heartbeat_loop() -> None:
        # Keep this worker's in-flight entries "fresh" so only a dead worker's
        # jobs ever exceed the reclaim idle threshold.
        while not stop.is_set():
            await asyncio.sleep(_HEARTBEAT_SEC)
            try:
                await jobstore.heartbeat(consumer, list(_inflight))
            except Exception:
                logger.exception("heartbeat failed")

    async def _reclaim_loop() -> None:
        # Periodically reclaim jobs left pending by a crashed worker and reprocess.
        while not stop.is_set():
            await asyncio.sleep(_RECLAIM_INTERVAL_SEC)
            try:
                claimed = await jobstore.reclaim(consumer, count=WORKER_CONCURRENCY)
            except Exception:
                logger.exception("reclaim scan failed")
                continue
            for entry_id, job in claimed:
                logger.warning("reclaiming orphaned job %s (entry %s)", job.get("job_id"), entry_id)
                await sem.acquire()
                if stop.is_set():
                    sem.release()
                    return
                if not await _dispatch(entry_id, job):
                    sem.release()

    sweeper = asyncio.create_task(_sweep_artifacts_loop())
    reclaimer = asyncio.create_task(_reclaim_loop())
    heartbeater = asyncio.create_task(_heartbeat_loop())

    try:
        while not stop.is_set():
            await sem.acquire()
            if stop.is_set():
                sem.release()
                break
            try:
                entry = await jobstore.dequeue(consumer, timeout=5)
            except Exception:
                # A transient Redis error must never kill the worker (goal:
                # "server không chết"). Log, back off briefly, keep serving.
                logger.exception("dequeue failed; backing off")
                sem.release()
                await asyncio.sleep(1.0)
                continue
            if entry is None:            # idle timeout — re-check stop flag
                sem.release()
                continue
            entry_id, job = entry
            if not await _dispatch(entry_id, job):
                sem.release()
    finally:
        sweeper.cancel()
        reclaimer.cancel()
        heartbeater.cancel()
        _executor.shutdown(wait=True, cancel_futures=False)
        await jobstore.close_redis()
        await close_db()
        logger.info("worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
