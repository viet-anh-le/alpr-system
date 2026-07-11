"""core/jobstore.py — Redis-backed job substrate.

Replaces the previous in-process registries (``_jobs`` / ``_job_owners`` and the
in-memory preprocessed-video artifact dict) so the web API and the GPU worker(s)
can run as separate processes/containers and coordinate through Redis.

Responsibilities:
  * job queue            — a Redis list; API RPUSHes, workers BLPOP.
  * per-job event log     — one Redis Stream per job (XADD by worker,
                            XREAD by the API's SSE endpoint). Replayable +
                            blockable + ordered, so a browser that connects
                            after the job started still gets the full history.
  * job meta              — a Redis hash (owner, status, timestamps).
  * per-user in-flight set — bounds how many jobs one user can hold at once so a
                            single user cannot monopolise the workers.
  * artifact registry     — preprocessed-video download metadata (owner + path),
                            the file itself living on the shared volume.

Async client is used by the API; the worker emits stream events with a *sync*
client (see api/worker.py) because those writes happen inside the event-loop
thread via ``loop.call_soon_threadsafe`` and must not await.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import redis.asyncio as aioredis
import redis.exceptions as redis_exceptions

# ── Configuration ─────────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# The work queue is a Redis Stream with a consumer group (not a plain list), so a
# job a worker popped but never finished (crash mid-processing) stays in the
# group's pending list and is reclaimed/redelivered instead of being lost.
QUEUE_STREAM = os.environ.get("ALPR_QUEUE_STREAM", "alpr:qstream")
CONSUMER_GROUP = os.environ.get("ALPR_CONSUMER_GROUP", "alpr-workers")
# A pending (unacked) message whose idle time exceeds this is treated as
# belonging to a dead worker and reclaimed by a live one. A LIVE worker
# heartbeats its in-flight entries (resets their idle) more often than this, so
# a long-but-alive job is never wrongly reclaimed/double-processed.
RECLAIM_IDLE_MS = int(os.environ.get("ALPR_RECLAIM_IDLE_MS", "60000"))

JOB_TTL_SEC = int(os.environ.get("ALPR_JOB_TTL_SEC", str(6 * 3600)))
EVENT_TTL_SEC = int(os.environ.get("ALPR_EVENT_TTL_SEC", str(6 * 3600)))
EVENT_MAXLEN = int(os.environ.get("ALPR_EVENT_MAXLEN", "20000"))
# Max concurrent (queued + processing) jobs a single user may hold. Prevents one
# user's backlog of long videos from starving everyone else.
MAX_INFLIGHT_PER_USER = int(os.environ.get("ALPR_MAX_INFLIGHT_PER_USER", "3"))

_TERMINAL_STATUSES = {"completed", "failed", "done"}


# ── Key helpers ───────────────────────────────────────────────────────────────
def job_key(job_id: str) -> str:
    return f"alpr:job:{job_id}"


def events_key(job_id: str) -> str:
    return f"alpr:events:{job_id}"


def user_set_key(user_id: str) -> str:
    return f"alpr:user:{user_id}:jobs"


def artifact_key(job_id: str) -> str:
    return f"alpr:artifact:{job_id}"


# ── Async client (API side) ───────────────────────────────────────────────────
_redis: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def ping() -> bool:
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        finally:
            _redis = None


# ── Consumer group / enqueue / dequeue ────────────────────────────────────────
async def ensure_group() -> None:
    """Create the stream + consumer group if absent (idempotent)."""
    try:
        await get_redis().xgroup_create(QUEUE_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
    except redis_exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def user_active_count(user_id: str) -> int:
    """Number of jobs currently queued or processing for this user."""
    return int(await get_redis().scard(user_set_key(user_id)))


async def enqueue_job(job_id: str, owner: str, payload: dict[str, Any]) -> None:
    """Register job meta + user accounting, then append it to the work stream.

    ``payload`` must include everything the worker needs to run the job:
    job_id, video_path (on the shared volume), filename, preprocess_mode,
    ocr_backend, user_id.
    """
    r = get_redis()
    now = int(time.time())
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(
            job_key(job_id),
            mapping={
                "owner": owner,
                "status": "queued",
                "filename": payload.get("filename", "video.mp4"),
                "preprocess_mode": payload.get("preprocess_mode", "none"),
                "ocr_backend": payload.get("ocr_backend", "default"),
                "created_at": now,
            },
        )
        pipe.expire(job_key(job_id), JOB_TTL_SEC)
        pipe.sadd(user_set_key(owner), job_id)
        pipe.expire(user_set_key(owner), JOB_TTL_SEC)
        pipe.xadd(QUEUE_STREAM, {"data": json.dumps(payload, ensure_ascii=False)})
        await pipe.execute()


async def dequeue(consumer: str, timeout: int = 5) -> Optional[tuple[str, dict[str, Any]]]:
    """Read the next unread job for this consumer via the group.

    Returns ``(entry_id, payload)``; the caller MUST call ``ack(entry_id)`` once
    the job is fully handled, so an unacked entry survives a crash and can be
    reclaimed. Returns None when idle (no new messages / blocking read timeout).
    """
    try:
        resp = await get_redis().xreadgroup(
            CONSUMER_GROUP, consumer, {QUEUE_STREAM: ">"}, count=1, block=timeout * 1000
        )
    except redis_exceptions.TimeoutError:
        return None
    if not resp:
        return None
    _stream, entries = resp[0]
    if not entries:
        return None
    entry_id, fields = entries[0]
    try:
        return entry_id, json.loads(fields["data"])
    except (ValueError, TypeError, KeyError):
        await ack(entry_id)  # unparseable — drop it
        return None


async def ack(entry_id: str) -> None:
    """Acknowledge + remove a fully-handled entry from the stream."""
    r = get_redis()
    async with r.pipeline(transaction=False) as pipe:
        pipe.xack(QUEUE_STREAM, CONSUMER_GROUP, entry_id)
        pipe.xdel(QUEUE_STREAM, entry_id)
        await pipe.execute()


async def heartbeat(consumer: str, entry_ids: list[str]) -> None:
    """Reset the idle time of this worker's in-flight entries so a live worker's
    long-running jobs are never reclaimed as orphans. No-op if nothing in flight."""
    if not entry_ids:
        return
    try:
        await get_redis().xclaim(
            QUEUE_STREAM, CONSUMER_GROUP, consumer,
            min_idle_time=0, message_ids=list(entry_ids), justid=True,
        )
    except redis_exceptions.ResponseError:
        pass


async def reclaim(consumer: str, count: int = 10) -> list[tuple[str, dict[str, Any]]]:
    """Claim jobs left pending by a crashed worker (idle > RECLAIM_IDLE_MS).

    Uses XAUTOCLAIM, which scans the whole group's pending list, so any live
    worker can pick up a dead worker's in-flight jobs and reprocess them.
    """
    try:
        res = await get_redis().xautoclaim(
            QUEUE_STREAM, CONSUMER_GROUP, consumer, RECLAIM_IDLE_MS, start_id="0-0", count=count
        )
    except redis_exceptions.ResponseError:
        return []
    # redis-py returns [next_cursor, [(id, fields|None), ...], [deleted_ids]]
    entries = res[1] if len(res) > 1 else []
    claimed: list[tuple[str, dict[str, Any]]] = []
    for entry_id, fields in entries:
        if not fields:  # entry was deleted meanwhile
            await ack(entry_id)
            continue
        try:
            claimed.append((entry_id, json.loads(fields["data"])))
        except (ValueError, TypeError, KeyError):
            await ack(entry_id)
    return claimed


async def queue_depth() -> int:
    """Entries in the stream not yet acked+deleted (waiting + in-flight)."""
    return int(await get_redis().xlen(QUEUE_STREAM))


# ── Status / ownership ────────────────────────────────────────────────────────
async def get_owner(job_id: str) -> Optional[str]:
    return await get_redis().hget(job_key(job_id), "owner")


async def get_status(job_id: str) -> Optional[str]:
    return await get_redis().hget(job_key(job_id), "status")


async def set_status(job_id: str, status: str) -> None:
    """Update job status. On a terminal status, free the user's in-flight slot."""
    r = get_redis()
    owner = await r.hget(job_key(job_id), "owner")
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(job_key(job_id), "status", status)
        pipe.expire(job_key(job_id), JOB_TTL_SEC)
        if status in _TERMINAL_STATUSES and owner:
            pipe.srem(user_set_key(owner), job_id)
        await pipe.execute()


# ── Event stream (SSE) ────────────────────────────────────────────────────────
async def stream_events(
    job_id: str,
    *,
    block_ms: int = 15000,
) -> AsyncIterator[str]:
    """Yield each job event as a JSON string, oldest first, then live.

    Starts from the beginning of the stream so a late subscriber still replays
    the full history. Yields ``{"type":"ping"}`` on idle so the SSE connection
    stays warm, and stops after a terminal (``complete`` / ``error``) event.
    """
    r = get_redis()
    key = events_key(job_id)
    last_id = "0"
    while True:
        resp = await r.xread({key: last_id}, block=block_ms, count=100)
        if not resp:
            yield '{"type":"ping"}'
            continue
        for _stream, entries in resp:
            for entry_id, fields in entries:
                last_id = entry_id
                data = fields.get("data")
                if data is None:
                    continue
                yield data
                etype = _event_type(data)
                if etype in ("complete", "error"):
                    return


def _event_type(data: str) -> Optional[str]:
    try:
        return json.loads(data).get("type")
    except (ValueError, TypeError):
        return None


# ── Preprocessed-video artifact registry ──────────────────────────────────────
async def register_artifact(
    job_id: str,
    owner: str,
    path: str,
    ttl_sec: float,
) -> None:
    r = get_redis()
    ttl = max(1, int(ttl_sec))
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(
            artifact_key(job_id),
            mapping={"owner": owner, "path": path, "expires_at": int(time.time()) + ttl},
        )
        pipe.expire(artifact_key(job_id), ttl)
        await pipe.execute()


async def get_artifact(job_id: str, owner: str) -> Optional[dict[str, Any]]:
    """Return {owner, path} if it exists, belongs to ``owner`` and is on disk."""
    meta = await get_redis().hgetall(artifact_key(job_id))
    if not meta:
        return None
    if meta.get("owner") != owner:
        return None
    try:
        if int(meta.get("expires_at", "0")) <= int(time.time()):
            return None
    except (ValueError, TypeError):
        return None
    path = meta.get("path")
    if not path or not Path(path).exists():
        return None
    return meta
