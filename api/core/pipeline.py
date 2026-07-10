"""
core/pipeline.py — Video processing job (runs in thread-pool).

SSE event types emitted:
  "progress"  — frame index / total / pct
  "frame"     — preview JPEG + bounding boxes for frontend overlay
  "vehicle"   — per-vehicle OCR update (plate_b64 + vehicle_b64)
                Emitted AFTER track loss (not per-frame) in the new flow.
  "complete"  — final summary
  "error"     — exception info

OCR flow change (motion-based ALPR):
  Old: detect plate → OCR every stride → Levenshtein fuse
  New: detect plate → quality_score → buffer crop
       OCR each crop with the single-frame model
       on track loss (LOST_THRESHOLD strides absent) → segment/probability vote
       at video end → finalise remaining buffered tracks
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os

import cv2

from .database import upload_image as _storage_upload
from .models import ModelBundle
from .plate_format import chars_to_display_text
from .tracker import WebTrackletManager
from .config import (
    ALPR_DEBUG_TIMINGS,
)

logger = logging.getLogger(__name__)


# ── MongoDB helpers (called from thread-pool via run_coroutine_threadsafe) ────


def _session_create(
    session_id: str,
    filename: str,
    loop: asyncio.AbstractEventLoop,
    user_id: str | None = None,
    preprocess_mode: str = "none",
    ocr_backend: str = "default",
) -> None:
    """Create (or reset) the MongoDB session document for this job."""
    try:
        from api.database.mongodb import is_db_configured, upsert_session
        from api.database.models import RecognitionSession

        if not is_db_configured():
            return
        session = RecognitionSession(
            session_id=session_id,
            user_id=user_id,
            source_filename=filename,
            status="processing",
            preprocess_mode=preprocess_mode,
            ocr_backend=ocr_backend,
        )
        asyncio.run_coroutine_threadsafe(upsert_session(session), loop).result(timeout=5)
    except Exception:
        logger.exception("MongoDB: failed to create session %s", session_id)


def _session_update(session_id: str, patch: dict, loop: asyncio.AbstractEventLoop) -> None:
    """Partially update the session document (status, counters, etc.)."""
    try:
        from api.database.mongodb import is_db_configured, update_session

        if not is_db_configured():
            return
        asyncio.run_coroutine_threadsafe(update_session(session_id, patch), loop).result(timeout=5)
    except Exception:
        logger.exception("MongoDB: failed to update session %s", session_id)


class _FrozenTrackerView:
    """Read-only snapshot of ONE track's persistence state, captured synchronously
    before the track is released so the async DB save never touches the live
    (possibly already-cleaned-up) tracker.

    Exposes exactly the attributes/methods that ``_record_save`` reads; the
    captured TrackBuffer / image objects stay alive via these references even
    after ``WebTrackletManager.release_track()`` pops them from the tracker dicts.
    """

    def __init__(self, tracker: WebTrackletManager, tid: int) -> None:
        buf = tracker._buffers.get(tid)
        self._buffers = {tid: buf} if buf is not None else {}
        vimg = tracker._vehicle_img.get(tid)
        self._vehicle_img = {tid: vimg} if vimg is not None else {}
        self._cls = {tid: tracker._cls.get(tid, "vehicle")}
        self._clusters = list(tracker.cluster_results(tid))
        self._vehicle_track_id = tracker.vehicle_track_id(tid)
        self._plate_track_id = tracker.plate_track_id(tid)

    def cluster_results(self, tid: int) -> list[dict]:
        return self._clusters

    def vehicle_track_id(self, tid: int) -> int | None:
        return self._vehicle_track_id

    def plate_track_id(self, tid: int) -> int | None:
        return self._plate_track_id


def _record_save(
    session_id: str,
    tid: int,
    tracker: WebTrackletManager,
    char_probs: list[tuple[str, float]],
    ocr_method: str,
    vote_summary: dict[str, int],
    loop: asyncio.AbstractEventLoop,
    user_id: str | None = None,
) -> None:
    """
    Build a RecognitionRecord from finalized tracker state and fire-and-forget
    save it to MongoDB.  Never raises — all errors are logged.

    ``tracker`` may be a live WebTrackletManager or a _FrozenTrackerView snapshot.

    All images are uploaded to Supabase Storage; only public URLs are stored
    in MongoDB (no base64 blobs).
    """
    try:
        from api.database.mongodb import is_db_configured, upsert_record
        from api.database.models import (
            PlateFrame as DBFrame,
            RecognitionCluster as DBCluster,
            RecognitionRecord as DBRecord,
        )

        if not is_db_configured():
            return

        buf = tracker._buffers.get(tid)
        if not buf or not buf.crops:
            return

        entries = buf.top_k_entries(k=buf.max_size)
        if not entries:
            return

        def _db_frame_from_entry(entry, path: str, quality: int = 85) -> DBFrame:
            _, jpg = cv2.imencode(".jpg", entry.crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            url = _storage_upload(
                "evidence", path, bytes(jpg)
            )
            return DBFrame(
                frame_index=int(entry.frame_idx),
                quality_score=round(float(entry.quality_score), 4),
                image_url=url,
                ocr_text=chars_to_display_text(entry.char_probs) if entry.char_probs else None,
                ocr_confidence=round(float(entry.ocr_conf), 4),
            )

        def _db_frame_from_payload(frame: dict, path: str) -> DBFrame:
            url = frame.get("image_url")
            image_b64 = frame.get("image_b64")
            if image_b64:
                url = _storage_upload("evidence", path, base64.b64decode(image_b64))
            return DBFrame(
                frame_index=int(frame.get("frame_index", 0)),
                quality_score=round(float(frame.get("quality_score", 0.0)), 4),
                image_url=url,
                ocr_text=frame.get("ocr_text"),
                ocr_confidence=(
                    round(float(frame["ocr_confidence"]), 4)
                    if frame.get("ocr_confidence") is not None
                    else None
                ),
            )

        # Upload every buffered crop and build track_buffer
        track_frames: list[DBFrame] = []
        for i, entry in enumerate(entries):
            track_frames.append(
                _db_frame_from_entry(
                    entry,
                    f"{session_id}/track_{tid}_frame_{i}.jpg",
                )
            )

        if not track_frames:
            return

        best_entry = entries[0]

        best_plate_frame = _db_frame_from_entry(
            best_entry,
            f"{session_id}/plate_{tid}.jpg",
            quality=90,
        )

        # Vehicle thumbnail
        vehicle_url: str | None = None
        vehicle_img = tracker._vehicle_img.get(tid)
        if vehicle_img is not None:
            _, jpg = cv2.imencode(".jpg", vehicle_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            vehicle_url = _storage_upload("evidence", f"{session_id}/vehicle_{tid}.jpg", bytes(jpg))

        plate_text = chars_to_display_text(char_probs)
        plate_conf = sum(p for _, p in char_probs) / len(char_probs) if char_probs else 0.0
        clusters: list[DBCluster] = []
        for fallback_index, cluster in enumerate(tracker.cluster_results(tid)):
            cluster_index = int(cluster.get("cluster_index", fallback_index))
            cluster_frames = [
                _db_frame_from_payload(
                    frame,
                    f"{session_id}/track_{tid}_cluster_{cluster_index}_frame_{i}.jpg",
                )
                for i, frame in enumerate(cluster.get("track_buffer") or [])
            ]
            if not cluster_frames:
                continue

            best_cluster_frame = cluster_frames[0]
            if cluster.get("plate_b64"):
                best_cluster_url = _storage_upload(
                    "evidence",
                    f"{session_id}/plate_{tid}_cluster_{cluster_index}.jpg",
                    base64.b64decode(cluster["plate_b64"]),
                )
                best_cluster_frame = best_cluster_frame.model_copy(
                    update={"image_url": best_cluster_url}
                )

            cluster_chars = [
                (str(ch), float(conf))
                for ch, conf in cluster.get("chars", [])
            ]
            clusters.append(
                DBCluster(
                    cluster_index=cluster_index,
                    plate_text=cluster.get("plate_text") or cluster.get("plate", ""),
                    chars=cluster_chars,
                    best_plate_frame=best_cluster_frame,
                    track_buffer=cluster_frames,
                    plate_text_confidence=round(
                        float(cluster.get("plate_text_confidence", cluster.get("confidence", 0.0))),
                        4,
                    ),
                    ocr_vote_summary=cluster.get("ocr_vote_summary")
                    or cluster.get("vote_summary")
                    or {},
                    ocr_method=cluster.get("ocr_method", "ocr_output_ctm"),
                    frame_count=int(cluster.get("frame_count", len(cluster_frames))),
                    template=cluster.get("template"),
                )
            )

        record = DBRecord(
            session_id=session_id,
            user_id=user_id,
            track_id=int(tid),
            vehicle_track_id=tracker.vehicle_track_id(tid),
            plate_track_id=tracker.plate_track_id(tid),
            vehicle_class=tracker._cls.get(tid, "vehicle"),
            best_plate_frame=best_plate_frame,
            track_buffer=track_frames,
            vehicle_thumbnail_url=vehicle_url,
            plate_text=plate_text,
            plate_text_confidence=round(plate_conf, 4),
            ocr_vote_summary=vote_summary,
            clusters=clusters,
            ocr_method=ocr_method,
            first_seen_frame=min(f.frame_index for f in track_frames),
            last_seen_frame=max(f.frame_index for f in track_frames),
        )

        asyncio.run_coroutine_threadsafe(upsert_record(record), loop)
        logger.debug("MongoDB: queued save for track %d (plate=%s)", tid, plate_text)

    except Exception:
        logger.exception("MongoDB: failed to save track %d", tid)


def _record_save_later(
    session_id: str,
    tid: int,
    tracker: WebTrackletManager,
    char_probs: list[tuple[str, float]],
    ocr_method: str,
    vote_summary: dict[str, int],
    loop: asyncio.AbstractEventLoop,
    user_id: str | None = None,
) -> None:
    """Snapshot the tracker synchronously (before the track is released), then
    persist evidence off the inference worker using that snapshot."""
    frozen = _FrozenTrackerView(tracker, tid)
    try:
        loop.run_in_executor(
            None,
            _record_save,
            session_id,
            tid,
            frozen,
            char_probs,
            ocr_method,
            vote_summary,
            loop,
            user_id,
        )
    except RuntimeError:
        logger.exception("MongoDB: failed to schedule save for track %d", tid)


# ── Main job ──────────────────────────────────────────────────────────────────


def run_job(
    video_path: str,
    job_id: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    models: ModelBundle,
    jobs: dict,
    filename: str = "video.mp4",
    mjpeg_queue: asyncio.Queue | None = None,
    preprocess_mode: str = "none",
    ocr_backend: str = "default",
    user_id: str | None = None,
    job_owners: dict | None = None,
) -> None:
    """Legacy upload-and-process-whole-video entry point. Thin wrapper around
    the async frame pipeline; owns video file lifecycle + session row."""
    from .frame_source import FileFrameSource
    from .preprocessed_video import (
        RecordingFrameSource,
        build_preprocessed_video_path,
        preprocessed_video_url,
        register_preprocessed_video_artifact,
    )
    from .preprocessing import normalize_preprocess_mode
    from .pipeline_async import _safe_put, process_frames_async as process_frames

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    normalized_mode = "none"
    recorder: RecordingFrameSource | None = None
    _session_create(job_id, filename, loop, user_id, preprocess_mode, ocr_backend)

    try:
        normalized_mode = normalize_preprocess_mode(preprocess_mode)
        raw_source = FileFrameSource(video_path)
        source = raw_source
        if normalized_mode != "none":
            recorder = RecordingFrameSource(
                raw_source,
                build_preprocessed_video_path(job_id),
            )
        timings: dict[str, float] | None = {} if ALPR_DEBUG_TIMINGS else None
        
        if ocr_backend == "vietnamese_yolov5":
            from .pipeline_yolov5_vietnamese import process_frames_yolov5_vietnamese
            summary = process_frames_yolov5_vietnamese(
                source,
                emit=emit,
                models=models,
                session_id=job_id,
                loop=loop,
                mjpeg_queue=mjpeg_queue,
                record_save=_record_save_later,
                timings=timings,
                user_id=user_id,
                preprocess_mode=normalized_mode,
                preprocessed_frame_recorder=recorder,
            )
        else:
            summary = process_frames(
                source,
                emit=emit,
                models=models,
                session_id=job_id,
                loop=loop,
                mjpeg_queue=mjpeg_queue,
                record_save=_record_save_later,
                timings=timings,
                ocr_backend=ocr_backend,
                user_id=user_id,
                preprocess_mode=normalized_mode,
                preprocessed_frame_recorder=recorder,
            )
            
        if timings is not None:
            logger.info(
                "ALPR timings job=%s %s",
                job_id,
                {key: round(value, 4) for key, value in sorted(timings.items())},
            )
        processed_video_url: str | None = None
        if recorder is not None:
            if recorder.available and user_id:
                register_preprocessed_video_artifact(job_id, user_id, recorder.output_path)
                processed_video_url = preprocessed_video_url(job_id)
            elif recorder.error:
                logger.warning(
                    "Preprocessed video artifact unavailable for job=%s: %s",
                    job_id,
                    recorder.error,
                )

        emit({
            "type": "complete",
            "total_vehicles": summary["total_vehicles"],
            "preprocess_mode": normalized_mode,
            "processed_video_url": processed_video_url,
        })
        _session_update(job_id, {
            "status": "completed",
            "total_records": summary["total_vehicles"],
            "processed_frames": summary.get("processed_frames", source.total_frames or 0),
            "preprocess_mode": normalized_mode,
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
            loop.call_soon_threadsafe(_safe_put, mjpeg_queue, None)  # sentinel
        if job_owners is not None:
            job_owners.pop(job_id, None)
