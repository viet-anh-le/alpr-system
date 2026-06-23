"""incident_analyzer — orchestrates a single mark→analysis job.

Wraps the async ALPR pipeline, translates its event types into
incident_* events so the SSE consumer can route them to the right card,
and persists the result to the `incidents` MongoDB collection.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from datetime import datetime, timezone
from typing import Literal

import cv2

from .config import ALPR_DEBUG_TIMINGS
from .frame_source import FrameSource
from .models import ModelBundle
from .pipeline_async import process_frames_async as process_frames

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
                plate_url = core_db.upload_image(
                    "evidence",
                    f"incidents/{incident_id}/plate_{v['id']}.jpg",
                    base64.b64decode(v["plate_b64"]),
                )
            if v.get("vehicle_b64"):
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
                vehicle_track_id=v.get("vehicle_track_id"),
                plate_track_id=v.get("plate_track_id"),
                plate_text=v["plate"],
                plate_text_confidence=round(conf, 4),
                chars=[(c, float(p)) for c, p in v.get("chars", [])],
                vehicle_class=v.get("cls", "vehicle"),
                plate_image_url=plate_url,
                vehicle_image_url=vehicle_url,
                ocr_method=v.get("ocr_method", "segment_vote"),
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
        fut = asyncio.run_coroutine_threadsafe(mongodb.upsert_incident(incident), loop)
        fut.add_done_callback(
            lambda f: logger.error("upsert_incident failed for %s: %s", incident_id, f.exception())
            if f.exception() else None
        )
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
    ocr_backend: str = "default",
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
        timings: dict[str, float] | None = {} if ALPR_DEBUG_TIMINGS else None
        if ocr_backend == "vietnamese_yolov5":
            from .pipeline_yolov5_vietnamese import process_frames_yolov5_vietnamese
            summary = process_frames_yolov5_vietnamese(
                source,
                emit=emit_translated,
                models=models,
                session_id="",
                loop=None,
                timings=timings,
            )
        else:
            summary = process_frames(
                source,
                emit=emit_translated,
                models=models,
                session_id="",   # we persist via _persist_incident, not _record_save
                loop=None,
                timings=timings,
                ocr_backend=ocr_backend,
            )
        dur_ms = int((time.monotonic() - started) * 1000)
        if timings is not None:
            logger.info(
                "Incident timings incident=%s source=%s %s",
                incident_id,
                source_type,
                {key: round(value, 4) for key, value in sorted(timings.items())},
            )
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
