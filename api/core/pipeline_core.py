"""pipeline_core — pure inference loop shared by run_job and event_analyzer.

Runs detect → track → buffer → OCR → vote on any FrameSource. Does NOT open
files, NOT touch MongoDB session documents, NOT delete temp files. Those
responsibilities live with the caller (run_job for the upload flow, or
event_analyzer for marks).
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Callable

import numpy as np
import torch

from .association import TrajectoryAssociator
from .cascade_plate import PlateTrackManager, detect_plate_tracks_cascade
from .config import (
    ALPR_PREVIEW_FPS,
    ASSOCIATION_AGREEMENT_RATIO,
    ASSOCIATION_MATCH_FRAMES,
    FRAME_STRIDE,
    VEHICLE_CLASSES,
)
from .frame_source import FrameSource
from .models import ModelBundle, ocr_batch, preprocess_plate_for_model, select_ocr_model
from .progress import make_progress_event
from .quality_router import PlateQualityRouter
from .route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
from .track_ocr import finalise_track_ocr as _finalise_track_ocr_impl
from .tracker import WebTrackletManager
from .video_processor import (
    crop_vehicle as _crop_vehicle,
    draw_annotated_frame as _draw_annotated_frame,
)

logger = logging.getLogger(__name__)


def _safe_put(q: asyncio.Queue, item: object) -> None:
    if not q.full():
        q.put_nowait(item)


def _fps_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0 or source_fps <= 0:
        return 0
    return max(1, int(round(source_fps / target_fps)))


def _finalise_track_ocr(
    tid: int,
    tracker: WebTrackletManager,
    models: ModelBundle,
    emit: Callable[[dict], None],
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    record_save: Callable | None,
    ocr_backend: str = "default",
    user_id: str | None = None,
) -> None:
    _finalise_track_ocr_impl(
        tid,
        tracker,
        models,
        emit,
        session_id,
        loop,
        record_save,
        ocr_backend=ocr_backend,
        user_id=user_id,
    )


def process_frames(
    source: FrameSource,
    emit: Callable[[dict], None],
    models: ModelBundle,
    *,
    session_id: str = "",
    loop: asyncio.AbstractEventLoop | None = None,
    mjpeg_queue: asyncio.Queue | None = None,
    record_save: Callable | None = None,
    timings: dict[str, float] | None = None,
    ocr_backend: str = "default",
    user_id: str | None = None,
) -> dict:
    """Run the full ALPR pipeline on a FrameSource.

    Returns: {total_vehicles, processed_frames}.
    """
    total_start = time.perf_counter()

    def _add_timing(name: str, started_at: float) -> None:
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started_at

    def emit_frame(jpg: bytes) -> None:
        if mjpeg_queue is not None and loop is not None:
            loop.call_soon_threadsafe(_safe_put, mjpeg_queue, jpg)

    tracker = WebTrackletManager()
    associator = TrajectoryAssociator(
        match_frames=ASSOCIATION_MATCH_FRAMES,
        agreement_ratio=ASSOCIATION_AGREEMENT_RATIO,
    )
    plate_tracker = PlateTrackManager()
    model_router = getattr(models, "quality_router", None)
    quality_router = (
        model_router if isinstance(model_router, PlateQualityRouter) else PlateQualityRouter()
    )
    vehicle_tracker = models.create_vehicle_tracker()

    total = source.total_frames or 0
    previously_tracked: set[int] = set()
    frame_idx = 0
    processed_seen = 0
    preview_seen = 0
    preview_stride = _fps_stride(source.fps, ALPR_PREVIEW_FPS)

    for src_idx, frame, _ts in source.iter_frames():
        frame_idx = src_idx + 1  # 1-based to match legacy run_job
        processed_seen += 1

        stage_start = time.perf_counter()
        v_pred = models.vehicle.predict(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        _add_timing("vehicle_detect", stage_start)
        if v_pred.boxes is not None and len(v_pred.boxes) > 0:
            xyxy = v_pred.boxes.xyxy.cpu().numpy()
            conf = v_pred.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = v_pred.boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)

        stage_start = time.perf_counter()
        boxes, ids, classes = vehicle_tracker.track(dets, frame)
        _add_timing("vehicle_track", stage_start)

        tracked: list[dict] = []
        currently_tracked: set[int] = set()
        for box, tid, cid in zip(boxes, ids, classes):
            tid = int(tid)
            tracker._cls[tid] = models.vehicle.names[int(cid)]
            tracked.append({"id": tid, "box": box.tolist()})
            currently_tracked.add(tid)
            if tid in tracker._lost_count:
                tracker.reset_lost(tid)

        if processed_seen % 10 == 0 and (not total or processed_seen < total):
            emit(
                make_progress_event(
                    processed_frames=processed_seen,
                    total_frames=total,
                    source_frame=frame_idx,
                )
            )

        if frame_idx % FRAME_STRIDE != 0:
            previously_tracked = currently_tracked
            continue

        for tid in previously_tracked - currently_tracked:
            if (
                tracker.should_ocr(tid)
                and tracker.mark_lost(tid)
                and tracker.ready_for_track_ocr(tid)
            ):
                _finalise_track_ocr(
                    tid,
                    tracker,
                    models,
                    emit,
                    session_id,
                    loop,
                    record_save,
                    ocr_backend,
                    user_id,
                )

        active_tids: set[int] = set()
        matched: list[tuple[int, np.ndarray, np.ndarray]] = []
        tracked_for_ocr = [v for v in tracked if tracker.should_ocr(int(v["id"]))]
        plate_tracks = detect_plate_tracks_cascade(
            frame,
            tracked_for_ocr,
            models.plate,
            plate_tracker,
            timings=timings,
        )

        stage_start = time.perf_counter()
        firm_matches = associator.process_frame(plate_tracks, tracked_for_ocr)
        _add_timing("association", stage_start)
        for v_tid, p in firm_matches:
            v_box = associator.vehicle_cache.get(v_tid)
            if v_box is not None:
                vehicle_crop = _crop_vehicle(frame, v_box)
                matched.append((v_tid, p["crop"], vehicle_crop))

        ocr_jobs, active_tids = prepare_route_ocr_jobs(
            matched,
            tracker,
            quality_router,
            frame_idx,
        )

        if ocr_jobs:
            target_ocr_model = select_ocr_model(models, ocr_backend)
            _tensors = torch.stack(
                [
                    preprocess_plate_for_model(target_ocr_model, job.candidate_crop)
                    for job in ocr_jobs
                ]
            ).to(models.device)
            stage_start = time.perf_counter()
            _ocr_results = ocr_batch(target_ocr_model, _tensors, models.device)
            _add_timing("ocr", stage_start)
            consume_route_ocr_results(
                ocr_jobs,
                _ocr_results,
                tracker,
                emit,
                session_id=session_id,
                loop=loop,
                record_save=record_save,
                user_id=user_id,
            )

        if mjpeg_queue is not None and preview_stride > 0:
            preview_seen += 1
        if mjpeg_queue is not None and preview_stride > 0 and preview_seen % preview_stride == 0:
            box_dicts = [
                {
                    "id": v["id"],
                    "box": [int(c) for c in v["box"]],
                    "state": (
                        "active"
                        if v["id"] in active_tids
                        else "done" if tracker._done.get(v["id"]) else "tracked"
                    ),
                    "plate": tracker.display_text(v["id"]) or "",
                    "cls": tracker._cls.get(v["id"], "vehicle"),
                }
                for v in tracked
            ]
            emit_frame(_draw_annotated_frame(frame, box_dicts))

        previously_tracked = currently_tracked

        if frame_idx % 90 == 0:
            gc.collect()

    # ── Finalise remaining buffered tracks ────────────────────────────────────
    if processed_seen:
        emit(
            make_progress_event(
                processed_frames=processed_seen,
                total_frames=total,
                source_frame=frame_idx,
                complete=True,
            )
        )

    for tid in list(tracker._buffers):
        if tracker.should_ocr(tid) and tracker.ready_for_track_ocr(tid):
            _finalise_track_ocr(
                tid,
                tracker,
                models,
                emit,
                session_id,
                loop,
                record_save,
                ocr_backend,
                user_id,
            )

    # ── Final snapshot ────────────────────────────────────────────────────────
    for tid in sorted(tracker._best):
        emit(
            {
                "type": "vehicle",
                "id": tid,
                "cls": tracker._cls.get(tid, ""),
                "plate": tracker.display_text(tid),
                "chars": tracker.chars_json(tid),
                "done": tracker._done.get(tid, False),
                "plate_b64": tracker.plate_b64(tid),
                "vehicle_b64": tracker.vehicle_b64(tid),
                "track_buffer": tracker.track_buffer_json(tid),
                "ocr_frames": tracker.ocr_frames(tid),
                "confidence": float(tracker._plate_img_conf.get(tid, 0)),
                "final": True,
            }
        )

    if timings is not None:
        timings["total"] = time.perf_counter() - total_start

    return {
        "total_vehicles": len(tracker._best),
        "processed_frames": processed_seen,
    }
