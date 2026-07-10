"""pipeline_async — 3-stage pipelined ALPR inference using threading queues.

Architecture (Pipeline Parallelism):
  Stage 1 │ Reader Thread     │  I/O bound — reads frames from FrameSource
           │                   │  → puts (frame_idx, frame, ts) into frame_q
  Stage 2 │ Vehicle Thread    │  GPU bound — YOLO vehicle detect + BoT-SORT track
           │                   │  → puts (frame_idx, frame, tracked) into crop_q
  Stage 3 │ Plate+OCR Thread  │  GPU bound — cascade plate detect + Transformer OCR
           │                   │  → calls emit() with recognition events

Why threading (not multiprocessing)?
  PyTorch releases the GIL during C++/CUDA forward passes, so two GPU-bound
  threads can genuinely run in parallel on separate CUDA streams. Using
  multiprocessing would require serialising CUDA tensors between processes,
  which is both complex and slower.

Drop-in replacement:
  ``process_frames_async`` has the same signature as ``pipeline_core.process_frames``
  so callers (run_benchmark.py, pipeline.py, event_analyzer.py) only need to
  change the import.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import queue
import threading
import time
from typing import Callable

import numpy as np
import torch

from .association import TrajectoryAssociator
from .cascade_plate import detect_plate_tracks_cascade
from .config import (
    ALPR_PREVIEW_FPS,
    ASSOCIATION_AGREEMENT_RATIO,
    ASSOCIATION_MATCH_FRAMES,
    FRAME_STRIDE,
    VEHICLE_CLASSES,
)
from .frame_source import FrameSource
from .models import ModelBundle, ocr_batch, preprocess_plate_for_model, select_ocr_model
from .preprocessing import apply_preprocessing, normalize_preprocess_mode
from .progress import make_progress_event
from .preview_frame import make_preview_frame_event
from .quality_router import PlateQualityRouter
from .route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
from .track_ocr import finalise_track_ocr as _finalise_track_ocr_impl
from .tracker import WebTrackletManager
from .video_processor import crop_vehicle as _crop_vehicle

logger = logging.getLogger(__name__)

# ── Sentinel used as poison-pill to signal thread shutdown ────────────────────
_STOP = object()

# ── Queue size caps — tune to balance throughput vs. peak RAM/VRAM usage ─────
_FRAME_Q_SIZE = 32  # raw BGR frames waiting for vehicle detection
_CROP_Q_SIZE = 16  # (frame, tracked_list) waiting for plate/OCR


def _safe_put(q: asyncio.Queue, item: object) -> None:
    if not q.full():
        q.put_nowait(item)


def _fps_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0 or source_fps <= 0:
        return 0
    return max(1, int(round(source_fps / target_fps)))


# ── Stage-3 helper: identical logic to pipeline_core._finalise_track_ocr ─────


def _finalise_track_ocr(
    tid: int,
    tracker: WebTrackletManager,
    models: ModelBundle,
    emit: Callable[[dict], None],
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    record_save: Callable | None,
    ocr_backend: str,
    user_id: str | None = None,
) -> None:
    _finalise_track_ocr_impl(
        tid, tracker, models, emit, session_id, loop, record_save, ocr_backend=ocr_backend,
        user_id=user_id
    )


# ── Stage 1: Reader ────────────────────────────────────────────────────────────


def _reader_worker(
    source: FrameSource,
    frame_q: queue.Queue,
    stop_event: threading.Event,
    timings: dict[str, float] | None = None,
) -> None:
    """Push (frame_idx, frame, ts) items into frame_q until source is exhausted.

    Measures ``s1_put_stall``: wall time blocked on frame_q.put() — high values
    indicate Stage 2 (vehicle detection) cannot keep up with I/O throughput.
    """
    try:
        for src_idx, frame, ts in source.iter_frames():
            if stop_event.is_set():
                break
            # Measure how long we block when Stage 2 is too slow to drain the queue.
            _t = time.perf_counter()
            frame_q.put((src_idx, frame, ts))  # blocking put — back-pressure
            if timings is not None:
                stall = time.perf_counter() - _t
                timings["s1_put_stall"] = timings.get("s1_put_stall", 0.0) + stall
    except Exception:
        logger.exception("Reader worker crashed")
        stop_event.set()
    finally:
        frame_q.put(_STOP)


# ── Stage 2: Vehicle Detect + Track ───────────────────────────────────────────


def _vehicle_worker(
    frame_q: queue.Queue,
    crop_q: queue.Queue,
    models: ModelBundle,
    vehicle_tracker,
    tracker: WebTrackletManager,
    total_frames: int,
    emit: Callable[[dict], None],
    stop_event: threading.Event,
    timings: dict[str, float] | None,
    preprocess_mode: str,
    preprocessed_frame_recorder: object | None,
) -> None:
    """
    Consume frames from frame_q, run vehicle detection and tracking,
    forward (frame_idx, frame, tracked) to crop_q.

    Timing keys added:
      ``s2_get_stall`` — time waiting for Stage 1 (reader) to produce a frame.
                         High → reader / disk I/O is the bottleneck.
      ``s2_put_stall`` — time blocked pushing to crop_q because Stage 3 is full.
                         High → Stage 3 (plate/OCR) is the bottleneck.

    NOTE: BoT-SORT requires strictly sequential frame order — this must be
    a *single* thread (no parallel workers here).
    """

    def _add_timing(name: str, started_at: float) -> None:
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started_at

    processed_count = 0

    try:
        while not stop_event.is_set():
            _get_t = time.perf_counter()
            try:
                item = frame_q.get(timeout=1.0)
            except queue.Empty:
                continue
            _add_timing("s2_get_stall", _get_t)

            if item is _STOP:
                break

            src_idx, frame, _ts = item
            frame_idx = src_idx + 1  # 1-based, matches pipeline_core convention
            vehicle_frame = (
                frame
                if preprocess_mode == "none"
                else apply_preprocessing(frame, preprocess_mode)
            )
            if preprocessed_frame_recorder is not None:
                try:
                    preprocessed_frame_recorder.record_frame(vehicle_frame)
                except Exception:
                    logger.exception("Preprocessed video recorder failed")

            # ── Vehicle detection ─────────────────────────────────────────────
            stage_start = time.perf_counter()
            v_pred = models.vehicle.predict(vehicle_frame, classes=VEHICLE_CLASSES, verbose=False)[0]
            _add_timing("vehicle_detect", stage_start)

            if v_pred.boxes is not None and len(v_pred.boxes) > 0:
                xyxy = v_pred.boxes.xyxy.cpu().numpy()
                conf = v_pred.boxes.conf.cpu().numpy().reshape(-1, 1)
                cls = v_pred.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
            else:
                dets = np.zeros((0, 6), dtype=np.float32)

            # ── Tracking ──────────────────────────────────────────────────────
            stage_start = time.perf_counter()
            boxes, ids, classes = vehicle_tracker.track(dets, vehicle_frame)
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

            processed_count += 1

            if processed_count % 10 == 0 and (not total_frames or processed_count < total_frames):
                emit(
                    make_progress_event(
                        processed_frames=processed_count,
                        total_frames=total_frames,
                        source_frame=frame_idx,
                    )
                )

            # Forward to Stage 3 — measure stall if crop_q is full (Stage 3 slow)
            _put_t = time.perf_counter()
            crop_q.put((frame_idx, processed_count, frame, vehicle_frame, tracked, currently_tracked))
            _add_timing("s2_put_stall", _put_t)

    except Exception:
        logger.exception("Vehicle worker crashed")
        stop_event.set()
    finally:
        if preprocessed_frame_recorder is not None:
            try:
                preprocessed_frame_recorder.finish()
            except Exception:
                logger.exception("Preprocessed video recorder finalization failed")
        # Poison pill for plate/OCR worker
        crop_q.put(_STOP)


# ── Stage 3: Plate Detect + OCR ───────────────────────────────────────────────


def _plate_ocr_worker(
    crop_q: queue.Queue,
    models: ModelBundle,
    tracker: WebTrackletManager,
    associator: TrajectoryAssociator,
    emit: Callable[[dict], None],
    session_id: str,
    loop: asyncio.AbstractEventLoop | None,
    mjpeg_queue: asyncio.Queue | None,
    record_save: Callable | None,
    stop_event: threading.Event,
    timings: dict[str, float] | None,
    frame_count_out: list[int],
    preview_stride: int,
    ocr_backend: str,
    user_id: str | None,
) -> None:
    """
    Consume (frame_idx, frame, tracked) from crop_q.
    Run cascade plate detection and Transformer OCR.
    Respects FRAME_STRIDE — skips OCR on non-stride frames.
    """

    def _add_timing(name: str, started_at: float) -> None:
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started_at

    previously_tracked: set[int] = set()
    model_router = getattr(models, "quality_router", None)
    quality_router = (
        model_router if isinstance(model_router, PlateQualityRouter) else PlateQualityRouter()
    )
    preview_seen = 0

    try:
        while not stop_event.is_set():
            _get_t = time.perf_counter()
            try:
                item = crop_q.get(timeout=1.0)
            except queue.Empty:
                continue
            _add_timing("s3_get_stall", _get_t)

            if item is _STOP:
                break

            frame_idx, processed_count, frame, vehicle_frame, tracked, currently_tracked = item
            frame_count_out[0] = processed_count

            # ── Handle lost tracks (tracks present before but missing now) ────
            # Only finalise on stride frames to keep timing consistent.
            if frame_idx % FRAME_STRIDE == 0:
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

            # Skip plate detection on non-stride frames
            if frame_idx % FRAME_STRIDE != 0:
                previously_tracked = currently_tracked
                continue

            # ── Cascade plate detection ───────────────────────────────────────
            active_tids: set[int] = set()
            tracked_for_ocr = [v for v in tracked if tracker.should_ocr(int(v["id"]))]
            plate_tracks = detect_plate_tracks_cascade(
                frame, tracked_for_ocr, models.plate, timings=timings
            )

            stage_start = time.perf_counter()
            firm_matches = associator.process_frame(plate_tracks, tracked_for_ocr)
            _add_timing("association", stage_start)

            matched: list[tuple[int, np.ndarray, np.ndarray]] = []
            for v_tid, p in firm_matches:
                v_box = associator.vehicle_cache.get(v_tid)
                if v_box is not None:
                    vehicle_crop = _crop_vehicle(frame, v_box)
                    matched.append((v_tid, p["crop"], vehicle_crop))

            # ── Batch OCR ─────────────────────────────────────────────────────
            stage_start = time.perf_counter()
            ocr_jobs, active_tids = prepare_route_ocr_jobs(
                matched,
                tracker,
                quality_router,
                frame_idx,
            )
            _add_timing("classify", stage_start)
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

            # ── SSE preview frame ────────────────────────────────────────────
            if preview_stride > 0:
                preview_seen += 1
            if (
                preview_stride > 0
                and preview_seen % preview_stride == 0
            ):
                box_dicts = [
                    {
                        "id": v["id"],
                        "kind": "vehicle",
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
                if box_dicts:
                    emit(make_preview_frame_event(frame, box_dicts, frame_index=frame_idx))
                    if vehicle_frame is not frame:
                        preprocessed_event = make_preview_frame_event(
                            vehicle_frame,
                            box_dicts,
                            frame_index=frame_idx,
                        )
                        preprocessed_event["type"] = "preprocessed_frame"
                        emit(preprocessed_event)

            previously_tracked = currently_tracked

            if frame_idx % 90 == 0:
                gc.collect()

    except Exception:
        logger.exception("Plate/OCR worker crashed")
        stop_event.set()


# ── Public entry point ─────────────────────────────────────────────────────────


def process_frames_async(
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
    emit_preview: bool = True,
    preprocess_mode: str = "none",
    preprocessed_frame_recorder: object | None = None,
) -> dict:
    """Run the full ALPR pipeline asynchronously using 3 pipeline-parallel threads.

    Drop-in replacement for ``pipeline_core.process_frames``.
    Returns: {total_vehicles, processed_frames}.
    """
    total_start = time.perf_counter()
    total_frames = source.total_frames or 0
    normalized_preprocess_mode = normalize_preprocess_mode(preprocess_mode)

    # Shared state
    tracker = WebTrackletManager()
    associator = TrajectoryAssociator(
        match_frames=ASSOCIATION_MATCH_FRAMES,
        agreement_ratio=ASSOCIATION_AGREEMENT_RATIO,
    )
    vehicle_tracker = models.create_vehicle_tracker()
    vehicle_tracker.reset()  # reset boxmot's process-global track-id counter per session

    stop_event = threading.Event()
    frame_q: queue.Queue = queue.Queue(maxsize=_FRAME_Q_SIZE)
    crop_q: queue.Queue = queue.Queue(maxsize=_CROP_Q_SIZE)

    # Mutable counter so Stage-3 thread can report processed frame count back
    frame_count_out: list[int] = [0]
    preview_stride = _fps_stride(source.fps, ALPR_PREVIEW_FPS) if emit_preview else 0

    # ── Spin up threads ────────────────────────────────────────────────────────
    t_reader = threading.Thread(
        target=_reader_worker,
        args=(source, frame_q, stop_event, timings),
        name="alpr-reader",
        daemon=True,
    )
    t_vehicle = threading.Thread(
        target=_vehicle_worker,
        args=(
            frame_q,
            crop_q,
            models,
            vehicle_tracker,
            tracker,
            total_frames,
            emit,
            stop_event,
            timings,
            normalized_preprocess_mode,
            preprocessed_frame_recorder if normalized_preprocess_mode != "none" else None,
        ),
        name="alpr-vehicle",
        daemon=True,
    )
    t_plate = threading.Thread(
        target=_plate_ocr_worker,
        args=(
            crop_q,
            models,
            tracker,
            associator,
            emit,
            session_id,
            loop,
            mjpeg_queue,
            record_save,
            stop_event,
            timings,
            frame_count_out,
            preview_stride,
            ocr_backend,
            user_id,
        ),
        name="alpr-plate-ocr",
        daemon=True,
    )

    t_reader.start()
    t_vehicle.start()
    t_plate.start()

    # ── Wait for pipeline to drain ────────────────────────────────────────────
    t_reader.join()
    t_vehicle.join()
    t_plate.join()

    # ── Finalise any remaining buffered tracks ────────────────────────────────
    if frame_count_out[0]:
        emit(
            make_progress_event(
                processed_frames=frame_count_out[0],
                total_frames=total_frames,
                complete=True,
            )
        )

    for tid in list(tracker._buffers):
        if tracker.should_ocr(tid) and tracker.ready_for_track_ocr(tid):
            _finalise_track_ocr(
                tid, tracker, models, emit, session_id, loop, record_save, ocr_backend,
                user_id
            )

    # ── Final snapshot ────────────────────────────────────────────────────────
    for tid in sorted(tracker._best):
        event: dict = {
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
        cluster_data = tracker.cluster_results(tid)
        if len(cluster_data) > 1:
            event["clusters"] = cluster_data
        emit(event)
        tracker.release_track(tid, recognized=True)

    if timings is not None:
        timings["total"] = time.perf_counter() - total_start

    return {
        "total_vehicles": tracker.recognized_vehicle_count(),
        "processed_frames": frame_count_out[0],
    }
