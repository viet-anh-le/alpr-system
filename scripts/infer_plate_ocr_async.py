"""Run async ALPR inference with plate detection + OCR only (no vehicle model).

Architecture:
  Stage 1 │ Reader thread  │ reads frames from video
  Stage 2 │ Plate+OCR thread │ full-frame plate detect → track → OCR

Usage:
    /home/vietanh/anaconda3/envs/myenv/bin/python scripts/infer_plate_ocr_async.py \\
        data/realworld-videos/chunks/hcm_night_01.mp4 \\
        --output data/outputs/hcm_night_01_plates.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.core import cascade_plate  # noqa: E402
from api.core.cascade_plate import (  # noqa: E402
    VehicleCrop,
    _extract_obb_candidates,
)
from api.core.config import (  # noqa: E402
    FRAME_STRIDE,
    OCR_BACKEND,
    PLATE_MODEL_PATH,
    SMALL_LPR_LINE_CTC_CKPT_PATH,
)
from api.core.frame_source import FileFrameSource  # noqa: E402
from api.core.models import (  # noqa: E402
    ModelBundle,
    load_small_lpr_line_ctc_model,
    normalize_ocr_backend,
    ocr_batch,
    preprocess_plate_for_model,
    select_ocr_model,
)
from api.core.plate_format import is_vn_plate_text  # noqa: E402
from api.core.progress import make_progress_event  # noqa: E402
from api.core.quality_router import DegradationTags, PlateQualityResult  # noqa: E402
from api.core.route_ocr import (  # noqa: E402
    consume_route_ocr_results,
    prepare_route_ocr_jobs,
)
from api.core.track_ocr import finalise_track_ocr  # noqa: E402
from api.core.tracker import WebTrackletManager  # noqa: E402

logger = logging.getLogger(__name__)

_STOP = object()
_FRAME_Q_SIZE = 32
_PLATE_DET_CONF = 0.25


class _DirectQualityRouter:
    """Force per-frame OCR for plate-only inference (night video friendly)."""

    def route(self, crop_bgr) -> PlateQualityResult:
        return PlateQualityResult(
            legibility="good",
            quality_bin="suitable",
            router_conf=1.0,
            tags=DegradationTags(),
            route="direct",
            quality_numeric=0.8,
        )


def load_plate_ocr_models() -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plate = YOLO(str(PLATE_MODEL_PATH))

    ocr_backend = normalize_ocr_backend(OCR_BACKEND)
    if ocr_backend == "vietnamese_yolov5":
        raise ValueError(
            "scripts/infer_plate_ocr_async.py supports only SmallLPR-Line-CTC; "
            "use the web/API pipeline for YOLOv5 Vietnamese."
        )
    ocr = load_small_lpr_line_ctc_model(SMALL_LPR_LINE_CTC_CKPT_PATH, device=device)

    return ModelBundle(
        device=device,
        vehicle=SimpleNamespace(names={0: "scene"}),
        plate=plate,
        ocr=ocr,
        reid_weights=Path(""),
        tracker_device=str(device),
        quality_router=_DirectQualityRouter(),
        ocr_backend=ocr_backend,
    )


def _reader_worker(source: FileFrameSource, frame_q: queue.Queue, stop_event: threading.Event) -> None:
    try:
        for src_idx, frame, ts in source.iter_frames():
            if stop_event.is_set():
                break
            frame_q.put((src_idx, frame, ts))
    except Exception:
        logger.exception("Reader worker crashed")
        stop_event.set()
    finally:
        frame_q.put(_STOP)


def _assign_plate_only_candidate_ids(candidates: list[dict], frame_idx: int) -> list[dict]:
    return [
        {**candidate, "id": frame_idx * 1000 + index}
        for index, candidate in enumerate(candidates, start=1)
    ]


def _plate_ocr_worker(
    frame_q: queue.Queue,
    models: ModelBundle,
    tracker: WebTrackletManager,
    emit,
    stop_event: threading.Event,
    frame_count_out: list[int],
    total_frames: int,
    ocr_backend: str,
) -> None:
    router = models.quality_router
    use_half = torch.cuda.is_available()
    old_conf = cascade_plate.PLATE_DET_CONF
    cascade_plate.PLATE_DET_CONF = _PLATE_DET_CONF

    try:
        while not stop_event.is_set():
            try:
                item = frame_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _STOP:
                break

            src_idx, frame, _ts = item
            frame_idx = src_idx + 1
            frame_count_out[0] = frame_idx

            if frame_idx % FRAME_STRIDE != 0:
                continue

            height, width = frame.shape[:2]
            full_crop = VehicleCrop(
                vehicle_id=-1,
                vehicle_box=(0, 0, width, height),
                crop_box=(0, 0, width, height),
                offset=(0, 0),
                image=frame,
            )
            with torch.inference_mode():
                result = models.plate.predict(
                    frame, conf=_PLATE_DET_CONF, verbose=False, half=use_half
                )[0]
            candidates = _extract_obb_candidates(result, full_crop, frame)
            plate_tracks = _assign_plate_only_candidate_ids(candidates, frame_idx)

            matched: list[tuple[int, object, object]] = []
            seen: set[int] = set()
            for plate_track in plate_tracks:
                plate_tid = int(plate_track["id"])
                if plate_tid in seen:
                    continue
                seen.add(plate_tid)
                tracker._cls[plate_tid] = "plate"
                matched.append((plate_tid, plate_track["crop"], frame))

            jobs, active_tids = prepare_route_ocr_jobs(matched, tracker, router, frame_idx)
            if jobs:
                target_ocr = select_ocr_model(models, ocr_backend)
                tensors = torch.stack(
                    [preprocess_plate_for_model(target_ocr, job.candidate_crop) for job in jobs]
                ).to(models.device)
                ocr_results = ocr_batch(target_ocr, tensors, models.device)
                consume_route_ocr_results(
                    jobs,
                    ocr_results,
                    tracker,
                    emit,
                )

            for tid in active_tids:
                tracker.reset_lost(tid)
            for tid in list(tracker._buffers):
                if tid in active_tids or not tracker.should_ocr(tid):
                    continue
                if tracker.mark_lost(tid) and tracker.ready_for_track_ocr(tid):
                    finalise_track_ocr(
                        tid, tracker, models, emit, "", None, None, ocr_backend=ocr_backend
                    )

            if frame_idx % 10 == 0 and (not total_frames or frame_idx < total_frames):
                emit(
                    make_progress_event(
                        processed_frames=frame_idx,
                        total_frames=total_frames,
                        source_frame=frame_idx,
                    )
                )
    except Exception:
        logger.exception("Plate/OCR worker crashed")
        stop_event.set()
    finally:
        cascade_plate.PLATE_DET_CONF = old_conf


def process_plate_ocr_async(
    source: FileFrameSource,
    emit,
    models: ModelBundle,
    *,
    ocr_backend: str = "default",
) -> dict:
    tracker = WebTrackletManager()
    stop_event = threading.Event()
    frame_q: queue.Queue = queue.Queue(maxsize=_FRAME_Q_SIZE)
    frame_count_out = [0]
    total_frames = source.total_frames or 0

    t_reader = threading.Thread(
        target=_reader_worker,
        args=(source, frame_q, stop_event),
        name="plate-reader",
        daemon=True,
    )
    t_plate = threading.Thread(
        target=_plate_ocr_worker,
        args=(
            frame_q,
            models,
            tracker,
            emit,
            stop_event,
            frame_count_out,
            total_frames,
            ocr_backend,
        ),
        name="plate-ocr",
        daemon=True,
    )
    t_reader.start()
    t_plate.start()
    t_reader.join()
    t_plate.join()

    for tid in list(tracker._buffers):
        if tracker.should_ocr(tid) and tracker.ready_for_track_ocr(tid):
            finalise_track_ocr(tid, tracker, models, emit, "", None, None, ocr_backend=ocr_backend)

    final_rows: list[dict] = []
    for tid in sorted(tracker._best):
        plate = tracker.display_text(tid)
        if not plate:
            continue
        final_rows.append(
            {
                "track_id": tid,
                "plate": plate,
                "confidence": float(tracker._plate_img_conf.get(tid, 0)),
                "ocr_frames": tracker.ocr_frames(tid),
                "valid_format": is_vn_plate_text(plate),
                "final": True,
            }
        )
        emit(
            {
                "type": "vehicle",
                **tracker.identity_fields(tid),
                "cls": tracker._cls.get(tid, "plate"),
                "plate": plate,
                "done": tracker._done.get(tid, False),
                "confidence": float(tracker._plate_img_conf.get(tid, 0)),
                "ocr_frames": tracker.ocr_frames(tid),
                "final": True,
            }
        )

    return {
        "processed_frames": frame_count_out[0],
        "total_plate_tracks": len(final_rows),
        "final_plates": final_rows,
    }


def _normalize_plate_key(plate: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", plate.upper())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plate detect + OCR async inference on a video.")
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "outputs" / "plate_ocr_results.txt",
        help="Output results file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    source = FileFrameSource(video_path)
    models = load_plate_ocr_models()
    latest_frame: int | None = None
    live_events: list[dict] = []

    def emit(event: dict) -> None:
        nonlocal latest_frame
        if event.get("type") == "progress":
            frame = event.get("frame")
            if frame is not None:
                latest_frame = int(frame)
            return
        if event.get("type") in {"vehicle", "rejected_vehicle"} and event.get("plate"):
            row = {
                "event_type": event.get("type"),
                "track_id": event.get("id"),
                "plate": event.get("plate"),
                "confidence": event.get("confidence"),
                "ocr_frames": event.get("ocr_frames"),
                "approx_frame": latest_frame,
                "approx_time_sec": (
                    round(max(latest_frame - 1, 0) / source.fps, 3)
                    if latest_frame is not None and source.fps > 0
                    else None
                ),
                "final": event.get("final"),
            }
            live_events.append(row)
            if event.get("type") == "vehicle":
                print(
                    f"frame~{latest_frame} | track={row['track_id']} | "
                    f"{row['plate']} | conf={row['confidence']}"
                )

    print(f"Running plate+OCR async pipeline on {video_path}")
    started = time.perf_counter()
    summary = process_plate_ocr_async(source=source, emit=emit, models=models)
    elapsed = time.perf_counter() - started

    # Prefer final per-track results; fall back to best live vehicle event per track.
    by_track: dict[int, dict] = {}
    for row in live_events:
        if row["event_type"] != "vehicle":
            continue
        tid = int(row["track_id"])
        prev = by_track.get(tid)
        if prev is None or row.get("final") or (row.get("confidence") or 0) > (prev.get("confidence") or 0):
            by_track[tid] = row

    final_plates = summary.get("final_plates", [])
    if final_plates:
        accepted = [row for row in final_plates if row.get("valid_format")]
        rejected = [row for row in final_plates if not row.get("valid_format")]
    else:
        accepted = [row for row in by_track.values() if is_vn_plate_text(str(row.get("plate", "")))]
        rejected = [row for row in by_track.values() if not is_vn_plate_text(str(row.get("plate", "")))]

    # Collapse near-duplicate strings (OCR jitter across tracks).
    collapsed: dict[str, dict] = {}
    for row in sorted(accepted, key=lambda r: (r.get("confidence") or 0), reverse=True):
        key = _normalize_plate_key(str(row["plate"]))
        if not key or key in collapsed:
            continue
        collapsed[key] = row
    unique_accepted = list(collapsed.values())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Video: {video_path.name}",
        "Pipeline: async (plate detection + OCR only, full-frame)",
        f"OCR backend: {models.ocr_backend}",
        f"Plate detect conf: {_PLATE_DET_CONF}",
        f"Processed at: {datetime.now().isoformat(timespec='seconds')}",
        f"Processed frames: {summary.get('processed_frames', 0)}",
        f"Wall time (s): {elapsed:.2f}",
        f"Accepted plates (unique): {len(unique_accepted)}",
        f"Rejected / invalid format: {len(rejected)}",
        "",
        "Accepted plates:",
    ]
    if not unique_accepted:
        lines.append("  (none)")
    else:
        for i, row in enumerate(unique_accepted, start=1):
            lines.append(
                f"  {i}. {row['plate']} | track={row.get('track_id', '')} | "
                f"conf={row.get('confidence', '')} | ocr_frames={row.get('ocr_frames', '')}"
            )

    if rejected:
        lines.extend(["", "Rejected / low-confidence plates:"])
        for i, row in enumerate(rejected[:20], start=1):
            lines.append(f"  {i}. {row.get('plate', '')} | track={row.get('track_id', '')}")

    lines.extend(
        [
            "",
            "JSON:",
            json.dumps(
                {
                    "video": video_path.name,
                    "accepted": unique_accepted,
                    "rejected": rejected,
                    "all_vehicle_events": by_track,
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved results to {output_path}")
    print(f"Accepted unique plates: {len(unique_accepted)}")


if __name__ == "__main__":
    main()
