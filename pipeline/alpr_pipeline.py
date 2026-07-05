"""
core/pipeline.py — Video processing job (runs in thread-pool).

SSE event types emitted:
  "progress"  — frame index / total / pct
  "frame"     — base64 JPEG of annotated frame (bounding boxes drawn in OpenCV)
  "vehicle"   — per-vehicle OCR update (plate_b64 + vehicle_b64)
  "complete"  — final summary
  "error"     — exception info
"""

from __future__ import annotations

import asyncio
import base64
import gc
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import (
    FRAME_STRIDE,
    VEHICLE_CLASSES,
)
from .models import ModelBundle, ocr_batch, preprocess_plate
from .tracker import WebTrackletManager
from api.core.association import TrajectoryAssociator
from api.core.cascade_plate import detect_plate_tracks_cascade

# Max width for streamed annotated frames (keeps SSE payload small)
_STREAM_W = 960
_STREAM_QUAL = 75
_VEHICLE_PAD = 16  # padding around vehicle bbox when capturing vehicle crop


# ── Frame annotation ──────────────────────────────────────────────────────────


def _draw_boxes(
    frame: np.ndarray,
    tracked: list[dict],
    tracker: WebTrackletManager,
    active_tids: set[int],
) -> np.ndarray:
    """
    Draw vehicle bounding boxes on a copy of frame.

    Colour coding:
      Yellow / thick  — vehicle currently being sent to OCR this stride
      Green           — vehicle already confirmed (done)
      White / thin    — vehicle tracked but no plate matched yet
    """
    vis = frame.copy()

    for v in tracked:
        tid = v["id"]
        x1, y1, x2, y2 = (int(c) for c in v["box"])

        is_active = tid in active_tids
        is_done = tracker._done.get(tid, False)

        color = (0, 210, 255) if is_active else (0, 220, 60) if is_done else (180, 180, 180)
        thickness = 3 if is_active else 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Label: class + id + current plate text
        plate_text = tracker.display_text(tid)
        cls_name = tracker._cls.get(tid, "vehicle")
        label = f"{cls_name} #{tid}"
        if plate_text:
            label += f"  {plate_text}"

        font, fs, thick_txt = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick_txt)
        ly = max(y1 - 6, th + 6)
        cv2.rectangle(vis, (x1, ly - th - 6), (x1 + tw + 8, ly + 2), color, -1)
        cv2.putText(vis, label, (x1 + 4, ly - 2), font, fs, (0, 0, 0), thick_txt)

    return vis


def _encode_frame(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > _STREAM_W:
        frame = cv2.resize(
            frame,
            (_STREAM_W, int(h * _STREAM_W / w)),
            interpolation=cv2.INTER_AREA,
        )
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_QUAL])
    return base64.b64encode(buf).decode()


# ── Vehicle crop helper ───────────────────────────────────────────────────────


def _crop_vehicle(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (int(c) for c in box)
    x1 = max(0, x1 - _VEHICLE_PAD)
    y1 = max(0, y1 - _VEHICLE_PAD)
    x2 = min(W, x2 + _VEHICLE_PAD)
    y2 = min(H, y2 + _VEHICLE_PAD)
    return frame[y1:y2, x1:x2]


# ── Main job ──────────────────────────────────────────────────────────────────


def run_job(
    video_path: str,
    job_id: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    models: ModelBundle,
    jobs: dict,
    filename: str = "video.mp4",
) -> None:
    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        cap = cv2.VideoCapture(video_path)
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        tracker = WebTrackletManager()
        associator = TrajectoryAssociator(match_frames=5, agreement_ratio=0.6)
        frame_idx = 0
        # Resolved once — absolute path avoids CWD sensitivity in thread pool
        tracker_cfg = str(Path(__file__).resolve().parents[1] / "configs/tracking/botsort.yaml")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ── Vehicle tracking (every frame keeps BotSORT state intact) ─────
            v_res = models.vehicle.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                classes=VEHICLE_CLASSES,
                verbose=False,
            )[0]

            tracked: list[dict] = []
            if v_res.boxes.id is not None:
                boxes = v_res.boxes.xyxy.cpu().numpy().astype(int)
                ids = v_res.boxes.id.cpu().numpy().astype(int)
                clss = v_res.boxes.cls.cpu().numpy().astype(int)
                for box, tid, cid in zip(boxes, ids, clss):
                    tid = int(tid)
                    tracker._cls[tid] = models.vehicle.names[int(cid)]
                    tracked.append({"id": tid, "box": box})

            # ── Progress event ────────────────────────────────────────────────
            if frame_idx % 10 == 0 or frame_idx == total:
                emit(
                    {
                        "type": "progress",
                        "frame": frame_idx,
                        "total": total,
                        "pct": round(frame_idx / total * 100, 1),
                    }
                )

            # ── Skip plate detection on non-stride frames ─────────────────────
            if frame_idx % FRAME_STRIDE != 0:
                continue

            # ── Skip when every visible vehicle is already done ───────────────
            if tracked and not any(tracker.should_ocr(v["id"]) for v in tracked):
                # Still emit an annotated frame so the display updates
                emit(
                    {
                        "type": "frame",
                        "b64": _encode_frame(_draw_boxes(frame, tracked, tracker, set())),
                    }
                )
                continue

            # matched: (vehicle_tid, plate_crop, vehicle_crop)
            matched: list[tuple[int, np.ndarray, np.ndarray]] = []
            plate_tracks = detect_plate_tracks_cascade(
                frame,
                tracked,
                models.plate,
            )
            firm_matches = associator.process_frame(plate_tracks, tracked)
            for v_tid, p in firm_matches:
                v_box = associator.vehicle_cache.get(v_tid)
                if v_box is not None:
                    vehicle_crop = _crop_vehicle(frame, v_box)
                    matched.append((v_tid, p["crop"], vehicle_crop))

            # ── Batch OCR ─────────────────────────────────────────────────────
            active_tids: set[int] = set()
            to_ocr = [(tid, pc, vc) for tid, pc, vc in matched if tracker.should_ocr(tid)]

            if to_ocr:
                active_tids = {tid for tid, _, _ in to_ocr}
                tensors = torch.stack([preprocess_plate(pc) for _, pc, _ in to_ocr]).to(
                    models.device
                )

                for (tid, plate_crop, vehicle_crop), (char_probs, all_conf) in zip(
                    to_ocr, ocr_batch(models.ocr, tensors, models.device)
                ):
                    avg_conf = (
                        sum(p for _, p in char_probs) / len(char_probs) if char_probs else 0.0
                    )
                    tracker.update(tid, char_probs, all_conf)
                    tracker.update_plate_img(tid, plate_crop, char_probs)
                    tracker.update_vehicle_img(tid, vehicle_crop, avg_conf)

                    if tracker.plate_changed(tid):
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
                                "ocr_frames": tracker.ocr_frames(tid),
                            }
                        )

            # ── Annotated frame for live display ──────────────────────────────
            emit(
                {
                    "type": "frame",
                    "b64": _encode_frame(_draw_boxes(frame, tracked, tracker, active_tids)),
                }
            )

            if frame_idx % 90 == 0:
                gc.collect()

        cap.release()

        # ── Final snapshot ────────────────────────────────────────────────────
        vehicles_for_db = []
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
                    "ocr_frames": tracker.ocr_frames(tid),
                    "confidence": float(tracker._vehicle_img_conf.get(tid, 0)),
                    "final": True,
                }
            )

            # # Prepare bytes for DB storage
            # p_img = tracker._plate_img.get(tid)
            # v_img = tracker._vehicle_img.get(tid)

            # p_bytes = cv2.imencode(".jpg", p_img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes() if p_img is not None else None
            # v_bytes = cv2.imencode(".jpg", v_img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() if v_img is not None else None

            # vehicles_for_db.append({
            #     "tid": tid,
            #     "cls": tracker._cls.get(tid, ""),
            #     "plate": tracker.display_text(tid),
            #     "confidence": int(tracker._vehicle_img_conf.get(tid, 0) * 100) if tracker._vehicle_img_conf.get(tid) else None,
            #     "plate_bytes": p_bytes,
            #     "vehicle_bytes": v_bytes
            # })

        # # Save to Supabase (synchronously in this background thread)
        # if vehicles_for_db:
        #     try:
        #         from .database import save_results
        #         save_results(job_id, filename, vehicles_for_db)
        #     except Exception as e:
        #         print(f"[Supabase] Error calling save_results: {e}")

        emit({"type": "complete", "total_vehicles": len(tracker._best)})

    except Exception as exc:
        import traceback

        emit({"type": "error", "message": str(exc), "detail": traceback.format_exc()})

    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass
        jobs.pop(job_id, None)
