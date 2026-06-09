"""Vehicle-first cascade plate detection helpers.

The pipeline still associates plates to vehicles in global frame space.  This
module only changes where plate inference runs: on tracked vehicle crops instead
of the full frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np
import torch

from .config import (
    CASCADE_PLATE_TRACK_BUFFER,
    CASCADE_PLATE_TRACK_IOU,
    CASCADE_VEHICLE_PAD_MIN,
    CASCADE_VEHICLE_PAD_RATIO,
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
)
from .gates import is_router_candidate, is_sharp
from .video_processor import warp_plate_crop


@dataclass(frozen=True)
class VehicleCrop:
    vehicle_id: int
    vehicle_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    offset: tuple[int, int]
    image: np.ndarray


def expand_vehicle_box(
    frame_shape: tuple[int, int, int] | tuple[int, int],
    box: list[int] | tuple[int, int, int, int] | np.ndarray,
    *,
    pad_ratio: float = CASCADE_VEHICLE_PAD_RATIO,
    pad_min: int = CASCADE_VEHICLE_PAD_MIN,
) -> tuple[int, int, int, int]:
    """Expand a vehicle box and clamp it to the frame."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    pad = max(int(round(max(box_w, box_h) * pad_ratio)), pad_min)

    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def crop_vehicle_regions(
    frame: np.ndarray,
    tracked: list[dict],
    *,
    pad_ratio: float = CASCADE_VEHICLE_PAD_RATIO,
    pad_min: int = CASCADE_VEHICLE_PAD_MIN,
) -> list[VehicleCrop]:
    """Build valid vehicle crops from global tracked vehicle boxes."""
    crops: list[VehicleCrop] = []
    for vehicle in tracked:
        crop_box = expand_vehicle_box(
            frame.shape,
            vehicle["box"],
            pad_ratio=pad_ratio,
            pad_min=pad_min,
        )
        x1, y1, x2, y2 = crop_box
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(
            VehicleCrop(
                vehicle_id=int(vehicle["id"]),
                vehicle_box=tuple(int(v) for v in vehicle["box"]),
                crop_box=crop_box,
                offset=(x1, y1),
                image=crop,
            )
        )
    return crops


def map_crop_points_to_global(
    points: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    """Map OBB points from crop-local coordinates back to global frame space."""
    ox, oy = offset
    translated = points.astype(np.float32).copy()
    translated[:, 0] += ox
    translated[:, 1] += oy
    return translated


def _box_area(box: list[int] | tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _box_iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = _box_area((ix1, iy1, ix2, iy2))
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _box_center(box: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _smallest_containing_vehicle_id(
    box: list[int] | tuple[int, int, int, int],
    tracked: list[dict],
) -> int | None:
    cx, cy = _box_center(box)
    best_id: int | None = None
    best_area = float("inf")
    for vehicle in tracked:
        x1, y1, x2, y2 = (int(v) for v in vehicle["box"])
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = _box_area((x1, y1, x2, y2))
            if area < best_area:
                best_area = area
                best_id = int(vehicle["id"])
    return best_id


def deduplicate_plate_candidates(
    candidates: list[dict],
    tracked: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Collapse duplicate plate detections produced by overlapping vehicle crops."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            _smallest_containing_vehicle_id(candidate["box"], tracked) != candidate.get("source_vehicle_id"),
            -float(candidate.get("conf", 0.0)),
        ),
    )
    deduped: list[dict] = []
    for candidate in ordered:
        if any(_box_iou(candidate["box"], kept["box"]) >= iou_threshold for kept in deduped):
            continue
        owner_id = _smallest_containing_vehicle_id(candidate["box"], tracked)
        if owner_id is not None:
            candidate = {**candidate, "source_vehicle_id": owner_id}
        deduped.append(candidate)
    return deduped


class PlateTrackManager:
    """Assign stable global IDs to cascade plate detections."""

    def __init__(
        self,
        *,
        iou_threshold: float = CASCADE_PLATE_TRACK_IOU,
        lost_buffer: int = CASCADE_PLATE_TRACK_BUFFER,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.lost_buffer = lost_buffer
        self._next_id = 1
        self._tracks: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(self, candidates: list[dict]) -> list[dict]:
        matched_track_ids: set[int] = set()
        updated: list[dict] = []

        for candidate in sorted(candidates, key=lambda c: float(c.get("conf", 0.0)), reverse=True):
            best_tid: int | None = None
            best_score = 0.0
            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                score = _box_iou(candidate["box"], track["box"])
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is None or best_score < self.iou_threshold:
                best_tid = self._next_id
                self._next_id += 1

            matched_track_ids.add(best_tid)
            self._tracks[best_tid] = {
                "box": list(candidate["box"]),
                "lost": 0,
            }
            updated.append({**candidate, "id": best_tid})

        for tid in list(self._tracks):
            if tid in matched_track_ids:
                continue
            self._tracks[tid]["lost"] += 1
            if self._tracks[tid]["lost"] > self.lost_buffer:
                del self._tracks[tid]

        return updated


def _extract_obb_candidates(
    result: Any,
    vehicle_crop: VehicleCrop,
    frame: np.ndarray,
) -> list[dict]:
    if result.obb is None:
        return []

    obb = result.obb
    if obb.xyxyxyxy is None:
        return []

    pts_list = obb.xyxyxyxy.cpu().numpy().astype(np.float32)
    confs = obb.conf.cpu().numpy() if obb.conf is not None else np.ones((len(pts_list),), dtype=np.float32)
    candidates: list[dict] = []

    for crop_pts, det_conf in zip(pts_list, confs):
        if float(det_conf) < PLATE_DET_CONF:
            continue

        global_pts = map_crop_points_to_global(crop_pts, vehicle_crop.offset)
        raw_x, raw_y, raw_w, raw_h = cv2.boundingRect(global_pts.astype(np.int32))
        if raw_w < MIN_PLATE_W or raw_h < MIN_PLATE_H:
            continue

        plate_crop = warp_plate_crop(frame, global_pts)
        if plate_crop.size == 0:
            continue
        if not is_router_candidate(plate_crop):
            continue

        candidates.append({
            "box": [raw_x, raw_y, raw_x + raw_w, raw_y + raw_h],
            "pts": global_pts,
            "crop": plate_crop,
            "conf": float(det_conf),
            "source_vehicle_id": vehicle_crop.vehicle_id,
        })
    return candidates


def detect_plate_tracks_cascade(
    frame: np.ndarray,
    tracked: list[dict],
    plate_model: Any,
    plate_tracker: PlateTrackManager,
    *,
    use_half: bool | None = None,
    timings: dict[str, float] | None = None,
) -> list[dict]:
    """Detect plates from tracked vehicle crops and return global plate tracks."""
    start = time.perf_counter()
    vehicle_crops = crop_vehicle_regions(frame, tracked)
    if timings is not None:
        timings["crop_prep"] = timings.get("crop_prep", 0.0) + time.perf_counter() - start
    if not vehicle_crops:
        plate_tracker.update([])
        return []

    images = [crop.image for crop in vehicle_crops]
    if use_half is None:
        use_half = torch.cuda.is_available()

    start = time.perf_counter()
    with torch.inference_mode():
        results = plate_model.predict(images, verbose=False, half=use_half)
    if timings is not None:
        timings["plate_cascade"] = timings.get("plate_cascade", 0.0) + time.perf_counter() - start

    start = time.perf_counter()
    candidates: list[dict] = []
    for result, vehicle_crop in zip(results, vehicle_crops):
        candidates.extend(_extract_obb_candidates(result, vehicle_crop, frame))

    deduped = deduplicate_plate_candidates(candidates, tracked)
    plate_tracks = plate_tracker.update(deduped)
    if timings is not None:
        timings["plate_postprocess"] = timings.get("plate_postprocess", 0.0) + time.perf_counter() - start
    return plate_tracks
