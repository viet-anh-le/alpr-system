from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from api.core.cascade_plate import crop_vehicle_regions
from api.core.config import (
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
    VEHICLE_CLASSES,
)
from api.core.frame_source import FileFrameSource
from api.core.gates import is_sharp
from api.core.models import load_models
from api.core.video_processor import warp_plate_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose which filter removes plate candidates in the cascade plate pipeline."
    )
    parser.add_argument("video", type=Path, help="Path to the input video.")
    parser.add_argument("--t-start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--t-end", type=float, default=5.0, help="End time in seconds.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
      raise FileNotFoundError(f"Video not found: {video_path}")

    source = FileFrameSource(video_path, t_start=args.t_start, t_end=args.t_end)
    models = load_models()
    vehicle_tracker = models.create_vehicle_tracker()

    counts = Counter()
    per_frame = []
    use_half = torch.cuda.is_available()

    for i, (_src_idx, frame, _ts) in enumerate(source.iter_frames(), start=1):
        if args.max_frames is not None and i > args.max_frames:
            break

        v_pred = models.vehicle.predict(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        if v_pred.boxes is not None and len(v_pred.boxes) > 0:
            xyxy = v_pred.boxes.xyxy.cpu().numpy()
            conf = v_pred.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = v_pred.boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)

        boxes, ids, classes = vehicle_tracker.track(dets, frame)
        tracked = []
        for box, tid, cid in zip(boxes, ids, classes):
            tracked.append({"id": int(tid), "box": box.tolist()})

        vehicle_crops = crop_vehicle_regions(frame, tracked)
        counts["frames"] += 1
        counts["tracked_vehicles"] += len(tracked)
        counts["vehicle_crops"] += len(vehicle_crops)

        if not vehicle_crops:
            per_frame.append((i, 0, 0, 0, 0, 0))
            continue

        images = [crop.image for crop in vehicle_crops]
        with torch.inference_mode():
            results = models.plate.predict(images, verbose=False, half=use_half)

        frame_stats = Counter()
        for result, vehicle_crop in zip(results, vehicle_crops):
            if result.obb is None or result.obb.xyxyxyxy is None:
                frame_stats["no_obb"] += 1
                continue

            pts_list = result.obb.xyxyxyxy.cpu().numpy().astype(np.float32)
            confs = result.obb.conf.cpu().numpy() if result.obb.conf is not None else np.ones((len(pts_list),), dtype=np.float32)

            if len(pts_list) == 0:
                frame_stats["no_obb"] += 1
                continue

            for crop_pts, det_conf in zip(pts_list, confs):
                counts["raw_plate_dets"] += 1
                if float(det_conf) < PLATE_DET_CONF:
                    counts["reject_low_conf"] += 1
                    frame_stats["reject_low_conf"] += 1
                    continue

                global_pts = crop_pts.copy()
                global_pts[:, 0] += vehicle_crop.offset[0]
                global_pts[:, 1] += vehicle_crop.offset[1]
                raw_x, raw_y, raw_w, raw_h = cv2.boundingRect(global_pts.astype(np.int32))
                if raw_w < MIN_PLATE_W or raw_h < MIN_PLATE_H:
                    counts["reject_small"] += 1
                    frame_stats["reject_small"] += 1
                    continue

                plate_crop = warp_plate_crop(frame, global_pts)
                if plate_crop.size == 0:
                    counts["reject_empty_warp"] += 1
                    frame_stats["reject_empty_warp"] += 1
                    continue

                if not is_sharp(plate_crop):
                    counts["reject_blurry"] += 1
                    frame_stats["reject_blurry"] += 1
                    continue

                counts["accepted"] += 1
                frame_stats["accepted"] += 1

        per_frame.append(
            (
                i,
                frame_stats["accepted"],
                frame_stats["reject_low_conf"],
                frame_stats["reject_small"],
                frame_stats["reject_empty_warp"],
                frame_stats["reject_blurry"],
            )
        )

    print("COUNTS", dict(counts))
    print("FRAMES", counts["frames"])
    print("TRACKED_VEHICLES", counts["tracked_vehicles"])
    print("VEHICLE_CROPS", counts["vehicle_crops"])
    print("RAW_PLATE_DETS", counts["raw_plate_dets"])
    print("ACCEPTED", counts["accepted"])
    print("REJECT_LOW_CONF", counts["reject_low_conf"])
    print("REJECT_SMALL", counts["reject_small"])
    print("REJECT_EMPTY_WARP", counts["reject_empty_warp"])
    print("REJECT_BLURRY", counts["reject_blurry"])
    print("PER_FRAME_SAMPLE", per_frame[:20])


if __name__ == "__main__":
    main()
