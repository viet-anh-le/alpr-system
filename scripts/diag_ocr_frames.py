"""
Diagnostic script: show every per-crop OCR result and every rejection gate
for a given video.  Run from ALPR_Vietnamese/:

    python scripts/diag_ocr_frames.py data/realworld-videos/chunks/hn_oto_18.mp4

Output per track:
  - Each buffered crop → raw OCR text + per-character confidence
  - Whether the crop passed is_sharp, quality_score threshold
  - What _segment_vote / _prob_vote produced
  - Whether _plate_valid accepted or rejected the voted result
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "LPRNet"))

import cv2
import numpy as np
import torch

from api.core.config import (
    CHARS,
    CONF_THRESHOLD,
    FRAME_STRIDE,
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
    TOP_K_FRAMES,
    VEHICLE_CLASSES,
)
from api.core.gates import is_sharp
from api.core.models import load_models, ocr_batch, preprocess_plate
from api.core.quality_scorer import quality_score
from api.core.tracker import WebTrackletManager
from api.core.tracker_adapter import VehicleTracker
from api.core.association import TrajectoryAssociator
from api.core.video_processor import crop_vehicle, warp_plate_crop

_PLATE_TRACKER_CFG = str(ROOT / "configs/tracking/bytetrack_plate.yaml")

_VN_PLATE_RE = re.compile(
    r"^(?:"
    r"\d{2}[A-Z]{1,2}-\d{5}"
    r"|\d{2}-(?:[A-Z]\d|[A-Z]{2})-\d{5}"
    r"|\d{2}[A-Z]-\d{4}"
    r"|\d{2}-[A-Z]\d-\d{4}"
    r")$"
)


def plate_valid(char_probs: list[tuple[str, float]]) -> bool:
    return bool(_VN_PLATE_RE.match("".join(c for c, _ in char_probs)))


# ── Per-track crop log ────────────────────────────────────────────────────────

class CropLog:
    def __init__(self):
        # frame_idx → list of (crop, quality, ocr_text, char_probs, all_conf)
        self.frames: list[tuple[int, float, str, list[tuple[str, float]], bool]] = []

    def add(self, frame_idx, quality, text, char_probs, all_conf):
        self.frames.append((frame_idx, quality, text, char_probs, all_conf))


def run(video_path: str) -> None:
    print(f"\n{'='*70}")
    print(f"Video: {video_path}")
    print(f"Thresholds: PLATE_DET_CONF={PLATE_DET_CONF}, CONF_THRESHOLD={CONF_THRESHOLD}")
    print(f"TOP_K_FRAMES={TOP_K_FRAMES}, FRAME_STRIDE={FRAME_STRIDE}")
    print(f"{'='*70}\n")

    models = load_models()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = WebTrackletManager()
    associator = TrajectoryAssociator(match_frames=5, agreement_ratio=0.6)
    models.vehicle_tracker.reset()

    # per-track diagnostic log
    crop_logs: dict[int, CropLog] = defaultdict(CropLog)

    # gate rejection counters
    gate_counts = {
        "det_conf_fail": 0,
        "size_fail": 0,
        "sharp_fail": 0,
        "no_firm_match": 0,
        "already_done": 0,
        "buffered": 0,
    }

    previously_tracked: set[int] = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  [frame {frame_idx}/{total}]", flush=True)

        v_pred = models.vehicle.predict(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
        if v_pred.boxes is not None and len(v_pred.boxes) > 0:
            xyxy = v_pred.boxes.xyxy.cpu().numpy()
            conf = v_pred.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls  = v_pred.boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
        else:
            dets = np.zeros((0, 6), dtype=np.float32)

        boxes, ids, classes = models.vehicle_tracker.track(dets, frame)

        tracked: list[dict] = []
        currently_tracked: set[int] = set()
        for box, tid, cid in zip(boxes, ids, classes):
            tid = int(tid)
            tracker._cls[tid] = models.vehicle.names[int(cid)]
            tracked.append({"id": tid, "box": box.tolist()})
            currently_tracked.add(tid)
            if tid in tracker._lost_count:
                tracker.reset_lost(tid)

        if frame_idx % FRAME_STRIDE != 0:
            previously_tracked = currently_tracked
            continue

        # Finalise tracks that just disappeared
        for tid in previously_tracked - currently_tracked:
            if tracker.should_ocr(tid) and tracker.mark_lost(tid) and tracker.ready_for_track_ocr(tid):
                _finalise(tid, tracker, crop_logs, models)

        p_res = models.plate.track(frame, persist=True, tracker=_PLATE_TRACKER_CFG, verbose=False)[0]

        plate_tracks: list[dict] = []
        H, W = frame.shape[:2]

        if p_res.obb is not None and p_res.obb.id is not None:
            obb_pts  = p_res.obb.xyxyxyxy.cpu().numpy().astype(int)
            obb_conf = p_res.obb.conf.cpu().numpy()
            obb_ids  = p_res.obb.id.cpu().numpy().astype(int)

            for pts, det_conf, p_tid in zip(obb_pts, obb_conf, obb_ids):
                if float(det_conf) < PLATE_DET_CONF:
                    gate_counts["det_conf_fail"] += 1
                    continue
                raw_rx, raw_ry, raw_rw, raw_rh = cv2.boundingRect(pts)
                if raw_rw < MIN_PLATE_W or raw_rh < MIN_PLATE_H:
                    gate_counts["size_fail"] += 1
                    continue
                plate_crop = warp_plate_crop(frame, pts)
                if plate_crop.size == 0:
                    continue
                if not is_sharp(plate_crop):
                    gate_counts["sharp_fail"] += 1
                    continue
                plate_tracks.append({
                    "id": int(p_tid),
                    "box": [raw_rx, raw_ry, raw_rx+raw_rw, raw_ry+raw_rh],
                    "crop": plate_crop,
                    "conf": float(det_conf),
                })

        firm_matches = associator.process_frame(plate_tracks, tracked)

        # Track which plate_track ids got matched
        matched_p_tids = {p_tid for _, p in firm_matches
                          for p_tid in [p["id"]] if p is not None}
        not_matched = len(plate_tracks) - len(firm_matches)
        gate_counts["no_firm_match"] += not_matched

        for v_tid, p in firm_matches:
            v_box = associator.vehicle_cache.get(v_tid)
            if not tracker.should_ocr(v_tid):
                gate_counts["already_done"] += 1
                continue

            plate_crop = p["crop"]
            q = quality_score(plate_crop)

            # Run per-frame OCR (same path as the fixed pipeline_core)
            tensor = preprocess_plate(plate_crop).unsqueeze(0).to(models.device)
            results = ocr_batch(models.ocr, tensor, models.device)
            char_probs, all_conf = results[0]
            ocr_conf = (
                sum(p for _, p in char_probs) / len(char_probs)
                if char_probs else 0.10
            )
            tracker.buffer_crop(v_tid, plate_crop, q, ocr_conf, char_probs, frame_idx)
            if v_box is not None:
                tracker.update_vehicle_img(v_tid, crop_vehicle(frame, v_box), q)
            text = "".join(c for c, _ in char_probs)
            crop_logs[v_tid].add(frame_idx, q, text, char_probs, all_conf)
            gate_counts["buffered"] += 1

        previously_tracked = currently_tracked

    cap.release()

    # Finalise remaining tracks
    for tid in list(tracker._buffers):
        if tracker.should_ocr(tid) and tracker.ready_for_track_ocr(tid):
            _finalise(tid, tracker, crop_logs, models)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("GATE SUMMARY")
    print(f"{'='*70}")
    for k, v in gate_counts.items():
        print(f"  {k:25s}: {v}")

    print(f"\n{'='*70}")
    print("PER-TRACK OCR FRAMES (all buffered crops)")
    print(f"{'='*70}\n")

    for tid in sorted(crop_logs):
        log = crop_logs[tid]
        n = len(log.frames)
        texts = [t for _, _, t, _, _ in log.frames]
        confs = [
            [round(p, 3) for _, p in cp]
            for _, _, _, cp, _ in log.frames
        ]
        avg_q = sum(q for _, q, *_ in log.frames) / n if n else 0
        print(f"Track {tid:>4d}  |  {n} OCR frames  |  avg quality={avg_q:.3f}")
        for f_idx, q, text, cp, all_c in log.frames:
            all_conf_mark = "✓" if all_c else "✗"
            valid_mark    = "✓" if plate_valid(cp) else "✗"
            min_p = min((p for _, p in cp), default=0.0)
            print(f"         f{f_idx:04d}  q={q:.3f}  [{all_conf_mark}allconf] [{valid_mark}valid]  "
                  f"'{text}'  min_char_conf={min_p:.3f}  "
                  f"chars={[(c, round(p,3)) for c,p in cp]}")
        print()


def _finalise(
    tid: int,
    tracker: WebTrackletManager,
    crop_logs: dict,
    models,
) -> None:
    from api.core.tracker import WebTrackletManager

    crops, scores, prob_lists_cached = tracker._buffers[tid].top_k(k=TOP_K_FRAMES)
    if not crops:
        return

    prob_lists = prob_lists_cached

    vote_texts = [("".join(c for c, _ in pl), pl) for pl in prob_lists]

    seg = WebTrackletManager._segment_vote(prob_lists)
    if seg is not None:
        method = "segment_vote"
        voted = seg
    else:
        voted = WebTrackletManager._prob_vote(prob_lists)
        method = "prob_vote"

    voted_text = "".join(c for c, _ in voted)
    valid = bool(re.match(
        r"^(?:\d{2}[A-Z]{1,2}-\d{5}|\d{2}-(?:[A-Z]\d|[A-Z]{2})-\d{5}|\d{2}[A-Z]-\d{4}|\d{2}-[A-Z]\d-\d{4})$",
        voted_text
    ))

    log = crop_logs[tid]
    frames_in_buf = [f for f, *_ in log.frames]

    print(f"\n  ── FINALISE track {tid} ──────────────────────────────────────────")
    print(f"     Method: {method}")
    print(f"     Top-{TOP_K_FRAMES} crops passed to vote (frames: {frames_in_buf}):")
    for i, (text, pl) in enumerate(vote_texts):
        min_p = min((p for _, p in pl), default=0.0)
        print(f"       [{i}] '{text}'  min_char_conf={min_p:.3f}")
    print(f"     Voted result: '{voted_text}'  valid={valid}")
    print(f"     Chars: {[(c, round(p,3)) for c,p in voted]}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "data/realworld-videos/chunks/hn_oto_18.mp4"
    )
    run(video)
