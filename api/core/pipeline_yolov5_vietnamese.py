from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Callable

import numpy as np
import torch
import cv2

from .config import ALPR_PREVIEW_FPS, FRAME_STRIDE
from .frame_source import FrameSource
from .models import ModelBundle
from .progress import make_progress_event
from .video_processor import draw_annotated_frame

import sys
from .config import ROOT
if str(ROOT / "references" / "Character-Time-series-Matching") not in sys.path:
    sys.path.insert(0, str(ROOT / "references" / "Character-Time-series-Matching"))
if str(ROOT / "references" / "Character-Time-series-Matching" / "yolov5") not in sys.path:
    sys.path.insert(0, str(ROOT / "references" / "Character-Time-series-Matching" / "yolov5"))

import process_plate
from utils.general import non_max_suppression, scale_coords
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

logger = logging.getLogger(__name__)

def preprocess_image_object(bgr: np.ndarray, size=(1280, 1280), device='cpu'):
    # Resizing logic from DETECTION.py (ResizeImg)
    h1, w1, _ = bgr.shape
    h, w = size
    if w1 < h1 * (w / h):
        img_rs = cv2.resize(bgr, (int(float(w1 / h1) * h), h))
        mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
        img = cv2.hconcat([img_rs, mask])
        trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
        trans_y = 0
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = img.shape[:2]
        img = cv2.warpAffine(img, trans_m, (width, height))
    else:
        img_rs = cv2.resize(bgr, (w, int(float(h1 / w1) * w)))
        mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
        img = cv2.vconcat([img_rs, mask])
        trans_x = 0
        trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = img.shape[:2]
        img = cv2.warpAffine(img, trans_m, (width, height))
        
    image = img.copy()[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device)
    image = image.float() / 255.0
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    return image, img, trans_x, trans_y

def preprocess_image_char(bgr: np.ndarray, size=(128, 128), device='cpu'):
    h1, w1, _ = bgr.shape
    h, w = size
    stride = 64
    import math
    if w1 < h1*(w/h):
        char_digit = cv2.resize(bgr, (int(float(w1/h1)*h), h), cv2.INTER_LANCZOS4)
        a = math.ceil(int(float(w1/h1)*h)/stride)
        b = a*stride-int(float(w1/h1)*h)
        mask1 = np.full((h, b//2, 3), 114, np.uint8)
        mask2 = np.full((h, b-b//2, 3), 114, np.uint8)
        thresh = cv2.hconcat([mask2, char_digit, mask1])
    else:
        char_digit = cv2.resize(bgr, (w, int(float(h1/w1)*w)), cv2.INTER_LANCZOS4)
        a = math.ceil(int(float(h1/w1)*w)/stride)
        b = a*stride-int(float(h1/w1)*w)
        mask1 = np.full((b//2, w, 3), 114, np.uint8)
        mask2 = np.full((b-b//2, w, 3), 114, np.uint8)
        thresh = cv2.vconcat([mask2, char_digit, mask1])
        
    image = thresh.copy()[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device)
    image = image.float() / 255.0
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    return image, thresh


def get_final_plate_text(track_boxs, Hs, Ws):
    old_char = np.zeros((0, 0))
    arr_track = old_char
    for track_box in track_boxs:
        if len(track_box) == 0:
            continue
        arr_track = process_plate.matching_char(old_char, track_box)
        old_char = arr_track
        
    if not isinstance(arr_track, np.ndarray):
        arr_track = np.array(arr_track)
        
    if len(arr_track) == 0:
        return ""
        
    Hm = np.mean(np.array(Hs)) if len(Hs)>0 else 0
    Wm = np.mean(np.array(Ws)) if len(Ws)>0 else 0

    if arr_track.shape[0] > 7:
        arr_track = process_plate.merge_box_arr_track(arr_track)
    arr_track = sorted(arr_track, key=lambda x: float(x[0]))
    
    re = ""
    # Filter chars appearing in >= 50% frames
    arr_track = np.array([arr_ for arr_ in arr_track if len(arr_[5]) >= 0.5 * len(track_boxs)], dtype=object)
    
    for arr_ in arr_track:
        clss = process_plate.get_maximum_conf_char(arr_)   
        re += clss         
        
    if Hm * 2 > Wm and len(arr_track) > 0:
        center_x = (arr_track[:, 0] + arr_track[:, 2]) / 2
        center_y = (arr_track[:, 1] + arr_track[:, 3]) / 2
        chars = ["{}".format(process_plate.get_maximum_conf_char(track_box_)) for track_box_ in arr_track]
        try:
            _, re = process_plate.find_chars_plate(center_x, center_y, chars)
        except Exception:
            pass
            
    re = re.replace("-", "")
    if len(re) >= 3:
        re = re[0:3].replace("0", "O").replace("1", "I") + re[3:]
    return re


def process_frames_yolov5_vietnamese(
    source: FrameSource,
    emit: Callable[[dict], None],
    models: ModelBundle,
    *,
    session_id: str = "",
    loop: asyncio.AbstractEventLoop | None = None,
    mjpeg_queue: asyncio.Queue | None = None,
    record_save: Callable | None = None,
    timings: dict[str, float] | None = None,
) -> dict:
    total_start = time.perf_counter()
    
    def _add_timing(name: str, started_at: float) -> None:
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started_at

    def emit_frame(jpg: bytes) -> None:
        if mjpeg_queue is not None and loop is not None:
            if not mjpeg_queue.full():
                loop.call_soon_threadsafe(mjpeg_queue.put_nowait, jpg)

    plate_tracker = ByteTrack(
        min_conf=0.1,
        track_thresh=0.45,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=30,
    )
    
    total = source.total_frames or 0
    frame_idx = 0
    processed_seen = 0
    preview_seen = 0
    preview_stride = max(1, int(round(source.fps / ALPR_PREVIEW_FPS))) if ALPR_PREVIEW_FPS > 0 else 0

    if getattr(models, "yolov5_object", None) is None or getattr(models, "ocr_yolov5", None) is None:
        logger.error("YOLOv5 models (object.pt or char.pt) are not loaded. Cannot run YOLOv5 Vietnamese pipeline.")
        return {"total_vehicles": 0, "processed_frames": processed_seen}

    obj_model = models.yolov5_object.model
    obj_names = models.yolov5_object.names
    char_model = models.ocr_yolov5.model
    char_names = models.ocr_yolov5.names
    
    # Track states
    # tid -> list of track_box arrays
    plate_char_history = {}
    plate_dims_history = {}
    done_tids = set()
    best_results = {}
    best_plate_crop = {}
    previously_tracked = set()
    
    def encode_img(img):
        if img is None:
            return ""
        _, buf = cv2.imencode(".jpg", img)
        import base64
        return base64.b64encode(buf).decode("utf-8")

    for src_idx, frame, _ts in source.iter_frames():
        frame_idx = src_idx + 1
        processed_seen += 1

        # 1. Object Detection (Plates)
        stage_start = time.perf_counter()
        img_tensor, resized_img, trans_x, trans_y = preprocess_image_object(frame, size=(1280, 1280), device=models.device)
        preds = obj_model(img_tensor, augment=False)[0]
        detections = non_max_suppression(preds, conf_thres=0.5, iou_thres=0.5, multi_label=True, max_det=100)
        _add_timing("vehicle_detect", stage_start)

        dets_for_tracker = []
        for det in detections:
            if len(det):
                det[:, :4] = scale_coords(resized_img.shape[:2], det[:, :4], frame.shape[:2]).round()
                for *xyxy, conf, cls in det.tolist():
                    name = obj_names[int(cls)]
                    if name in ['square license plate', 'rectangle license plate', 'car', 'truck', 'van', 'bus', 'motorbike', 'delivery tricycle']:
                        dets_for_tracker.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
        
        dets_arr = np.array(dets_for_tracker, dtype=np.float32) if dets_for_tracker else np.zeros((0, 6), dtype=np.float32)

        # 2. Tracking
        stage_start = time.perf_counter()
        
        if len(dets_arr) > 0:
            tracked_res = plate_tracker.update(dets_arr, frame)
        else:
            tracked_res = np.zeros((0, 8), dtype=np.float32)
            
        _add_timing("vehicle_track", stage_start)

        currently_tracked = set()
        active_plate_boxes = []

        for res in tracked_res:
            box = res[:4]
            tid = int(res[4])
            cid = int(res[6])
            
            currently_tracked.add(tid)
            name = obj_names[int(cid)]
            
            # Only process if it's a license plate
            if name in ['square license plate', 'rectangle license plate']:
                x1, y1, x2, y2 = [int(v) for v in box]
                active_plate_boxes.append({"id": tid, "box": [x1, y1, x2, y2], "cls": "plate"})
                
                # Crop plate
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                    
                plate_crop = frame[y1:y2, x1:x2]
                
                vx1 = max(0, x1 - (x2 - x1))
                vy1 = max(0, y1 - (y2 - y1))
                vx2 = min(frame.shape[1], x2 + (x2 - x1))
                vy2 = min(frame.shape[0], y2 + (y2 - y1))
                veh_crop = frame[vy1:vy2, vx1:vx2]
                
                area = plate_crop.shape[0] * plate_crop.shape[1]
                if tid not in best_plate_crop or area > best_plate_crop[tid][0]:
                    best_plate_crop[tid] = (area, plate_crop.copy(), veh_crop.copy())
                
                # Character Detection
                char_tensor, char_resized = preprocess_image_char(plate_crop, size=(128, 128), device=models.device)
                char_preds = char_model(char_tensor, augment=False)[0]
                char_dets = non_max_suppression(char_preds, conf_thres=0.1, iou_thres=0.1, multi_label=True, max_det=100)
                
                raw_char_detections = []
                for det in char_dets:
                    if len(det):
                        # Don't scale coords, evaluate.py works on the resized image scale or original?
                        # evaluate.py uses char_model.detect which doesn't scale back to original crop size.
                        # It uses the coords from `resized_img` directly!
                        for *xyxy, conf, cls_idx in det.tolist():
                            xc, yc = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
                            w_, h_ = (xyxy[2]-xyxy[0]), (xyxy[3]-xyxy[1])
                            raw_char_detections.append([char_names[int(cls_idx)], str(conf), (xc, yc, w_, h_)])
                            
                merged_dets = process_plate.merge_box(raw_char_detections)
                track_box = []
                for label, confidence, box_c in merged_dets:
                    track_box.append([
                        int(round(box_c[0] - box_c[2]/2)), int(round(box_c[1] - box_c[3]/2)),
                        int(round(box_c[0] + box_c[2]/2)), int(round(box_c[1] + box_c[3]/2)),
                        [[float(c)] for c in confidence.split("-")], [[l] for l in label.split("-")]
                    ])
                
                if track_box:
                    track_box = np.array(track_box, dtype=object)
                    if tid not in plate_char_history:
                        plate_char_history[tid] = []
                        plate_dims_history[tid] = {'H': [], 'W': []}
                    plate_char_history[tid].append(track_box)
                    plate_dims_history[tid]['H'].append(plate_crop.shape[0])
                    plate_dims_history[tid]['W'].append(plate_crop.shape[1])

        # Finalize lost tracks
        for tid in previously_tracked - currently_tracked:
            if tid in plate_char_history and tid not in done_tids:
                plate_text = get_final_plate_text(
                    plate_char_history[tid],
                    plate_dims_history[tid]['H'],
                    plate_dims_history[tid]['W']
                )
                if len(plate_text) > 0:
                    best_results[tid] = plate_text
                    plate_b64_str = encode_img(best_plate_crop.get(tid, (0, None, None))[1])
                    veh_b64_str = encode_img(best_plate_crop.get(tid, (0, None, None))[2])
                    
                    emit({
                        "type": "vehicle",
                        "id": tid,
                        "cls": "plate",
                        "plate": plate_text,
                        "done": True,
                        "chars": [],
                        "plate_b64": plate_b64_str, 
                        "vehicle_b64": veh_b64_str,
                        "ocr_frames": len(plate_char_history[tid]),
                        "confidence": 1.0,
                        "final": True,
                    })
                done_tids.add(tid)

        if processed_seen % 10 == 0 or (total and processed_seen >= total):
            emit(
                make_progress_event(
                    processed_frames=processed_seen,
                    total_frames=total,
                    source_frame=frame_idx,
                )
            )

        if mjpeg_queue is not None and preview_stride > 0:
            preview_seen += 1
            if preview_seen % preview_stride == 0:
                box_dicts = [
                    {
                        "id": v["id"],
                        "box": v["box"],
                        "state": "active",
                        "plate": best_results.get(v["id"], ""),
                        "cls": v["cls"],
                    }
                    for v in active_plate_boxes
                ]
                emit_frame(draw_annotated_frame(frame, box_dicts))

        previously_tracked = currently_tracked

    # Finalize remaining
    for tid in currently_tracked:
        if tid in plate_char_history and tid not in done_tids:
            plate_text = get_final_plate_text(
                plate_char_history[tid],
                plate_dims_history[tid]['H'],
                plate_dims_history[tid]['W']
            )
            if len(plate_text) > 0:
                best_results[tid] = plate_text
                plate_b64_str = encode_img(best_plate_crop.get(tid, (0, None, None))[1])
                veh_b64_str = encode_img(best_plate_crop.get(tid, (0, None, None))[2])
                
                emit({
                    "type": "vehicle",
                    "id": tid,
                    "cls": "plate",
                    "plate": plate_text,
                    "done": True,
                    "chars": [],
                    "plate_b64": plate_b64_str,
                    "vehicle_b64": veh_b64_str,
                    "ocr_frames": len(plate_char_history[tid]),
                    "confidence": 1.0,
                    "final": True,
                })

    if timings is not None:
        timings["total"] = time.perf_counter() - total_start

    return {
        "total_vehicles": len(best_results),
        "processed_frames": processed_seen,
    }

