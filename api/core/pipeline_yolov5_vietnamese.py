from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from typing import Callable

import cv2
import numpy as np
import torch

from .config import ALPR_PREVIEW_FPS, ROOT
from .frame_source import FrameSource
from .models import ModelBundle
from .preprocessing import apply_preprocessing, normalize_preprocess_mode
from .preview_frame import make_preview_frame_event
from .progress import make_progress_event

if str(ROOT / "references" / "Character-Time-series-Matching") not in sys.path:
    sys.path.insert(0, str(ROOT / "references" / "Character-Time-series-Matching"))
if str(ROOT / "references" / "Character-Time-series-Matching" / "yolov5") not in sys.path:
    sys.path.insert(0, str(ROOT / "references" / "Character-Time-series-Matching" / "yolov5"))

import process_plate
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from utils.general import non_max_suppression, scale_coords

logger = logging.getLogger(__name__)

CHARACTER_NAMES_PATH = ROOT / "references" / "Character-Time-series-Matching" / "character_name.txt"
# Char_detection_yolo.py / evaluate.py defaults
CHAR_DET_CONF = 0.05
CHAR_DET_IOU = 0.01
ROTATION_MIN_DEG = 3.0
ROTATION_MAX_DEG = 25.0
# Fair-comparison normalization: resize each plate crop to a fixed height before
# character detection so char-box coordinates stay comparable across a track
# (Che et al. evaluated on pre-cropped, consistent-size plate tracks).
CHAR_INPUT_HEIGHT = 96


def load_vn_character_names() -> list[str]:
    """Class labels for char.pt — matches reference character_name.txt."""
    if not CHARACTER_NAMES_PATH.exists():
        return []
    return [
        line.strip()
        for line in CHARACTER_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rotate_plate_crop(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a plate crop like evaluate.py (imutils.rotate, same canvas size)."""
    if abs(angle_deg) < 1e-6:
        return bgr
    height, width = bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def update_rotation_alpha(track_box: np.ndarray, alpha: float) -> float:
    """Accumulate deskew angle from detected character centers (evaluate.py)."""
    center_x = (track_box[:, 0] + track_box[:, 2]) / 2
    center_y = (track_box[:, 1] + track_box[:, 3]) / 2
    degree = process_plate.find_angle(center_x, center_y)
    if ROTATION_MIN_DEG < abs(math.degrees(degree)) < ROTATION_MAX_DEG:
        alpha -= degree
    return alpha

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

def preprocess_image_char(bgr: np.ndarray, size=(128, 128), device="cpu"):
    h1, w1, _ = bgr.shape
    h, w = size
    stride = 64
    if w1 < h1 * (w / h):
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


def detect_char_track_box(
    plate_crop: np.ndarray,
    *,
    char_model: torch.nn.Module,
    char_names: list[str],
    device: torch.device,
) -> np.ndarray | None:
    """Run char.pt on a (possibly rotated) plate crop; return track_box or None."""
    char_tensor, _char_resized = preprocess_image_char(plate_crop, size=(128, 128), device=device)
    char_preds = char_model(char_tensor, augment=False)[0]
    char_dets = non_max_suppression(
        char_preds,
        conf_thres=CHAR_DET_CONF,
        iou_thres=CHAR_DET_IOU,
        multi_label=True,
        max_det=1000,
    )

    raw_char_detections: list[list[object]] = []
    for det in char_dets:
        if not len(det):
            continue
        for *xyxy, conf, cls_idx in det.tolist():
            xc = (xyxy[0] + xyxy[2]) / 2
            yc = (xyxy[1] + xyxy[3]) / 2
            w_ = xyxy[2] - xyxy[0]
            h_ = xyxy[3] - xyxy[1]
            cls_id = int(cls_idx)
            if cls_id < 0 or cls_id >= len(char_names):
                continue
            raw_char_detections.append([char_names[cls_id], str(conf), (xc, yc, w_, h_)])

    if not raw_char_detections:
        return None

    merged_dets = process_plate.merge_box(raw_char_detections)
    track_box: list[list[object]] = []
    for label, confidence, box_c in merged_dets:
        track_box.append(
            [
                int(round(box_c[0] - box_c[2] / 2)),
                int(round(box_c[1] - box_c[3] / 2)),
                int(round(box_c[0] + box_c[2] / 2)),
                int(round(box_c[1] + box_c[3] / 2)),
                [[float(c)] for c in confidence.split("-")],
                [[char] for char in label.split("-")],
            ]
        )
    if not track_box:
        return None
    return np.array(track_box, dtype=object)


def _normalize_plate_scale(bgr: np.ndarray, target_h: int = CHAR_INPUT_HEIGHT) -> np.ndarray:
    """Resize a plate crop to a fixed height (aspect preserved) so the character
    detector sees a consistent scale every frame, keeping char-box coordinates
    stable across a track for matching_char."""
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return bgr
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (new_w, target_h), interpolation=interp)


def _char_confidence(track_box) -> float:
    """Support share of the winning character among its accumulated votes."""
    conf_list = np.array(track_box[4], dtype=float).flatten()
    if conf_list.size == 0:
        return 0.0
    cls_list = np.array([str(c) for c in np.array(track_box[5]).flatten()])
    sums = np.array([conf_list[cls_list == u].sum() for u in np.unique(cls_list)])
    total = float(conf_list.sum()) or 1.0
    return round(float(sums.max()) / total, 3)


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
        return "", []
        
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
            
    # Per-character evidence for the web UI (char + support-share confidence).
    char_probs = [
        [str(process_plate.get_maximum_conf_char(a)), _char_confidence(a)]
        for a in arr_track
    ]
    # Removed the Brazilian first-3-chars-are-letters rule (LLL-DDDD): Vietnamese
    # plates start with 2 DIGITS (province) + 1 letter, so it corrupts the province.
    re = re.replace("-", "")
    return re, char_probs


def _persist_vn_record(
    session_id: str,
    tid: int,
    plate_text: str,
    char_probs: list,
    crop_entries: list[dict],
    best_crop,
    veh_crop,
    loop,
    user_id,
) -> None:
    """Persist a Vietnamese-YOLOv5 track as a RecognitionRecord so the
    /records/{job}/{track} endpoint can serve it. Mirrors pipeline._record_save
    but builds the record from this pipeline's own state (no WebTrackletManager)."""
    if loop is None:
        return
    try:
        from api.database.mongodb import is_db_configured, upsert_record
        from api.database.models import PlateFrame as DBFrame, RecognitionRecord as DBRecord
        from .database import upload_image as _storage_upload

        if not is_db_configured():
            return

        def _upload(img, path, quality=85):
            if img is None:
                return None
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return _storage_upload("evidence", path, bytes(jpg)) if ok else None

        track_frames = [
            DBFrame(
                frame_index=int(e.get("frame_index", i)),
                quality_score=1.0,
                image_url=_upload(e.get("crop"), f"{session_id}/track_{tid}_frame_{i}.jpg"),
                ocr_confidence=1.0,
            )
            for i, e in enumerate(crop_entries)
        ]
        if not track_frames:
            return

        best_frame = DBFrame(
            frame_index=track_frames[0].frame_index,
            quality_score=1.0,
            image_url=_upload(best_crop, f"{session_id}/plate_{tid}.jpg", quality=90)
            or track_frames[0].image_url,
            ocr_text=plate_text,
            ocr_confidence=1.0,
        )
        plate_conf = (
            sum(float(p) for _, p in char_probs) / len(char_probs) if char_probs else 0.0
        )
        record = DBRecord(
            session_id=session_id,
            user_id=user_id,
            track_id=int(tid),
            vehicle_track_id=int(tid),
            plate_track_id=int(tid),
            vehicle_class="plate",
            best_plate_frame=best_frame,
            track_buffer=track_frames,
            vehicle_thumbnail_url=_upload(veh_crop, f"{session_id}/vehicle_{tid}.jpg"),
            plate_text=plate_text,
            plate_text_confidence=round(plate_conf, 4),
            ocr_vote_summary={},
            clusters=[],
            ocr_method="vietnamese_char_ctm",
            first_seen_frame=min(f.frame_index for f in track_frames),
            last_seen_frame=max(f.frame_index for f in track_frames),
        )
        asyncio.run_coroutine_threadsafe(upsert_record(record), loop)
    except Exception:
        logger.exception("VN pipeline: failed to save record for track %d", tid)


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
    user_id: str | None = None,
    emit_preview: bool = True,
    preprocess_mode: str = "none",
    preprocessed_frame_recorder: object | None = None,
) -> dict:
    del record_save  # incompatible callback (WebTrackletManager); VN persists via _persist_vn_record
    total_start = time.perf_counter()
    
    def _add_timing(name: str, started_at: float) -> None:
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started_at

    plate_tracker = ByteTrack(
        min_conf=0.1,
        track_thresh=0.45,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=30,
    )
    
    total = source.total_frames or 0
    normalized_preprocess_mode = normalize_preprocess_mode(preprocess_mode)
    active_recorder = (
        preprocessed_frame_recorder
        if normalized_preprocess_mode != "none"
        else None
    )
    frame_idx = 0
    processed_seen = 0
    preview_seen = 0
    preview_stride = (
        max(1, int(round(source.fps / ALPR_PREVIEW_FPS)))
        if emit_preview and ALPR_PREVIEW_FPS > 0
        else 0
    )

    if getattr(models, "yolov5_object", None) is None or getattr(models, "ocr_yolov5", None) is None:
        logger.error("YOLOv5 models (object.pt or char.pt) are not loaded. Cannot run YOLOv5 Vietnamese pipeline.")
        return {"total_vehicles": 0, "processed_frames": processed_seen}

    obj_model = models.yolov5_object.model
    obj_names = models.yolov5_object.names
    char_model = models.ocr_yolov5.model
    char_names = load_vn_character_names()
    if not char_names:
        char_names = list(models.ocr_yolov5.names)
        logger.warning(
            "character_name.txt not found; falling back to char.pt embedded class names."
        )

    # Track states
    # tid -> list of track_box arrays
    plate_char_history: dict[int, list[np.ndarray]] = {}
    plate_dims_history: dict[int, dict[str, list[int]]] = {}
    plate_crop_history: dict[int, list[dict]] = {}  # tid -> [{frame_index, crop}]
    plate_rotation_alpha: dict[int, float] = {}
    done_tids: set[int] = set()
    best_results: dict[int, str] = {}
    best_plate_crop: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    previously_tracked: set[int] = set()
    
    def encode_img(img):
        if img is None:
            return ""
        _, buf = cv2.imencode(".jpg", img)
        import base64
        return base64.b64encode(buf).decode("utf-8")

    def build_track_buffer(tid: int) -> list[dict]:
        """Per-frame evidence crops for the web track-buffer viewer."""
        return [
            {
                "frame_index": int(e["frame_index"]),
                "image_b64": encode_img(e["crop"]),
                "route": "vietnamese_char",
                "ocr_confidence": 1.0,
                "quality_score": 1.0,
                "candidate_method": "char_detection",
            }
            for e in plate_crop_history.get(tid, [])
        ]

    def schedule_record_save(tid: int, plate_text: str, char_probs: list) -> None:
        """Persist the track to MongoDB (off the inference thread) so
        /records/{job}/{track} can serve it later."""
        if loop is None:
            return
        best = best_plate_crop.get(tid, (0, None, None))
        try:
            loop.run_in_executor(
                None,
                _persist_vn_record,
                session_id,
                tid,
                plate_text,
                char_probs,
                list(plate_crop_history.get(tid, [])),
                best[1],
                best[2],
                loop,
                user_id,
            )
        except RuntimeError:
            logger.exception("VN: failed to schedule record save for track %d", tid)

    for src_idx, frame, _ts in source.iter_frames():
        frame_idx = src_idx + 1
        processed_seen += 1
        vehicle_frame = (
            frame
            if normalized_preprocess_mode == "none"
            else apply_preprocessing(frame, normalized_preprocess_mode)
        )
        if active_recorder is not None:
            active_recorder.record_frame(vehicle_frame)

        # 1. Object Detection (Plates)
        stage_start = time.perf_counter()
        img_tensor, resized_img, trans_x, trans_y = preprocess_image_object(
            vehicle_frame,
            size=(1280, 1280),
            device=models.device,
        )
        preds = obj_model(img_tensor, augment=False)[0]
        detections = non_max_suppression(preds, conf_thres=0.5, iou_thres=0.5, multi_label=True, max_det=100)
        _add_timing("vehicle_detect", stage_start)

        dets_for_tracker = []
        for det in detections:
            if len(det):
                det[:, :4] = scale_coords(
                    resized_img.shape[:2],
                    det[:, :4],
                    vehicle_frame.shape[:2],
                ).round()
                for *xyxy, conf, cls in det.tolist():
                    name = obj_names[int(cls)]
                    if name in ['square license plate', 'rectangle license plate', 'car', 'truck', 'van', 'bus', 'motorbike', 'delivery tricycle']:
                        dets_for_tracker.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
        
        dets_arr = np.array(dets_for_tracker, dtype=np.float32) if dets_for_tracker else np.zeros((0, 6), dtype=np.float32)

        # 2. Tracking
        stage_start = time.perf_counter()
        
        if len(dets_arr) > 0:
            tracked_res = plate_tracker.update(dets_arr, vehicle_frame)
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

                alpha = plate_rotation_alpha.setdefault(tid, 0.0)
                rotated_crop = rotate_plate_crop(plate_crop, math.degrees(alpha))
                # Normalize scale so char-box coords stay comparable across frames.
                norm_crop = _normalize_plate_scale(rotated_crop)
                track_box = detect_char_track_box(
                    norm_crop,
                    char_model=char_model,
                    char_names=char_names,
                    device=models.device,
                )
                if track_box is not None:
                    plate_rotation_alpha[tid] = update_rotation_alpha(track_box, alpha)
                    if tid not in plate_char_history:
                        plate_char_history[tid] = []
                        plate_dims_history[tid] = {"H": [], "W": []}
                        plate_crop_history[tid] = []
                    plate_char_history[tid].append(track_box)
                    plate_dims_history[tid]["H"].append(norm_crop.shape[0])
                    plate_dims_history[tid]["W"].append(norm_crop.shape[1])
                    if len(plate_crop_history[tid]) < 30:
                        plate_crop_history[tid].append(
                            {"frame_index": frame_idx, "crop": rotated_crop.copy()}
                        )

        # Finalize lost tracks
        for tid in previously_tracked - currently_tracked:
            if tid in plate_char_history and tid not in done_tids:
                plate_text, char_probs = get_final_plate_text(
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
                        "chars": char_probs,
                        "track_buffer": build_track_buffer(tid),
                        "plate_b64": plate_b64_str,
                        "vehicle_b64": veh_b64_str,
                        "ocr_frames": len(plate_char_history[tid]),
                        "confidence": 1.0,
                        "final": True,
                    })
                    schedule_record_save(tid, plate_text, char_probs)
                done_tids.add(tid)

        if processed_seen % 10 == 0 or (total and processed_seen >= total):
            emit(
                make_progress_event(
                    processed_frames=processed_seen,
                    total_frames=total,
                    source_frame=frame_idx,
                )
            )

        if preview_stride > 0:
            preview_seen += 1
            if preview_seen % preview_stride == 0:
                box_dicts = [
                    {
                        "id": v["id"],
                        "kind": "plate",
                        "box": v["box"],
                        "state": "active",
                        "plate": best_results.get(v["id"], ""),
                        "cls": v["cls"],
                    }
                    for v in active_plate_boxes
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

    # Finalize remaining
    for tid in currently_tracked:
        if tid in plate_char_history and tid not in done_tids:
            plate_text, char_probs = get_final_plate_text(
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
                    "chars": char_probs,
                    "track_buffer": build_track_buffer(tid),
                    "plate_b64": plate_b64_str,
                    "vehicle_b64": veh_b64_str,
                    "ocr_frames": len(plate_char_history[tid]),
                    "confidence": 1.0,
                    "final": True,
                })
                schedule_record_save(tid, plate_text, char_probs)

    if timings is not None:
        timings["total"] = time.perf_counter() - total_start

    if active_recorder is not None:
        active_recorder.finish()

    return {
        "total_vehicles": len(best_results),
        "processed_frames": processed_seen,
    }
