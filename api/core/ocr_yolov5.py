"""
core/ocr_yolov5.py — YOLOv5 Character Detection OCR Backend
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from api.core.config import ROOT

# Ensure yolov5 is in path
YOLOV5_PATH = str(ROOT / "references" / "Character-Time-series-Matching" / "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression


@dataclass(frozen=True)
class YOLOv5CharOcrModel:
    model: torch.nn.Module
    names: list[str]

    def eval(self) -> "YOLOv5CharOcrModel":
        self.model.eval()
        return self


def load_yolov5_char_model(checkpoint_path: str | Path, *, device: torch.device) -> YOLOv5CharOcrModel:
    original_torch_load = torch.load
    def _patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    torch.load = _patched_load
    try:
        model = attempt_load(str(checkpoint_path), map_location=device)
    finally:
        torch.load = original_torch_load

    model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    return YOLOv5CharOcrModel(model=model, names=names)


def preprocess_plate_yolov5(bgr: np.ndarray, size: tuple[int, int] = (128, 128)) -> torch.Tensor:
    h1, w1, _ = bgr.shape
    h, w = size
    if w1 < h1 * (w / h):
        img_rs = cv2.resize(bgr, (int(float(w1 / h1) * h), h))
        mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
        img = cv2.hconcat([img_rs, mask])
        trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
        trans_y = 0
    else:
        img_rs = cv2.resize(bgr, (w, int(float(h1 / w1) * w)))
        mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
        img = cv2.vconcat([img_rs, mask])
        trans_x = 0
        trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
    
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = img.shape[:2]
    img = cv2.warpAffine(img, trans_m, (width, height))
    
    # BGR to RGB
    img_arr = img.copy()[:, :, ::-1].transpose(2, 0, 1)
    img_arr = np.ascontiguousarray(img_arr)
    
    tensor = torch.from_numpy(img_arr).float() / 255.0
    return tensor


def estimate_coef(x, y):
    n = np.size(x)
    if n == 0:
        return 0, 0
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    if SS_xx == 0:
        return 0, 0
    a = SS_xy / SS_xx
    b = m_y - a*m_x
    return (a, b)


def find_chars_plate_probs(center_x, center_y, chars, probs) -> list[tuple[str, float]]:
    a, b = estimate_coef(center_x, center_y)
    centers = np.vstack((center_x, center_y)).T

    uppers = []
    lowers = []
    for center, char, prob in zip(centers, chars, probs):
        if a * center[0] + b - center[1] < 0:
            lowers.append([center[0], char, prob])
        else:
            uppers.append([center[0], char, prob])

    uppers = sorted(uppers, key=lambda x: x[0])
    lowers = sorted(lowers, key=lambda x: x[0])
    
    result = []
    for upper in uppers:
        result.append((str(upper[1]), float(upper[2])))
    if len(uppers) > 0 and len(lowers) > 0:
        result.append(("-", 1.0))
    for lower in lowers:
        result.append((str(lower[1]), float(lower[2])))

    return result


@torch.no_grad()
def yolov5_char_ocr_batch(
    wrapper: YOLOv5CharOcrModel,
    images: torch.Tensor,
    device: torch.device,
) -> list[tuple[list[tuple[str, float]], bool]]:
    model = wrapper.model.to(device).eval()
    
    # Run YOLOv5 on batch
    preds = model(images.to(device, non_blocking=True), augment=False)[0]
    
    # NMS
    detections = non_max_suppression(preds, conf_thres=0.1, iou_thres=0.5, multi_label=True, max_det=1000)
    
    results = []
    for det in detections:
        det_list = det.tolist()
        if not len(det_list):
            results.append(([], False))
            continue
            
        centers_x, centers_y, chars, probs = [], [], [], []
        for *xyxy, conf, cls in det_list:
            xc = (xyxy[0] + xyxy[2]) / 2
            yc = (xyxy[1] + xyxy[3]) / 2
            centers_x.append(xc)
            centers_y.append(yc)
            chars.append(wrapper.names[int(cls)])
            probs.append(conf)
            
        if len(chars) > 0:
            try:
                char_probs = find_chars_plate_probs(np.array(centers_x), np.array(centers_y), chars, probs)
                # Determine confident based on actual characters (ignore '-')
                all_confident = True
                for c, p in char_probs:
                    if c != "-" and p < 0.85: # YOLOv5 confidence threshold
                        all_confident = False
                        break
                results.append((char_probs, all_confident))
            except Exception:
                char_probs = [(c, p) for c, p in zip(chars, probs)]
                results.append((char_probs, False))
        else:
            results.append(([], False))

    return results
