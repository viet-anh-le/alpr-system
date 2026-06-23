"""YOLOv5 vehicle detector adapter for the shared ALPR pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from api.core.config import ROOT

YOLOV5_PATH = str(ROOT / "references" / "Character-Time-series-Matching" / "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_PATH)

from models.experimental import attempt_load  # noqa: E402
from utils.general import non_max_suppression, scale_coords  # noqa: E402


@dataclass(frozen=True)
class YOLOv5Boxes:
    xyxy: torch.Tensor
    conf: torch.Tensor
    cls: torch.Tensor

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


@dataclass(frozen=True)
class YOLOv5Prediction:
    boxes: YOLOv5Boxes


@dataclass(frozen=True)
class YOLOv5VehicleDetector:
    model: torch.nn.Module
    names: list[str]
    device: torch.device
    input_size: tuple[int, int] = (1280, 1280)

    @torch.no_grad()
    def predict(
        self,
        bgr: np.ndarray,
        *,
        classes: list[int] | None = None,
        verbose: bool = False,
    ) -> list[YOLOv5Prediction]:
        del verbose
        image, resized = preprocess_vehicle_frame(bgr, size=self.input_size, device=self.device)
        model = self.model.to(self.device).eval()
        preds = model(image, augment=False)[0]
        detections = non_max_suppression(
            preds,
            conf_thres=0.5,
            iou_thres=0.5,
            classes=classes,
            multi_label=True,
            max_det=100,
        )
        det = detections[0] if detections else torch.zeros((0, 6), device=self.device)
        if len(det):
            det = det.clone()
            det[:, :4] = scale_coords(resized.shape[:2], det[:, :4], bgr.shape[:2]).round()
            det = det.to(self.device)
            boxes = YOLOv5Boxes(xyxy=det[:, :4], conf=det[:, 4], cls=det[:, 5])
        else:
            empty = torch.zeros((0,), device=self.device)
            boxes = YOLOv5Boxes(
                xyxy=torch.zeros((0, 4), device=self.device),
                conf=empty,
                cls=empty,
            )
        return [YOLOv5Prediction(boxes=boxes)]


def load_yolov5_vehicle_detector(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> YOLOv5VehicleDetector:
    original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        model = attempt_load(str(checkpoint_path), map_location=device)
    finally:
        torch.load = original_torch_load

    model.eval()
    names = model.module.names if hasattr(model, "module") else model.names
    return YOLOv5VehicleDetector(model=model, names=list(names), device=device)


def preprocess_vehicle_frame(
    bgr: np.ndarray,
    *,
    size: tuple[int, int] = (1280, 1280),
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    h1, w1, _ = bgr.shape
    h, w = size
    if w1 < h1 * (w / h):
        resized_w = int(float(w1 / h1) * h)
        img_rs = cv2.resize(bgr, (resized_w, h))
        mask = np.zeros((h, w - resized_w, 3), np.uint8)
        img = cv2.hconcat([img_rs, mask])
        trans_x = int(w / 2) - int(resized_w / 2)
        trans_y = 0
    else:
        resized_h = int(float(h1 / w1) * w)
        img_rs = cv2.resize(bgr, (w, resized_h))
        mask = np.zeros((h - resized_h, w, 3), np.uint8)
        img = cv2.vconcat([img_rs, mask])
        trans_x = 0
        trans_y = int(h / 2) - int(resized_h / 2)

    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = img.shape[:2]
    img = cv2.warpAffine(img, trans_m, (width, height))
    image = img.copy()[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image).to(device).float() / 255.0
    return tensor.unsqueeze(0), img
