from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VEHICLE_WEIGHTS = ROOT / "weights" / "detection" / "vehicle_best.pt"
DEFAULT_PLATE_WEIGHTS = ROOT / "weights" / "detection" / "best.pt"
DEFAULT_OUTPUT = ROOT / "data" / "outputs" / "vehicle_and_plate_bbox.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO vehicle and plate detectors on one frame and save a combined annotation."
    )
    parser.add_argument("image", type=Path, help="Path to the input frame.")
    parser.add_argument(
        "--vehicle-weights",
        type=Path,
        default=DEFAULT_VEHICLE_WEIGHTS,
        help=f"Vehicle detector weights. Default: {DEFAULT_VEHICLE_WEIGHTS}",
    )
    parser.add_argument(
        "--plate-weights",
        type=Path,
        default=DEFAULT_PLATE_WEIGHTS,
        help=f"Plate detector weights. Default: {DEFAULT_PLATE_WEIGHTS}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Annotated output image. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--vehicle-conf", type=float, default=0.25, help="Vehicle confidence threshold.")
    parser.add_argument("--plate-conf", type=float, default=0.15, help="Plate confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    return parser.parse_args()


def _draw_label(image: np.ndarray, x: int, y: int, label: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y - text_h - baseline - 6)
    cv2.rectangle(image, (x, top), (x + text_w + 8, top + text_h + baseline + 6), color, -1)
    cv2.putText(image, label, (x + 4, top + text_h + 2), font, scale, (0, 0, 0), thickness)


def _extract_plate_detections(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

    if result.obb is not None and result.obb.xyxyxyxy is not None:
        points = result.obb.xyxyxyxy.cpu().numpy().astype(np.float32)
        confs = result.obb.conf.cpu().numpy()
        classes = result.obb.cls.cpu().numpy().astype(int) if result.obb.cls is not None else np.zeros(len(points), dtype=int)
        for pts, conf, cls_id in zip(points, confs, classes):
            x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
            detections.append({
                "type": "obb",
                "class_id": int(cls_id),
                "class_name": names.get(int(cls_id), "plate"),
                "confidence": float(conf),
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "points": pts.round(2).tolist(),
            })
        return detections

    if result.boxes is not None and result.boxes.xyxy is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
        for box, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append({
                "type": "xyxy",
                "class_id": int(cls_id),
                "class_name": names.get(int(cls_id), "plate"),
                "confidence": float(conf),
                "box": [x1, y1, x2, y2],
            })
    return detections


def _extract_vehicle_detections(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    if result.boxes is None or result.boxes.xyxy is None:
        return []

    detections: list[dict[str, Any]] = []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box.astype(int).tolist()
        detections.append({
            "type": "xyxy",
            "class_id": int(cls_id),
            "class_name": names.get(int(cls_id), "vehicle"),
            "confidence": float(conf),
            "box": [x1, y1, x2, y2],
        })
    return detections


def draw_combined(image: np.ndarray, vehicle_dets: list[dict[str, Any]], plate_dets: list[dict[str, Any]]) -> np.ndarray:
    annotated = image.copy()

    for det in vehicle_dets:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 140, 0), 2)
        _draw_label(annotated, x1, y1, label, (255, 140, 0))

    for det in plate_dets:
        if det["type"] == "obb":
            pts = np.array(det["points"], dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = pts[:, 0].min(), pts[:, 1].min()
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            _draw_label(annotated, int(x), int(y), label, (0, 255, 0))
            continue

        x1, y1, x2, y2 = det["box"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _draw_label(annotated, x1, y1, label, (0, 255, 0))

    return annotated


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    vehicle_weights = args.vehicle_weights.expanduser().resolve()
    plate_weights = args.plate_weights.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not vehicle_weights.exists():
        raise FileNotFoundError(f"Vehicle weights not found: {vehicle_weights}")
    if not plate_weights.exists():
        raise FileNotFoundError(f"Plate weights not found: {plate_weights}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    vehicle_model = YOLO(str(vehicle_weights))
    plate_model = YOLO(str(plate_weights))

    vehicle_result = vehicle_model.predict(str(image_path), conf=args.vehicle_conf, imgsz=args.imgsz, verbose=False)[0]
    plate_result = plate_model.predict(str(image_path), conf=args.plate_conf, imgsz=args.imgsz, verbose=False)[0]

    vehicle_names = vehicle_result.names if isinstance(vehicle_result.names, dict) else vehicle_model.names
    plate_names = plate_result.names if isinstance(plate_result.names, dict) else plate_model.names

    vehicle_dets = _extract_vehicle_detections(vehicle_result, vehicle_names)
    plate_dets = _extract_plate_detections(plate_result, plate_names)

    annotated = draw_combined(image, vehicle_dets, plate_dets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not write output image: {output_path}")

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({"vehicles": vehicle_dets, "plates": plate_dets}, indent=2),
        encoding="utf-8",
    )

    print(f"saved_image={output_path}")
    print(f"saved_json={metadata_path}")
    print(f"vehicles={len(vehicle_dets)}")
    print(f"plates={len(plate_dets)}")


if __name__ == "__main__":
    main()
