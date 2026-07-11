from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "detection" / "best.pt"
DEFAULT_OUTPUT = ROOT / "data" / "outputs" / "plate_inference.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the YOLO plate detector on one frame and save an annotated image."
    )
    parser.add_argument("image", type=Path, help="Path to the input frame.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"YOLO plate weights. Default: {DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Annotated output image. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    return parser.parse_args()


def _box_label(confidence: float, class_name: str | None) -> str:
    prefix = class_name or "plate"
    return f"{prefix} {confidence:.2f}"


def _draw_label(image: np.ndarray, x: int, y: int, label: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y - text_h - baseline - 6)
    cv2.rectangle(image, (x, top), (x + text_w + 8, top + text_h + baseline + 6), (0, 255, 0), -1)
    cv2.putText(image, label, (x + 4, top + text_h + 2), font, scale, (0, 0, 0), thickness)


def _extract_obb(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    if result.obb is None or result.obb.xyxyxyxy is None:
        return []

    points = result.obb.xyxyxyxy.cpu().numpy().astype(np.float32)
    confs = result.obb.conf.cpu().numpy()
    classes = result.obb.cls.cpu().numpy().astype(int) if result.obb.cls is not None else np.zeros(len(points), dtype=int)

    detections: list[dict[str, Any]] = []
    for pts, conf, cls_id in zip(points, confs, classes):
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        detections.append(
            {
                "type": "obb",
                "class_id": int(cls_id),
                "class_name": names.get(int(cls_id), "plate"),
                "confidence": float(conf),
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "points": pts.round(2).tolist(),
            }
        )
    return detections


def _extract_xyxy(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    if result.boxes is None or result.boxes.xyxy is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)

    detections: list[dict[str, Any]] = []
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box.astype(int).tolist()
        detections.append(
            {
                "type": "xyxy",
                "class_id": int(cls_id),
                "class_name": names.get(int(cls_id), "plate"),
                "confidence": float(conf),
                "box": [x1, y1, x2, y2],
            }
        )
    return detections


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        label = _box_label(detection["confidence"], detection.get("class_name"))
        if detection["type"] == "obb":
            pts = np.array(detection["points"], dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = pts[:, 0].min(), pts[:, 1].min()
            _draw_label(annotated, int(x), int(y), label)
            continue

        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _draw_label(annotated, x1, y1, label)
    return annotated


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    model = YOLO(str(weights_path))
    result = model.predict(str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    names = result.names if isinstance(result.names, dict) else model.names

    detections = _extract_obb(result, names) or _extract_xyxy(result, names)
    annotated = draw_detections(image, detections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not write output image: {output_path}")

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"detections": detections}, indent=2), encoding="utf-8")
    print(f"saved_image={output_path}")
    print(f"saved_json={metadata_path}")
    print(f"detections={len(detections)}")


if __name__ == "__main__":
    main()
