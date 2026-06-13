from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data" / "trafficvision_snapshots"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "lp_finetune_vehicle_crops"
DEFAULT_VEHICLE_WEIGHTS = ROOT / "weights" / "detection" / "vehicle_best.pt"
DEFAULT_PLATE_WEIGHTS = ROOT / "weights" / "detection" / "best.pt"
DEFAULT_ALLOWED_VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle", "motorbike_rider")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cropped-vehicle LP fine-tuning data in YOLO OBB format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing source frames. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where images/ and labels/ will be created. Default: {DEFAULT_OUTPUT_DIR}",
    )
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
        help=f"License plate detector weights. Default: {DEFAULT_PLATE_WEIGHTS}",
    )
    parser.add_argument(
        "--vehicle-conf", type=float, default=0.25, help="Vehicle confidence threshold."
    )
    parser.add_argument(
        "--plate-conf", type=float, default=0.15, help="License plate confidence threshold."
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument(
        "--vehicle-classes",
        type=str,
        default=",".join(DEFAULT_ALLOWED_VEHICLE_CLASSES),
        help="Comma-separated vehicle class names to keep from stage 1.",
    )
    parser.add_argument(
        "--save-negative-crops",
        action="store_true",
        help="Keep cropped vehicles even when no plate is detected.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cropped image and label files if they already exist.",
    )
    return parser.parse_args()


def load_yolo_model(weights_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for this script. Install dependencies from requirements.txt first."
        ) from exc

    return YOLO(str(weights_path))


def clip_bbox_to_image(
    box: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    clipped_x1 = max(0, min(int(np.floor(x1)), image_width))
    clipped_y1 = max(0, min(int(np.floor(y1)), image_height))
    clipped_x2 = max(0, min(int(np.ceil(x2)), image_width))
    clipped_y2 = max(0, min(int(np.ceil(y2)), image_height))

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None

    return (clipped_x1, clipped_y1, clipped_x2, clipped_y2)


def crop_image(
    image: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image_height, image_width = image.shape[:2]
    clipped_box = clip_bbox_to_image(box, image_width=image_width, image_height=image_height)
    if clipped_box is None:
        raise ValueError(f"Cannot crop invalid box: {box}")

    x1, y1, x2, y2 = clipped_box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Crop is empty after clipping: {clipped_box}")

    return crop, clipped_box


def xyxy_box_to_obb_points(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def clamp_obb_points(points: Iterable[float], image_width: int, image_height: int) -> list[float]:
    clamped_points: list[float] = []
    raw_points = list(points)
    if len(raw_points) != 8:
        raise ValueError(f"OBB points must have 8 values, received {len(raw_points)}")

    for index, value in enumerate(raw_points):
        upper_bound = image_width if index % 2 == 0 else image_height
        clamped_points.append(float(min(max(value, 0.0), float(upper_bound))))

    return clamped_points


def normalize_obb_points(
    points: Iterable[float], image_width: int, image_height: int
) -> list[float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive for normalization.")

    normalized_points: list[float] = []
    clamped_points = clamp_obb_points(points, image_width=image_width, image_height=image_height)

    for index, value in enumerate(clamped_points):
        if index % 2 == 0:
            normalized_points.append(value / image_width)
        else:
            normalized_points.append(value / image_height)

    return normalized_points


def build_yolo_obb_label_line(
    class_id: int, points: Iterable[float], image_width: int, image_height: int
) -> str:
    normalized_points = normalize_obb_points(
        points, image_width=image_width, image_height=image_height
    )
    coordinate_tokens = " ".join(f"{value:.6f}" for value in normalized_points)
    return f"{class_id} {coordinate_tokens}"


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if is_image_file(path))


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir


def select_vehicle_detections(
    result: Any, allowed_vehicle_classes: set[str]
) -> list[tuple[int, str, float, tuple[float, float, float, float]]]:
    if result.boxes is None or result.boxes.xyxy is None:
        return []

    names = result.names if isinstance(result.names, dict) else {}
    boxes = result.boxes.xyxy.cpu().tolist()
    confs = (
        result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [1.0] * len(boxes)
    )
    class_ids = (
        result.boxes.cls.cpu().tolist() if result.boxes.cls is not None else [0.0] * len(boxes)
    )

    detections: list[tuple[int, str, float, tuple[float, float, float, float]]] = []
    for box, conf, class_id in zip(boxes, confs, class_ids):
        class_index = int(class_id)
        class_name = str(names.get(class_index, class_index)).lower()
        if class_name not in allowed_vehicle_classes:
            continue

        x1, y1, x2, y2 = box
        detections.append((class_index, class_name, float(conf), (x1, y1, x2, y2)))

    return detections


def extract_plate_label_lines(result: Any, crop_width: int, crop_height: int) -> list[str]:
    label_lines: list[str] = []

    if result.obb is not None and result.obb.xyxyxyxy is not None:
        obb_points_array = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
        class_ids = (
            result.obb.cls.cpu().numpy().astype(int).tolist()
            if result.obb.cls is not None
            else [0] * len(obb_points_array)
        )

        for class_id, points in zip(class_ids, obb_points_array):
            label_lines.append(
                build_yolo_obb_label_line(
                    class_id=class_id,
                    points=points.tolist(),
                    image_width=crop_width,
                    image_height=crop_height,
                )
            )
        return label_lines

    if result.boxes is None or result.boxes.xyxy is None:
        return label_lines

    boxes = result.boxes.xyxy.cpu().tolist()
    class_ids = (
        result.boxes.cls.cpu().numpy().astype(int).tolist()
        if result.boxes.cls is not None
        else [0] * len(boxes)
    )

    for class_id, box in zip(class_ids, boxes):
        label_lines.append(
            build_yolo_obb_label_line(
                class_id=class_id,
                points=xyxy_box_to_obb_points(tuple(float(value) for value in box)),
                image_width=crop_width,
                image_height=crop_height,
            )
        )

    return label_lines


def build_crop_stem(image_path: Path, vehicle_index: int) -> str:
    return f"{image_path.stem}_vehicle_{vehicle_index:04d}"


def process_image(
    image_path: Path,
    vehicle_model: Any,
    plate_model: Any,
    images_dir: Path,
    labels_dir: Path,
    *,
    vehicle_conf: float,
    plate_conf: float,
    imgsz: int,
    allowed_vehicle_classes: set[str],
    save_negative_crops: bool,
    overwrite: bool,
) -> dict[str, int]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    frame_height, frame_width = frame.shape[:2]
    if frame_height == 0 or frame_width == 0:
        raise ValueError(f"Image has invalid dimensions: {image_path}")

    vehicle_result = vehicle_model.predict(frame, conf=vehicle_conf, imgsz=imgsz, verbose=False)[0]
    vehicle_detections = select_vehicle_detections(
        vehicle_result, allowed_vehicle_classes=allowed_vehicle_classes
    )

    saved_crops = 0
    saved_labels = 0
    discarded_crops = 0

    for vehicle_index, (_, _, _, box) in enumerate(vehicle_detections):
        clipped_box = clip_bbox_to_image(box, image_width=frame_width, image_height=frame_height)
        if clipped_box is None:
            discarded_crops += 1
            continue

        try:
            crop, _ = crop_image(frame, clipped_box)
        except ValueError:
            discarded_crops += 1
            continue

        crop_height, crop_width = crop.shape[:2]
        if crop_width == 0 or crop_height == 0:
            discarded_crops += 1
            continue

        plate_result = plate_model.predict(crop, conf=plate_conf, imgsz=imgsz, verbose=False)[0]
        label_lines = extract_plate_label_lines(
            plate_result, crop_width=crop_width, crop_height=crop_height
        )

        if not label_lines and not save_negative_crops:
            discarded_crops += 1
            continue

        crop_stem = build_crop_stem(image_path, vehicle_index)
        image_output_path = images_dir / f"{crop_stem}.jpg"
        label_output_path = labels_dir / f"{crop_stem}.txt"

        if not overwrite and (image_output_path.exists() or label_output_path.exists()):
            raise FileExistsError(
                f"Refusing to overwrite existing files for crop '{crop_stem}'. Use --overwrite to replace them."
            )

        if not cv2.imwrite(str(image_output_path), crop):
            raise RuntimeError(f"Failed to write cropped image: {image_output_path}")

        label_output_path.write_text("\n".join(label_lines), encoding="utf-8")
        saved_crops += 1
        if label_lines:
            saved_labels += 1

    return {
        "vehicles": len(vehicle_detections),
        "saved_crops": saved_crops,
        "saved_labels": saved_labels,
        "discarded_crops": discarded_crops,
    }


def validate_inputs(input_dir: Path, vehicle_weights: Path, plate_weights: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if not vehicle_weights.exists():
        raise FileNotFoundError(f"Vehicle weights not found: {vehicle_weights}")
    if not plate_weights.exists():
        raise FileNotFoundError(f"Plate weights not found: {plate_weights}")


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    vehicle_weights = args.vehicle_weights.expanduser().resolve()
    plate_weights = args.plate_weights.expanduser().resolve()
    allowed_vehicle_classes = {
        class_name.strip().lower()
        for class_name in args.vehicle_classes.split(",")
        if class_name.strip()
    }

    validate_inputs(
        input_dir=input_dir, vehicle_weights=vehicle_weights, plate_weights=plate_weights
    )
    image_paths = iter_image_paths(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in input directory: {input_dir}")

    images_dir, labels_dir = ensure_output_dirs(output_dir)
    vehicle_model = load_yolo_model(vehicle_weights)
    plate_model = load_yolo_model(plate_weights)

    total_images = 0
    total_vehicles = 0
    total_saved_crops = 0
    total_saved_labels = 0
    total_discarded_crops = 0
    total_failures = 0

    for image_path in tqdm(image_paths, desc="Generating LP fine-tune dataset", unit="image"):
        try:
            stats = process_image(
                image_path,
                vehicle_model,
                plate_model,
                images_dir,
                labels_dir,
                vehicle_conf=args.vehicle_conf,
                plate_conf=args.plate_conf,
                imgsz=args.imgsz,
                allowed_vehicle_classes=allowed_vehicle_classes,
                save_negative_crops=args.save_negative_crops,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            total_failures += 1
            tqdm.write(f"[WARN] Skipped {image_path}: {exc}")
            continue

        total_images += 1
        total_vehicles += stats["vehicles"]
        total_saved_crops += stats["saved_crops"]
        total_saved_labels += stats["saved_labels"]
        total_discarded_crops += stats["discarded_crops"]

    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"processed_images={total_images}")
    print(f"vehicle_detections={total_vehicles}")
    print(f"saved_crops={total_saved_crops}")
    print(f"saved_labels={total_saved_labels}")
    print(f"discarded_crops={total_discarded_crops}")
    print(f"failed_images={total_failures}")


if __name__ == "__main__":
    main()
