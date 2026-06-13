from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT / "data" / "raw" / "platesmania_vn"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECTION_CLASS_IDS = {"BSD": 0, "BSV": 1, "PLATE": 0}
DEFAULT_CROP_PADDING_RATIO = 0.05


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    plate_text: str
    vehicle_image_path: Path | None = None


@dataclass(frozen=True)
class ReviewedPolygon:
    record_id: str
    image_path: Path
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    label_name: str = "plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop manually reviewed Platesmania plate polygons exported from Label Studio."
    )
    parser.add_argument("--export-json", type=Path, required=True, help="Label Studio export JSON.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--crop-padding-ratio",
        type=float,
        default=DEFAULT_CROP_PADDING_RATIO,
        help="OCR crop padding per side, relative to the rectified plate width/height.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_plate_text(text: str) -> str:
    return " ".join(text.strip().upper().split())


def safe_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in value).strip("._")
    return safe or "plate"


def record_id_from_path(path: Path) -> str:
    return path.stem


def read_jsonl_records(path: Path) -> dict[str, SourceRecord]:
    records: dict[str, SourceRecord] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            record_id = str(raw.get("record_id", "")).strip()
            plate_text = normalize_plate_text(str(raw.get("plate_text_normalized") or raw.get("plate_text_raw") or ""))
            if not record_id or not plate_text:
                continue
            vehicle_image_path = raw.get("vehicle_image_path")
            records[record_id] = SourceRecord(
                record_id=record_id,
                plate_text=plate_text,
                vehicle_image_path=Path(str(vehicle_image_path)) if vehicle_image_path else None,
            )
    return records


def load_source_records(dataset_dir: Path) -> dict[str, SourceRecord]:
    records: dict[str, SourceRecord] = {}
    for path in (
        dataset_dir / "html_pages" / "gallery_records.jsonl",
        dataset_dir / "manifests" / "records.jsonl",
    ):
        records.update(read_jsonl_records(path))
    return records


def local_files_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.path == "/data/local-files/":
        query_path = parse_qs(parsed.query).get("d", [""])[0]
        if query_path:
            return Path(query_path)
    return Path(value)


def task_image_path(task: dict[str, Any]) -> Path:
    data = task.get("data")
    if not isinstance(data, dict):
        raise ValueError("Label Studio task is missing data object.")
    value = data.get("image") or data.get("img") or data.get("url")
    if not isinstance(value, str) or not value:
        raise ValueError("Label Studio task is missing data.image.")
    return local_files_path(value).expanduser().resolve()


def iter_label_studio_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    for container_name in ("annotations", "predictions"):
        containers = task.get(container_name)
        if not isinstance(containers, list):
            continue
        for container in containers:
            if isinstance(container, dict) and isinstance(container.get("result"), list):
                return [result for result in container["result"] if isinstance(result, dict)]
    return []


def absolute_polygon_points(result: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    value = result.get("value")
    if not isinstance(value, dict):
        return None
    points = value.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return None

    original_width = float(result.get("original_width") or 0)
    original_height = float(result.get("original_height") or 0)
    if original_width <= 0 or original_height <= 0:
        return None

    absolute: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            return None
        absolute.append((float(point[0]) * original_width / 100.0, float(point[1]) * original_height / 100.0))
    return tuple(absolute)


def polygon_label_name(result: dict[str, Any]) -> str:
    value = result.get("value")
    if not isinstance(value, dict):
        return "plate"
    for key in ("polygonlabels", "rectanglelabels", "labels"):
        labels = value.get(key)
        if isinstance(labels, list) and labels:
            return str(labels[0]).strip() or "plate"
    return "plate"


def detection_class_id(label_name: str) -> int:
    normalized = label_name.strip().upper()
    if normalized in DETECTION_CLASS_IDS:
        return DETECTION_CLASS_IDS[normalized]
    raise ValueError(f"Unsupported Label Studio class '{label_name}'. Expected one of: BSD, BSV.")


def extract_reviewed_polygons(export_json: Path) -> list[ReviewedPolygon]:
    tasks = json.loads(export_json.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Label Studio export must be a JSON array.")

    polygons: list[ReviewedPolygon] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        image_path = task_image_path(task)
        record_id = record_id_from_path(image_path)
        for result in iter_label_studio_results(task):
            points = absolute_polygon_points(result)
            if points is None:
                continue
            polygons.append(
                ReviewedPolygon(
                    record_id=record_id,
                    image_path=image_path,
                    points=points,  # type: ignore[arg-type]
                    label_name=polygon_label_name(result),
                )
            )
    return polygons


def order_points(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)].astype(np.float32)

    edge_lengths = [
        float(np.linalg.norm(ordered[(index + 1) % 4] - ordered[index]))
        for index in range(4)
    ]
    top_edge_index = int(np.argmax(edge_lengths))
    opposite_index = (top_edge_index + 2) % 4
    top_mid_y = (ordered[top_edge_index][1] + ordered[(top_edge_index + 1) % 4][1]) / 2.0
    opposite_mid_y = (ordered[opposite_index][1] + ordered[(opposite_index + 1) % 4][1]) / 2.0
    if opposite_mid_y < top_mid_y:
        top_edge_index = opposite_index

    top = [ordered[top_edge_index], ordered[(top_edge_index + 1) % 4]]
    bottom = [ordered[(top_edge_index + 2) % 4], ordered[(top_edge_index + 3) % 4]]
    top_left, top_right = sorted(top, key=lambda point: point[0])
    bottom_left, bottom_right = sorted(bottom, key=lambda point: point[0])
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def padding_pixels(size: int, padding_ratio: float) -> int:
    if padding_ratio < 0:
        raise ValueError("Crop padding ratio must be non-negative.")
    if padding_ratio == 0:
        return 0
    return max(1, int(round(size * padding_ratio)))


def warp_plate_crop(
    frame: np.ndarray,
    points: tuple[tuple[float, float], ...],
    *,
    padding_ratio: float = DEFAULT_CROP_PADDING_RATIO,
) -> np.ndarray:
    src = order_points(np.array(points, dtype=np.float32).reshape(4, 2))
    tl, tr, br, bl = src
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    pad_x = padding_pixels(width, padding_ratio)
    pad_y = padding_pixels(height, padding_ratio)
    output_width = width + (2 * pad_x)
    output_height = height + (2 * pad_y)
    dst = np.array(
        [
            [pad_x, pad_y],
            [pad_x + width - 1, pad_y],
            [pad_x + width - 1, pad_y + height - 1],
            [pad_x, pad_y + height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (output_width, output_height))


def write_ocr_sample(
    dataset_dir: Path,
    polygon: ReviewedPolygon,
    plate_text: str,
    *,
    split: str,
    overwrite: bool,
    crop_padding_ratio: float = DEFAULT_CROP_PADDING_RATIO,
) -> tuple[Path, Path]:
    frame = cv2.imread(str(polygon.image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {polygon.image_path}")
    crop = warp_plate_crop(frame, polygon.points, padding_ratio=crop_padding_ratio)
    if crop.size == 0:
        raise ValueError(f"Manual polygon produced empty crop: {polygon.record_id}")

    images_dir = dataset_dir / "ocr" / "images" / split
    labels_dir = dataset_dir / "ocr" / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"{safe_stem(polygon.record_id)}.jpg"
    label_path = labels_dir / f"{safe_stem(polygon.record_id)}.txt"
    if not overwrite and (image_path.exists() or label_path.exists()):
        raise FileExistsError(f"OCR sample already exists for {polygon.record_id}; use --overwrite.")
    if not cv2.imwrite(str(image_path), crop):
        raise RuntimeError(f"Could not write crop: {image_path}")
    label_path.write_text(f"{plate_text}\n", encoding="utf-8")
    return image_path, label_path


def copy_reviewed_detection_label(
    dataset_dir: Path,
    polygon: ReviewedPolygon,
    *,
    split: str,
    overwrite: bool,
) -> None:
    source_image = polygon.image_path
    image = cv2.imread(str(source_image))
    if image is None:
        raise ValueError(f"Could not read image: {source_image}")
    height, width = image.shape[:2]
    normalized = []
    for x, y in polygon.points:
        normalized.extend([min(max(x, 0.0), width) / width, min(max(y, 0.0), height) / height])
    line = f"{detection_class_id(polygon.label_name)} " + " ".join(f"{value:.6f}" for value in normalized)

    images_dir = dataset_dir / "detection" / "images" / split
    labels_dir = dataset_dir / "detection" / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    target_image = images_dir / source_image.name
    target_label = labels_dir / f"{source_image.stem}.txt"
    if not overwrite and (target_image.exists() or target_label.exists()):
        raise FileExistsError(f"Detection sample already exists for {polygon.record_id}; use --overwrite.")
    shutil.copy2(source_image, target_image)
    target_label.write_text(line + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    source_records = load_source_records(dataset_dir)
    polygons = extract_reviewed_polygons(args.export_json.expanduser().resolve())
    if not polygons:
        raise SystemExit("No 4-point polygons found in Label Studio export.")

    promoted = 0
    skipped = 0
    for polygon in polygons:
        source_record = source_records.get(polygon.record_id)
        if source_record is None:
            print(f"[WARN] Missing source label for {polygon.record_id}; skipped")
            skipped += 1
            continue
        write_ocr_sample(
            dataset_dir,
            polygon,
            source_record.plate_text,
            split=args.output_split,
            overwrite=args.overwrite,
            crop_padding_ratio=args.crop_padding_ratio,
        )
        copy_reviewed_detection_label(
            dataset_dir,
            polygon,
            split=args.output_split,
            overwrite=args.overwrite,
        )
        promoted += 1

    print(f"promoted={promoted}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
