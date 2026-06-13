from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTION_DIR = ROOT / "data" / "datasets" / "detection"
DEFAULT_LP_FINETUNE_DIR = ROOT / "data" / "lp_finetune_vehicle_crops"
DEFAULT_LSV_DIR = ROOT / "data" / "raw" / "LSV-LP_validation"
DEFAULT_UFPR_DIR = ROOT / "data" / "raw" / "UFPR-ALPR" / "UFPR-ALPR dataset"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "datasets" / "lp_detection_obb"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "BSD", 1: "BSV"}


class DatasetStats:
    def __init__(
        self, copied_images: int = 0, generated_images: int = 0, skipped_items: int = 0
    ) -> None:
        self.copied_images = copied_images
        self.generated_images = generated_images
        self.skipped_items = skipped_items

    def __add__(self, other: "DatasetStats") -> "DatasetStats":
        return DatasetStats(
            copied_images=self.copied_images + other.copied_images,
            generated_images=self.generated_images + other.generated_images,
            skipped_items=self.skipped_items + other.skipped_items,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged YOLOv8 OBB license plate detection dataset."
    )
    parser.add_argument("--detection-dir", type=Path, default=DEFAULT_DETECTION_DIR)
    parser.add_argument("--lp-finetune-dir", type=Path, default=DEFAULT_LP_FINETUNE_DIR)
    parser.add_argument("--lsv-dir", type=Path, default=DEFAULT_LSV_DIR)
    parser.add_argument("--ufpr-dir", type=Path, default=DEFAULT_UFPR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lpft-val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate only the final output dataset directory.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image has invalid dimensions: {image_path}")
    return image


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


def clamp_points(points: Iterable[float], image_width: int, image_height: int) -> list[float]:
    raw_points = list(points)
    if len(raw_points) != 8:
        raise ValueError(f"OBB points must have 8 values, received {len(raw_points)}")

    clamped_points: list[float] = []
    for index, value in enumerate(raw_points):
        upper_bound = image_width if index % 2 == 0 else image_height
        clamped_points.append(float(min(max(value, 0.0), float(upper_bound))))
    return clamped_points


def normalize_points(points: Iterable[float], image_width: int, image_height: int) -> list[float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive for normalization.")

    normalized: list[float] = []
    for index, value in enumerate(clamp_points(points, image_width, image_height)):
        normalized.append(value / image_width if index % 2 == 0 else value / image_height)
    return normalized


def build_yolo_obb_line(
    class_id: int, points: Iterable[float], image_width: int, image_height: int
) -> str:
    if class_id not in CLASS_NAMES:
        raise ValueError(f"Unsupported class id: {class_id}")

    normalized_points = normalize_points(points, image_width=image_width, image_height=image_height)
    coordinate_tokens = " ".join(f"{value:.6f}" for value in normalized_points)
    return f"{class_id} {coordinate_tokens}"


def validate_yolo_obb_line(line: str) -> None:
    tokens = line.strip().split()
    if len(tokens) != 9:
        raise ValueError(f"Expected 9 YOLO OBB tokens, received {len(tokens)}: {line}")

    try:
        class_id = int(tokens[0])
    except ValueError as exc:
        raise ValueError(f"Class id must be an integer: {line}") from exc

    if class_id not in CLASS_NAMES:
        raise ValueError(f"Unsupported class id {class_id}: {line}")

    for token in tokens[1:]:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Coordinate must be a float: {line}") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Coordinate out of [0, 1] range: {line}")


def validate_label_text(label_text: str, label_path: Path | None = None) -> None:
    for raw_line in label_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            validate_yolo_obb_line(stripped)
        except ValueError as exc:
            source = f" in {label_path}" if label_path is not None else ""
            raise ValueError(f"Invalid YOLO OBB label{source}: {exc}") from exc


def canonicalize_yolo_label_line(line: str, image_width: int, image_height: int) -> str:
    tokens = line.strip().split()
    if len(tokens) < 9:
        raise ValueError(f"Expected at least 9 YOLO polygon tokens, received {len(tokens)}: {line}")
    if (len(tokens) - 1) % 2 != 0:
        raise ValueError(f"YOLO polygon coordinates must be x/y pairs: {line}")

    try:
        class_id = int(tokens[0])
    except ValueError as exc:
        raise ValueError(f"Class id must be an integer: {line}") from exc
    if class_id not in CLASS_NAMES:
        raise ValueError(f"Unsupported class id {class_id}: {line}")

    try:
        normalized_points = [float(token) for token in tokens[1:]]
    except ValueError as exc:
        raise ValueError(f"Coordinates must be floats: {line}") from exc

    if len(normalized_points) == 8:
        absolute_points: list[float] = []
        for index in range(0, len(normalized_points), 2):
            absolute_points.extend(
                [
                    normalized_points[index] * image_width,
                    normalized_points[index + 1] * image_height,
                ]
            )
        return build_yolo_obb_line(
            class_id=class_id,
            points=absolute_points,
            image_width=image_width,
            image_height=image_height,
        )

    polygon = np.array(
        [
            [normalized_points[index] * image_width, normalized_points[index + 1] * image_height]
            for index in range(0, len(normalized_points), 2)
        ],
        dtype=np.float32,
    )
    rect = cv2.minAreaRect(polygon)
    obb_points = cv2.boxPoints(rect).reshape(-1).tolist()
    return build_yolo_obb_line(
        class_id=class_id,
        points=obb_points,
        image_width=image_width,
        image_height=image_height,
    )


def canonicalize_label_text(label_text: str, image_path: Path, label_path: Path) -> str:
    image = read_image(image_path)
    image_height, image_width = image.shape[:2]
    canonical_lines: list[str] = []
    for raw_line in label_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            canonical_lines.append(
                canonicalize_yolo_label_line(
                    stripped,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label in {label_path}: {exc}") from exc
    return "\n".join(canonical_lines)


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    matches = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0] if matches else None


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dir}")
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_text = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: BSD\n"
        "  1: BSV\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def copy_pair(
    image_path: Path, label_path: Path, output_dir: Path, split: str, output_stem: str
) -> None:
    label_text = label_path.read_text(encoding="utf-8")
    label_text = canonicalize_label_text(label_text, image_path=image_path, label_path=label_path)
    validate_label_text(label_text, label_path=label_path)

    image_output_path = output_dir / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
    label_output_path = output_dir / "labels" / split / f"{output_stem}.txt"
    if image_output_path.exists() or label_output_path.exists():
        raise FileExistsError(f"Output collision for stem: {output_stem}")

    shutil.copy2(image_path, image_output_path)
    label_output_path.write_text(
        label_text.rstrip() + ("\n" if label_text.strip() else ""), encoding="utf-8"
    )


def write_generated_sample(
    image: np.ndarray, label_lines: list[str], output_dir: Path, split: str, output_stem: str
) -> None:
    label_text = "\n".join(label_lines)
    validate_label_text(label_text)

    image_output_path = output_dir / "images" / split / f"{output_stem}.jpg"
    label_output_path = output_dir / "labels" / split / f"{output_stem}.txt"
    if image_output_path.exists() or label_output_path.exists():
        raise FileExistsError(f"Output collision for stem: {output_stem}")
    if not cv2.imwrite(str(image_output_path), image):
        raise RuntimeError(f"Failed to write image: {image_output_path}")
    label_output_path.write_text(label_text + "\n", encoding="utf-8")


def build_detection_source(detection_dir: Path, output_dir: Path) -> DatasetStats:
    stats = DatasetStats()
    for split in ("train", "val"):
        images_dir = detection_dir / "images" / split
        labels_dir = detection_dir / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Missing detection split directories for: {split}")

        image_paths = sorted(path for path in images_dir.iterdir() if is_image_file(path))
        for image_path in tqdm(image_paths, desc=f"Copying detection {split}", unit="image"):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing detection label for image: {image_path}")
            copy_pair(
                image_path=image_path,
                label_path=label_path,
                output_dir=output_dir,
                split=split,
                output_stem=f"detection_{image_path.stem}",
            )
            stats += DatasetStats(copied_images=1)
    return stats


def split_pairs_deterministically(
    pairs: list[tuple[Path, Path]], val_ratio: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("--lpft-val-ratio must be between 0 and 1")

    shuffled_pairs = list(pairs)
    random.Random(seed).shuffle(shuffled_pairs)
    val_count = int(round(len(shuffled_pairs) * val_ratio))
    val_pairs = shuffled_pairs[:val_count]
    train_pairs = shuffled_pairs[val_count:]
    return train_pairs, val_pairs


def build_lp_finetune_source(
    lp_finetune_dir: Path, output_dir: Path, val_ratio: float, seed: int
) -> DatasetStats:
    images_dir = lp_finetune_dir / "images"
    manual_labels_dir = lp_finetune_dir / "labels_LP_Finetune"
    if not images_dir.exists() or not manual_labels_dir.exists():
        raise FileNotFoundError("LP fine-tune source must contain images/ and labels_LP_Finetune/")

    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for label_path in sorted(manual_labels_dir.glob("*.txt")):
        image_path = find_image_for_stem(images_dir, label_path.stem)
        if image_path is None:
            skipped += 1
            continue
        pairs.append((image_path, label_path))

    train_pairs, val_pairs = split_pairs_deterministically(pairs, val_ratio=val_ratio, seed=seed)
    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in tqdm(split_pairs, desc=f"Copying lpft {split}", unit="image"):
            copy_pair(
                image_path=image_path,
                label_path=label_path,
                output_dir=output_dir,
                split=split,
                output_stem=f"lpft_{image_path.stem}",
            )

    return DatasetStats(copied_images=len(pairs), skipped_items=skipped)


def points_to_flat(points: Iterable[Iterable[float]]) -> list[float]:
    flat: list[float] = []
    for point in points:
        point_values = list(point)
        if len(point_values) != 2:
            raise ValueError(f"Expected point with 2 coordinates, received: {point_values}")
        flat.extend([float(point_values[0]), float(point_values[1])])
    if len(flat) != 8:
        raise ValueError(f"Expected four points, received {len(flat) // 2}")
    return flat


def normalized_box_to_absolute(
    normalized_box: list[list[float]], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    if len(normalized_box) != 2:
        raise ValueError(f"Expected carBox with 2 points, received: {normalized_box}")
    (x1_norm, y1_norm), (x2_norm, y2_norm) = normalized_box
    return (
        float(x1_norm) * image_width,
        float(y1_norm) * image_height,
        float(x2_norm) * image_width,
        float(y2_norm) * image_height,
    )


def normalized_poly_to_absolute(
    normalized_poly: list[list[float]], image_width: int, image_height: int
) -> list[float]:
    absolute_points: list[float] = []
    for x_norm, y_norm in normalized_poly:
        absolute_points.extend([float(x_norm) * image_width, float(y_norm) * image_height])
    if len(absolute_points) != 8:
        raise ValueError(f"Expected licPoly with four points, received {len(absolute_points) // 2}")
    return absolute_points


def offset_points(points: Iterable[float], offset_x: float, offset_y: float) -> list[float]:
    offset: list[float] = []
    raw_points = list(points)
    if len(raw_points) != 8:
        raise ValueError(f"Expected 8 point values, received {len(raw_points)}")

    for index in range(0, len(raw_points), 2):
        offset.extend([raw_points[index] - offset_x, raw_points[index + 1] - offset_y])
    return offset


def build_lsv_source(lsv_dir: Path, output_dir: Path) -> DatasetStats:
    jsons_dir = lsv_dir / "jsons"
    videos_dir = lsv_dir / "videos"
    if not jsons_dir.exists() or not videos_dir.exists():
        raise FileNotFoundError("LSV source must contain jsons/ and videos/")

    generated = 0
    skipped = 0
    zip_paths = sorted(jsons_dir.glob("*.zip"))
    for zip_path in tqdm(zip_paths, desc="Processing LSV zip", unit="zip"):
        scenario = zip_path.stem
        with zipfile.ZipFile(zip_path) as archive:
            json_names = sorted(name for name in archive.namelist() if name.endswith(".json"))
            for json_name in json_names:
                parts = Path(json_name).parts
                if len(parts) < 3:
                    skipped += 1
                    continue

                video_id = parts[-2]
                frame_stem = Path(parts[-1]).stem
                image_path = videos_dir / scenario / scenario / video_id / f"{frame_stem}.jpg"
                if not image_path.exists():
                    skipped += 1
                    continue

                annotations = json.loads(archive.read(json_name).decode("utf-8"))
                image = read_image(image_path)
                image_height, image_width = image.shape[:2]
                for object_id, annotation in sorted(annotations.items(), key=lambda item: item[0]):
                    try:
                        car_box = normalized_box_to_absolute(
                            annotation["carBox"], image_width=image_width, image_height=image_height
                        )
                        plate_points = normalized_poly_to_absolute(
                            annotation["licPoly"], image_width=image_width, image_height=image_height
                        )
                        crop, clipped_box = crop_image(image, car_box)
                    except (KeyError, TypeError, ValueError):
                        skipped += 1
                        continue

                    crop_height, crop_width = crop.shape[:2]
                    x1, y1, _, _ = clipped_box
                    crop_points = offset_points(plate_points, offset_x=x1, offset_y=y1)
                    label_line = build_yolo_obb_line(
                        class_id=0,
                        points=crop_points,
                        image_width=crop_width,
                        image_height=crop_height,
                    )
                    output_stem = f"lsv_{scenario}_{video_id}_{frame_stem}_{object_id}"
                    write_generated_sample(crop, [label_line], output_dir, "train", output_stem)
                    generated += 1

    return DatasetStats(generated_images=generated, skipped_items=skipped)


def parse_ufpr_annotation(annotation_path: Path) -> tuple[tuple[float, float, float, float], list[float], str]:
    vehicle_box: tuple[float, float, float, float] | None = None
    plate_points: list[float] | None = None
    vehicle_type = ""

    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("position_vehicle:"):
            values = [float(value) for value in stripped.split(":", 1)[1].split()]
            if len(values) != 4:
                raise ValueError(f"Invalid position_vehicle in {annotation_path}: {stripped}")
            x, y, width, height = values
            vehicle_box = (x, y, x + width, y + height)
        elif stripped.startswith("type:"):
            vehicle_type = stripped.split(":", 1)[1].strip().lower()
        elif stripped.startswith("corners:"):
            point_tokens = stripped.split(":", 1)[1].split()
            points: list[float] = []
            for token in point_tokens:
                x_text, y_text = token.split(",", 1)
                points.extend([float(x_text), float(y_text)])
            if len(points) != 8:
                raise ValueError(f"Invalid corners in {annotation_path}: {stripped}")
            plate_points = points

    if vehicle_box is None:
        raise ValueError(f"Missing position_vehicle in {annotation_path}")
    if plate_points is None:
        raise ValueError(f"Missing corners in {annotation_path}")

    return vehicle_box, plate_points, vehicle_type


def ufpr_class_id(vehicle_type: str) -> int:
    normalized_type = vehicle_type.lower()
    return 1 if "motor" in normalized_type or "motorcycle" in normalized_type or "motorbike" in normalized_type else 0


def build_ufpr_source(ufpr_dir: Path, output_dir: Path) -> DatasetStats:
    if not ufpr_dir.exists():
        raise FileNotFoundError(f"UFPR source not found: {ufpr_dir}")

    generated = 0
    skipped = 0
    annotation_paths = sorted(
        path
        for split_dir in ("training", "validation", "testing")
        for path in (ufpr_dir / split_dir).rglob("*.txt")
    )
    for annotation_path in tqdm(annotation_paths, desc="Processing UFPR", unit="annotation"):
        image_path = annotation_path.with_suffix(".png")
        if not image_path.exists():
            skipped += 1
            continue

        try:
            image = read_image(image_path)
            vehicle_box, plate_points, vehicle_type = parse_ufpr_annotation(annotation_path)
            crop, clipped_box = crop_image(image, vehicle_box)
        except ValueError:
            skipped += 1
            continue

        crop_height, crop_width = crop.shape[:2]
        x1, y1, _, _ = clipped_box
        crop_points = offset_points(plate_points, offset_x=x1, offset_y=y1)
        label_line = build_yolo_obb_line(
            class_id=ufpr_class_id(vehicle_type),
            points=crop_points,
            image_width=crop_width,
            image_height=crop_height,
        )
        rel_parts = annotation_path.relative_to(ufpr_dir).with_suffix("").parts
        output_stem = "ufpr_" + "_".join(rel_parts).replace("[", "_").replace("]", "")
        write_generated_sample(crop, [label_line], output_dir, "train", output_stem)
        generated += 1

    return DatasetStats(generated_images=generated, skipped_items=skipped)


def validate_input_dirs(
    detection_dir: Path, lp_finetune_dir: Path, lsv_dir: Path, ufpr_dir: Path
) -> None:
    for source_dir in (detection_dir, lp_finetune_dir, lsv_dir, ufpr_dir):
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {source_dir}")


def build_dataset(
    *,
    detection_dir: Path,
    lp_finetune_dir: Path,
    lsv_dir: Path,
    ufpr_dir: Path,
    output_dir: Path,
    lpft_val_ratio: float,
    seed: int,
    overwrite: bool,
) -> DatasetStats:
    validate_input_dirs(
        detection_dir=detection_dir,
        lp_finetune_dir=lp_finetune_dir,
        lsv_dir=lsv_dir,
        ufpr_dir=ufpr_dir,
    )
    prepare_output_dir(output_dir, overwrite=overwrite)

    stats = DatasetStats()
    stats += build_detection_source(detection_dir=detection_dir, output_dir=output_dir)
    stats += build_lp_finetune_source(
        lp_finetune_dir=lp_finetune_dir,
        output_dir=output_dir,
        val_ratio=lpft_val_ratio,
        seed=seed,
    )
    stats += build_lsv_source(lsv_dir=lsv_dir, output_dir=output_dir)
    stats += build_ufpr_source(ufpr_dir=ufpr_dir, output_dir=output_dir)
    write_dataset_yaml(output_dir)
    return stats


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        detection_dir=args.detection_dir.expanduser().resolve(),
        lp_finetune_dir=args.lp_finetune_dir.expanduser().resolve(),
        lsv_dir=args.lsv_dir.expanduser().resolve(),
        ufpr_dir=args.ufpr_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        lpft_val_ratio=args.lpft_val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"output_dir={args.output_dir.expanduser().resolve()}")
    print(f"copied_images={stats.copied_images}")
    print(f"generated_images={stats.generated_images}")
    print(f"skipped_items={stats.skipped_items}")


if __name__ == "__main__":
    main()
