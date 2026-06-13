from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data" / "lp_finetune_vehicle_crops"
DEFAULT_OUTPUT = ROOT / "data" / "lp_finetune_vehicle_crops" / "ls_auto_annotations.json"
DEFAULT_CLASS_NAMES = {0: "BSD", 1: "BSV"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PLATESMANIA_DIR = ROOT / "data" / "raw" / "platesmania_vn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Label Studio import JSON from YOLO OBB labels."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Dataset root containing images/ and labels/. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the generated Label Studio JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--platesmania-review",
        action="store_true",
        help="Build import JSON from Platesmania pending_review, excluding records already promoted to OCR labels.",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="With --platesmania-review, include records that already have OCR labels.",
    )
    parser.add_argument(
        "--review-output-dir",
        type=Path,
        default=None,
        help="With --platesmania-review, directory where filtered images/ and labels/ are prepared.",
    )
    parser.add_argument(
        "--overwrite-review",
        action="store_true",
        help="With --platesmania-review, recreate the prepared review output directory.",
    )
    return parser.parse_args()


def iter_image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_local_files_url(image_path: Path) -> str:
    return f"/data/local-files/?d={image_path.resolve()}"


def convert_points_to_label_studio(
    points: list[list[float]], image_width: int, image_height: int
) -> list[list[float]]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    return [
        [(float(x) / image_width) * 100.0, (float(y) / image_height) * 100.0]
        for x, y in points
    ]


def denormalize_yolo_points(
    normalized_points: list[float], image_width: int, image_height: int
) -> list[list[float]]:
    if len(normalized_points) != 8:
        raise ValueError(f"Expected 8 coordinate values, received {len(normalized_points)}")

    absolute_points: list[list[float]] = []
    for index in range(0, len(normalized_points), 2):
        absolute_points.append(
            [
                normalized_points[index] * image_width,
                normalized_points[index + 1] * image_height,
            ]
        )

    return absolute_points


def parse_yolo_obb_line(line: str, image_width: int, image_height: int) -> dict[str, Any]:
    tokens = line.strip().split()
    if not tokens:
        raise ValueError("Label line is empty.")
    if len(tokens) != 9:
        raise ValueError(f"Expected 9 tokens in YOLO OBB label line, received {len(tokens)}: {line}")

    class_id = int(tokens[0])
    normalized_points = [float(value) for value in tokens[1:]]
    return {
        "label": DEFAULT_CLASS_NAMES.get(class_id, str(class_id)),
        "points": denormalize_yolo_points(
            normalized_points, image_width=image_width, image_height=image_height
        ),
    }


def read_yolo_detections(
    label_path: Path, image_width: int, image_height: int
) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []

    detections: list[dict[str, Any]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        detections.append(
            parse_yolo_obb_line(
                stripped, image_width=image_width, image_height=image_height
            )
        )
    return detections


def generate_result_id() -> str:
    token = uuid.uuid4().hex[:10]
    return f"{token[:8]}-{token[8:]}"


def build_prediction_result(
    points: list[list[float]],
    label: str,
    image_width: int,
    image_height: int,
    result_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": result_id or generate_result_id(),
        "type": "polygonlabels",
        "value": {
            "points": convert_points_to_label_studio(
                points, image_width=image_width, image_height=image_height
            ),
            "polygonlabels": [label],
        },
        "to_name": "image",
        "from_name": "label",
        "original_width": image_width,
        "original_height": image_height,
    }


def build_task(
    image_path: Path, image_width: int, image_height: int, detections: list[dict[str, Any]]
) -> dict[str, Any]:
    results = [
        build_prediction_result(
            points=detection["points"],
            label=detection["label"],
            image_width=image_width,
            image_height=image_height,
        )
        for detection in detections
    ]

    return {
        "data": {"image": build_local_files_url(image_path)},
        "predictions": [{"result": results}],
    }


def existing_ocr_label_stems(dataset_dir: Path) -> set[str]:
    stems: set[str] = set()
    for split in ("train", "val"):
        labels_dir = dataset_dir / "ocr" / "labels" / split
        if labels_dir.exists():
            stems.update(path.stem for path in labels_dir.glob("*.txt"))
    return stems


def prepare_platesmania_review_images(
    dataset_dir: Path,
    output_dir: Path,
    *,
    include_labeled: bool,
    overwrite: bool,
) -> tuple[Path, Path, int, int]:
    review_dir = dataset_dir / "review" / "pending_review"
    if not review_dir.exists():
        raise FileNotFoundError(f"Platesmania review directory not found: {review_dir}")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    labeled = existing_ocr_label_stems(dataset_dir)
    copied = 0
    skipped_labeled = 0
    for source_image in iter_image_paths(review_dir):
        if source_image.stem in labeled and not include_labeled:
            skipped_labeled += 1
            continue
        target_image = images_dir / source_image.name
        if not target_image.exists() or overwrite:
            shutil.copy2(source_image, target_image)
        copied += 1

    return images_dir, labels_dir, copied, skipped_labeled


def build_task_from_paths(image_path: Path, images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]
    image_relative_path = image_path.relative_to(images_dir)
    label_path = labels_dir / image_relative_path.with_suffix(".txt")
    detections = read_yolo_detections(
        label_path, image_width=image_width, image_height=image_height
    )

    return build_task(
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        detections=detections,
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if args.platesmania_review:
        dataset_dir = input_dir if input_dir != DEFAULT_INPUT_DIR.resolve() else DEFAULT_PLATESMANIA_DIR
        dataset_dir = dataset_dir.expanduser().resolve()
        review_output_dir = (
            args.review_output_dir.expanduser().resolve()
            if args.review_output_dir is not None
            else dataset_dir / "label_studio_review"
        )
        if args.output == DEFAULT_OUTPUT:
            output_path = review_output_dir / "import.json"
        images_dir, labels_dir, copied, skipped_labeled = prepare_platesmania_review_images(
            dataset_dir,
            review_output_dir,
            include_labeled=args.include_labeled,
            overwrite=args.overwrite_review,
        )
        image_paths = iter_image_paths(images_dir)
        tasks = [
            build_task_from_paths(
                image_path=image_path,
                images_dir=images_dir,
                labels_dir=labels_dir,
            )
            for image_path in tqdm(image_paths, desc="Generating Platesmania review tasks")
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        print(f"saved_json={output_path}")
        print(f"tasks={len(tasks)}")
        print(f"copied_images={copied}")
        print(f"skipped_labeled={skipped_labeled}")
        return

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_paths = iter_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {images_dir}")

    tasks = [
        build_task_from_paths(
            image_path=image_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
        )
        for image_path in tqdm(image_paths, desc="Generating Label Studio tasks")
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"saved_json={output_path}")
    print(f"tasks={len(tasks)}")


if __name__ == "__main__":
    main()
