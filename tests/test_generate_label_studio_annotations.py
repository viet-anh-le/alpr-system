from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "generate_label_studio_annotations.py"
)
SPEC = importlib.util.spec_from_file_location("generate_label_studio_annotations", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_image(path: Path, width: int = 100, height: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


@pytest.mark.unit
def test_build_local_files_url_uses_absolute_path() -> None:
    image_path = Path("data/lp_finetune_vehicle_crops/images/frame_0001_vehicle_0000.jpg").resolve()

    url = MODULE.build_local_files_url(image_path)

    assert url == f"/data/local-files/?d={image_path}"


@pytest.mark.unit
def test_convert_points_to_label_studio_percentages() -> None:
    points = [[192.0, 108.0], [384.0, 108.0], [384.0, 216.0], [192.0, 216.0]]

    normalized = MODULE.convert_points_to_label_studio(points, image_width=1920, image_height=1080)

    np.testing.assert_allclose(
        normalized,
        [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
    )


@pytest.mark.unit
def test_denormalize_yolo_points_restores_absolute_coordinates() -> None:
    points = MODULE.denormalize_yolo_points(
        [0.1, 0.4, 0.3, 0.4, 0.3, 0.8, 0.1, 0.8],
        image_width=100,
        image_height=50,
    )

    np.testing.assert_allclose(
        points,
        [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
    )


@pytest.mark.unit
def test_parse_yolo_obb_line_maps_class_name_and_points() -> None:
    detection = MODULE.parse_yolo_obb_line(
        "1 0.1 0.4 0.3 0.4 0.3 0.8 0.1 0.8",
        image_width=100,
        image_height=50,
    )

    assert detection["label"] == "BSV"
    np.testing.assert_allclose(
        detection["points"],
        [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
    )


@pytest.mark.unit
def test_build_task_keeps_empty_predictions() -> None:
    image_path = Path("data/lp_finetune_vehicle_crops/images/frame_0002_vehicle_0000.jpg").resolve()

    task = MODULE.build_task(image_path=image_path, image_width=640, image_height=480, detections=[])

    assert task == {
        "data": {"image": f"/data/local-files/?d={image_path}"},
        "predictions": [{"result": []}],
    }


@pytest.mark.unit
def test_build_task_from_paths_supports_nested_train_val_layout(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_path = images_dir / "train" / "sample.jpg"
    label_path = labels_dir / "train" / "sample.txt"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.zeros((50, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    label_path.write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n", encoding="utf-8")

    task = MODULE.build_task_from_paths(
        image_path=image_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
    )

    assert task["data"]["image"].endswith("/images/train/sample.jpg")
    assert len(task["predictions"][0]["result"]) == 1
    assert task["predictions"][0]["result"][0]["value"]["polygonlabels"] == ["BSD"]


@pytest.mark.unit
def test_prepare_platesmania_review_skips_already_promoted_ocr_labels(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "platesmania_vn"
    review_dir = dataset_dir / "review" / "pending_review"
    write_image(review_dir / "new_case.jpg")
    write_image(review_dir / "done_case.jpg")
    labels_dir = dataset_dir / "ocr" / "labels" / "train"
    labels_dir.mkdir(parents=True)
    (labels_dir / "done_case.txt").write_text("50E-190.54\n", encoding="utf-8")

    images_dir, output_labels_dir, copied, skipped = MODULE.prepare_platesmania_review_images(
        dataset_dir,
        dataset_dir / "label_studio_review",
        include_labeled=False,
        overwrite=True,
    )

    assert copied == 1
    assert skipped == 1
    assert (images_dir / "new_case.jpg").exists()
    assert not (images_dir / "done_case.jpg").exists()
    assert output_labels_dir.exists()
