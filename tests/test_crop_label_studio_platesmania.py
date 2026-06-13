from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "crop_label_studio_platesmania.py"
SPEC = importlib.util.spec_from_file_location("crop_label_studio_platesmania", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[30:50, 20:80] = (0, 255, 0)
    assert cv2.imwrite(str(path), image)


@pytest.mark.unit
def test_extract_reviewed_polygon_from_label_studio_export(tmp_path: Path) -> None:
    image_path = tmp_path / "nomer1.jpg"
    _write_image(image_path)
    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps(
            [
                {
                    "data": {"image": f"/data/local-files/?d={image_path}"},
                    "annotations": [
                        {
                            "result": [
                                {
                                    "type": "polygonlabels",
                                    "value": {
                                        "points": [[10, 30], [40, 30], [40, 50], [10, 50]],
                                        "polygonlabels": ["plate"],
                                    },
                                    "original_width": 200,
                                    "original_height": 100,
                                }
                            ]
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    polygons = MODULE.extract_reviewed_polygons(export_path)

    assert len(polygons) == 1
    assert polygons[0].record_id == "nomer1"
    assert polygons[0].label_name == "plate"
    assert polygons[0].points == ((20.0, 30.0), (80.0, 30.0), (80.0, 50.0), (20.0, 50.0))


@pytest.mark.unit
def test_write_reviewed_ocr_and_detection_samples(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "platesmania_vn"
    image_path = dataset_dir / "downloads" / "full_frames" / "nomer1.jpg"
    _write_image(image_path)
    (dataset_dir / "html_pages").mkdir(parents=True)
    (dataset_dir / "html_pages" / "gallery_records.jsonl").write_text(
        (
            '{"record_id":"nomer1","page_url":"https://platesmania.com/vn/gallery",'
            '"detail_url":"https://platesmania.com/vn/nomer1",'
            '"vehicle_image_url":"https://img03.platesmania.com/vehicle.jpg",'
            '"plate_text_raw":"84-L1 293.38","plate_text_normalized":"84-L1 293.38"}\n'
        ),
        encoding="utf-8",
    )
    polygon = MODULE.ReviewedPolygon(
        record_id="nomer1",
        image_path=image_path,
        points=((20.0, 30.0), (80.0, 30.0), (80.0, 50.0), (20.0, 50.0)),
    )
    source_records = MODULE.load_source_records(dataset_dir)

    ocr_image, ocr_label = MODULE.write_ocr_sample(
        dataset_dir,
        polygon,
        source_records["nomer1"].plate_text,
        split="train",
        overwrite=False,
    )
    MODULE.copy_reviewed_detection_label(dataset_dir, polygon, split="train", overwrite=False)

    assert ocr_image.exists()
    saved_crop = cv2.imread(str(ocr_image))
    assert saved_crop is not None
    assert saved_crop.shape[:2] == (22, 66)
    assert ocr_label.read_text(encoding="utf-8") == "84-L1 293.38\n"
    assert (dataset_dir / "detection" / "images" / "train" / "nomer1.jpg").exists()
    detection_label = dataset_dir / "detection" / "labels" / "train" / "nomer1.txt"
    assert len(detection_label.read_text(encoding="utf-8").strip().split()) == 9


@pytest.mark.unit
def test_warp_plate_crop_can_disable_padding() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = MODULE.warp_plate_crop(
        frame,
        ((20.0, 30.0), (80.0, 30.0), (80.0, 50.0), (20.0, 50.0)),
        padding_ratio=0.0,
    )

    assert crop.shape[:2] == (20, 60)


@pytest.mark.unit
def test_detection_label_uses_label_studio_bsd_bsv_class_ids(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "platesmania_vn"
    image_path = dataset_dir / "downloads" / "full_frames" / "nomer1.jpg"
    _write_image(image_path)
    polygon = MODULE.ReviewedPolygon(
        record_id="nomer1",
        image_path=image_path,
        points=((20.0, 30.0), (80.0, 30.0), (80.0, 50.0), (20.0, 50.0)),
        label_name="BSV",
    )

    MODULE.copy_reviewed_detection_label(dataset_dir, polygon, split="train", overwrite=False)

    detection_label = dataset_dir / "detection" / "labels" / "train" / "nomer1.txt"
    assert detection_label.read_text(encoding="utf-8").startswith("1 ")


@pytest.mark.unit
def test_warp_manual_diamond_polygon_is_non_empty() -> None:
    frame = np.zeros((350, 460, 3), dtype=np.uint8)
    points = (
        (399.20915712799166, 221.67013527575445),
        (401.6224973985432, 237.94484911550468),
        (417.8772112382934, 221.67013527575445),
        (413.0905306971904, 206.35275754422476),
    )

    crop = MODULE.warp_plate_crop(frame, points)

    assert crop.size > 0
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
