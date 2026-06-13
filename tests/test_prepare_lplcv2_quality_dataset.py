from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


def _load_script_module():
    path = Path("scripts/prepare_lplcv2_quality_dataset.py")
    spec = importlib.util.spec_from_file_location("prepare_lplcv2_quality_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_lplcv2_converter_writes_four_class_and_binary_crops(tmp_path) -> None:
    convert_dataset = _load_script_module().convert_dataset

    image_root = tmp_path / "images"
    image_root.mkdir()
    image = np.full((32, 64, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(image_root / "frame.jpg"), image)

    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({
            "images": [
                {
                    "file_name": "frame.jpg",
                    "plates": [
                        {"bbox": [4, 6, 24, 18], "legibility": 3},
                        {"bbox": [12, 8, 20, 12], "legibility": "poor"},
                    ],
                }
            ]
        }),
        encoding="utf-8",
    )

    out_dir = tmp_path / "quality"
    summary = convert_dataset(
        annotations,
        image_root,
        out_dir,
        bbox_format="xywh",
    )

    assert summary["counts"] == {"perfect": 1, "poor": 1}
    assert len(list((out_dir / "legibility4" / "perfect").glob("*.jpg"))) == 1
    assert len(list((out_dir / "legibility4" / "poor").glob("*.jpg"))) == 1
    assert len(list((out_dir / "binary" / "suitable").glob("*.jpg"))) == 1
    assert len(list((out_dir / "binary" / "unsuitable").glob("*.jpg"))) == 1


@pytest.mark.unit
def test_lplcv2_converter_supports_filename_keyed_lplcv25_annotations(tmp_path) -> None:
    mod = _load_script_module()

    image_root = tmp_path / "images"
    image_root.mkdir()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[394:420, 281:346] = 220
    cv2.imwrite(str(image_root / "0a0a12e8.jpg"), image)

    annotations = tmp_path / "lplcv2_annotations.json"
    annotations.write_text(
        json.dumps({
            "0a0a12e8.jpg": {
                "cam": 226,
                "time": "night",
                "faulty": False,
                "rain": False,
                "day": 1,
                "anns": [
                    {
                        "ocr": "BEW2I56",
                        "leg": 3,
                        "xy": [281, 394, 346, 394, 346, 420, 281, 420],
                        "car_valid": True,
                        "occ": False,
                    }
                ],
            }
        }),
        encoding="utf-8",
    )

    out_dir = tmp_path / "quality"
    summary = mod.convert_dataset(annotations, image_root, out_dir)

    assert summary["counts"] == {"perfect": 1}
    crops = list((out_dir / "legibility4" / "perfect").glob("*.jpg"))
    assert len(crops) == 1
    crop = cv2.imread(str(crops[0]))
    assert crop.shape[:2] == (26, 65)


@pytest.mark.unit
def test_lplcv2_extract_bbox_supports_xy_polygon() -> None:
    extract_bbox = _load_script_module().extract_bbox

    assert extract_bbox({"xy": [281, 394, 346, 394, 346, 420, 281, 420]}) == (
        281,
        394,
        346,
        420,
    )


@pytest.mark.unit
def test_lplcv2_normalizes_numeric_and_text_legibility_labels() -> None:
    normalize_legibility = _load_script_module().normalize_legibility

    assert normalize_legibility(0) == "illegible"
    assert normalize_legibility("1") == "poor"
    assert normalize_legibility("Good") == "good"
    assert normalize_legibility("unreadable") == "illegible"
