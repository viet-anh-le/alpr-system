from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_lp_finetune_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_lp_finetune_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.unit
def test_clip_bbox_to_image_rejects_invalid_boxes() -> None:
    assert MODULE.clip_bbox_to_image((20, 10, 20, 40), image_width=100, image_height=80) is None
    assert MODULE.clip_bbox_to_image((90, 10, 80, 40), image_width=100, image_height=80) is None
    assert MODULE.clip_bbox_to_image((10, 70, 40, 70), image_width=100, image_height=80) is None


@pytest.mark.unit
def test_clip_bbox_to_image_clamps_out_of_bounds_values() -> None:
    clipped = MODULE.clip_bbox_to_image((-10, 5, 120, 90), image_width=100, image_height=80)
    assert clipped == (0, 5, 100, 80)


@pytest.mark.unit
def test_crop_image_returns_clipped_crop() -> None:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    crop, clipped = MODULE.crop_image(image, (-2, 1, 4, 10))

    assert clipped == (0, 1, 4, 6)
    assert crop.shape == (5, 4, 3)


@pytest.mark.unit
def test_xyxy_box_to_obb_points_returns_clockwise_corners() -> None:
    points = MODULE.xyxy_box_to_obb_points((10.0, 20.0, 30.0, 40.0))
    assert points == [10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 10.0, 40.0]


@pytest.mark.unit
def test_normalize_obb_points_uses_crop_dimensions() -> None:
    normalized = MODULE.normalize_obb_points(
        [20.0, 10.0, 60.0, 10.0, 60.0, 30.0, 20.0, 30.0],
        image_width=80,
        image_height=40,
    )
    assert normalized == pytest.approx([0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75])


@pytest.mark.unit
def test_build_yolo_obb_label_line_prefixes_class_id() -> None:
    label_line = MODULE.build_yolo_obb_label_line(
        class_id=0,
        points=[20.0, 10.0, 60.0, 10.0, 60.0, 30.0, 20.0, 30.0],
        image_width=80,
        image_height=40,
    )
    assert label_line == "0 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000 0.250000 0.750000"
