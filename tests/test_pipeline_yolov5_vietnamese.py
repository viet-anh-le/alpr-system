from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from api.core import pipeline_yolov5_vietnamese as vn


@pytest.mark.unit
def test_load_vn_character_names_matches_reference_file() -> None:
    names = vn.load_vn_character_names()
    assert names[:5] == ["0", "1", "2", "3", "4"]
    assert "Z" in names


@pytest.mark.unit
def test_rotate_plate_crop_is_noop_for_zero_angle() -> None:
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    out = vn.rotate_plate_crop(img, 0.0)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


@pytest.mark.unit
def test_update_rotation_alpha_accumulates_on_tilted_chars() -> None:
    # ~15° diagonal — inside (ROTATION_MIN_DEG, ROTATION_MAX_DEG).
    track_box = np.array(
        [
            [10, 15, 20, 25, [[0.9]], [["A"]]],
            [30, 20, 40, 30, [[0.9]], [["B"]]],
            [50, 25, 60, 35, [[0.9]], [["C"]]],
            [70, 30, 80, 40, [[0.9]], [["D"]]],
        ],
        dtype=object,
    )
    alpha = vn.update_rotation_alpha(track_box, 0.0)
    assert alpha != 0.0
    assert vn.ROTATION_MIN_DEG < abs(math.degrees(alpha)) < vn.ROTATION_MAX_DEG


@pytest.mark.unit
def test_get_final_plate_text_two_line_layout() -> None:
    track_box = np.array(
        [
            [5, 5, 15, 15, [[0.9]], [["5"]]],
            [20, 5, 30, 15, [[0.9]], [["9"]]],
            [5, 20, 15, 30, [[0.9]], [["P"]]],
            [20, 20, 30, 30, [[0.9]], [["2"]]],
        ],
        dtype=object,
    )
    text = vn.get_final_plate_text([track_box], [40], [20])
    assert text


@pytest.mark.integration
def test_detect_char_track_box_on_reference_plate_image() -> None:
    from pathlib import Path

    import torch

    from api.core.config import ROOT, YOLOV5_CHAR_CKPT_PATH
    from api.core.ocr_yolov5 import load_yolov5_char_model

    sample = ROOT / "references" / "Character-Time-series-Matching" / "Vietnamese" / "img" / "plate2.jpg"
    if not sample.exists() or not YOLOV5_CHAR_CKPT_PATH.exists():
        pytest.skip("reference plate image or char.pt weights not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_model = load_yolov5_char_model(YOLOV5_CHAR_CKPT_PATH, device=device)
    img = cv2.imread(str(sample))
    assert img is not None

    track_box = vn.detect_char_track_box(
        img,
        char_model=char_model.model,
        char_names=vn.load_vn_character_names(),
        device=device,
    )
    assert track_box is not None
    text = vn.get_final_plate_text([track_box], [img.shape[0]], [img.shape[1]])
    assert len(text) >= 6
