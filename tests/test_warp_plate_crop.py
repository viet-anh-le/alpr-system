from __future__ import annotations
import numpy as np
from api.core.video_processor import warp_plate_crop


def _frame() -> np.ndarray:
    return np.zeros((300, 400, 3), dtype=np.uint8)


def test_axis_aligned_rect_returns_correct_dimensions():
    # Axis-aligned box [x=50..150, y=60..80] → width≈100, height≈20
    pts = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert abs(result.shape[1] - 100) <= 2
    assert abs(result.shape[0] - 20) <= 2


def test_tilted_box_returns_non_empty():
    pts = np.array([[100, 50], [140, 40], [145, 60], [105, 70]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.size > 0
    assert result.ndim == 3


def test_degenerate_pts_returns_empty():
    pts = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.size == 0


def test_order_invariant():
    # Same rect given in reverse winding order → same output shape
    pts_a = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    pts_b = np.array([[150, 80], [50, 80], [50, 60], [150, 60]], dtype=np.int32)
    a = warp_plate_crop(_frame(), pts_a)
    b = warp_plate_crop(_frame(), pts_b)
    assert a.shape == b.shape


def test_output_dtype_is_uint8():
    pts = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.dtype == np.uint8


def test_pixel_values_preserved():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[60:80, 50:150] = (0, 255, 0)   # green plate region
    pts = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    result = warp_plate_crop(frame, pts)
    assert result[:, :, 1].mean() > 200   # green channel dominates
