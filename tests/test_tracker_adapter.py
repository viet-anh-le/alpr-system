"""Unit tests for the BoxMOT-based vehicle tracker adapter.

We test:
  1. Adapter accepts (N, 6) detections and returns (M, 3) (boxes, ids, classes).
  2. IDs persist across frames for a stationary object.
  3. Empty detection input returns three empty arrays without crashing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from api.core.tracker_adapter import VehicleTracker

REID_WEIGHTS = Path("weights/tracking/vehicle_reid.onnx")


@pytest.fixture(scope="module")
def tracker() -> VehicleTracker:
    if not REID_WEIGHTS.exists():
        pytest.skip(f"ReID weights not available at {REID_WEIGHTS}")
    return VehicleTracker(reid_weights=REID_WEIGHTS, device="cpu", half=False)


@pytest.mark.unit
def test_track_returns_three_arrays_with_matching_lengths(tracker: VehicleTracker) -> None:
    # Arrange: one synthetic detection on a blank 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = np.array([[100, 100, 200, 200, 0.95, 2]], dtype=np.float32)  # x1,y1,x2,y2,conf,cls

    # Act
    boxes, ids, classes = tracker.track(dets, frame)

    # Assert
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert len(boxes) == len(ids) == len(classes)


@pytest.mark.unit
def test_track_returns_empty_arrays_for_empty_detections(tracker: VehicleTracker) -> None:
    # Arrange
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = np.zeros((0, 6), dtype=np.float32)

    # Act
    boxes, ids, classes = tracker.track(dets, frame)

    # Assert
    assert boxes.shape == (0, 4)
    assert ids.shape == (0,)
    assert classes.shape == (0,)


@pytest.mark.unit
def test_track_id_persists_for_stationary_object(tracker: VehicleTracker) -> None:
    # Arrange: same detection across three frames — track id must persist
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 128  # give the crop some content for ReID
    dets = np.array([[100, 100, 200, 200, 0.95, 2]], dtype=np.float32)

    # Act
    ids_per_frame: list[int] = []
    for _ in range(3):
        _, ids, _ = tracker.track(dets, frame)
        if len(ids) > 0:
            ids_per_frame.append(int(ids[0]))

    # Assert
    assert len(ids_per_frame) >= 2, "tracker should produce at least 2 confirmed frames"
    assert len(set(ids_per_frame)) == 1, f"id changed: {ids_per_frame}"
