from __future__ import annotations

import numpy as np
import pytest


def _frame(h: int = 120, w: int = 200) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = 128
    return frame


@pytest.mark.unit
def test_expand_vehicle_box_clamps_to_frame_edges() -> None:
    from api.core.cascade_plate import expand_vehicle_box

    expanded = expand_vehicle_box(
        frame_shape=(100, 200, 3),
        box=[2, 4, 42, 44],
        pad_ratio=0.25,
        pad_min=16,
    )

    assert expanded == (0, 0, 58, 60)


@pytest.mark.unit
def test_crop_vehicle_regions_skips_invalid_boxes() -> None:
    from api.core.cascade_plate import crop_vehicle_regions

    crops = crop_vehicle_regions(
        _frame(),
        [
            {"id": 1, "box": [10, 20, 90, 80]},
            {"id": 2, "box": [50, 50, 50, 90]},
        ],
        pad_ratio=0.0,
        pad_min=0,
    )

    assert len(crops) == 1
    assert crops[0].vehicle_id == 1
    assert crops[0].offset == (10, 20)
    assert crops[0].image.shape == (60, 80, 3)


@pytest.mark.unit
def test_map_obb_points_from_crop_to_global_coordinates() -> None:
    from api.core.cascade_plate import map_crop_points_to_global

    crop_pts = np.array([[1, 2], [11, 2], [11, 7], [1, 7]], dtype=np.float32)
    global_pts = map_crop_points_to_global(crop_pts, offset=(30, 40))

    np.testing.assert_array_equal(
        global_pts,
        np.array([[31, 42], [41, 42], [41, 47], [31, 47]], dtype=np.float32),
    )


@pytest.mark.unit
def test_deduplicate_plate_candidates_prefers_smallest_containing_vehicle() -> None:
    from api.core.cascade_plate import deduplicate_plate_candidates

    tracked = [
        {"id": 10, "box": [0, 0, 180, 100]},
        {"id": 20, "box": [40, 20, 120, 80]},
    ]
    candidates = [
        {
            "box": [60, 40, 90, 55],
            "pts": np.array([[60, 40], [90, 40], [90, 55], [60, 55]]),
            "crop": np.full((15, 30, 3), 100, dtype=np.uint8),
            "conf": 0.70,
            "source_vehicle_id": 10,
        },
        {
            "box": [61, 41, 91, 56],
            "pts": np.array([[61, 41], [91, 41], [91, 56], [61, 56]]),
            "crop": np.full((15, 30, 3), 110, dtype=np.uint8),
            "conf": 0.65,
            "source_vehicle_id": 20,
        },
    ]

    deduped = deduplicate_plate_candidates(candidates, tracked)

    assert len(deduped) == 1
    assert deduped[0]["source_vehicle_id"] == 20


@pytest.mark.unit
def test_plate_track_manager_keeps_id_for_small_motion() -> None:
    from api.core.cascade_plate import PlateTrackManager

    manager = PlateTrackManager(iou_threshold=0.3, lost_buffer=3)
    first = manager.update([{"box": [10, 10, 50, 30], "conf": 0.9}])
    second = manager.update([{"box": [12, 11, 52, 31], "conf": 0.9}])

    assert first[0]["id"] == second[0]["id"]


@pytest.mark.unit
def test_plate_track_manager_uses_new_id_for_far_plate() -> None:
    from api.core.cascade_plate import PlateTrackManager

    manager = PlateTrackManager(iou_threshold=0.3, lost_buffer=3)
    first = manager.update([{"box": [10, 10, 50, 30], "conf": 0.9}])
    second = manager.update([{"box": [120, 80, 170, 100], "conf": 0.9}])

    assert first[0]["id"] != second[0]["id"]


@pytest.mark.unit
def test_plate_track_manager_survives_short_disappearance() -> None:
    from api.core.cascade_plate import PlateTrackManager

    manager = PlateTrackManager(iou_threshold=0.3, lost_buffer=3)
    first = manager.update([{"box": [10, 10, 50, 30], "conf": 0.9}])
    manager.update([])
    manager.update([])
    reappeared = manager.update([{"box": [11, 10, 51, 30], "conf": 0.9}])

    assert first[0]["id"] == reappeared[0]["id"]


@pytest.mark.unit
def test_plate_track_manager_rejects_owner_change_despite_overlap() -> None:
    from api.core.cascade_plate import PlateTrackManager

    manager = PlateTrackManager(iou_threshold=0.3, lost_buffer=3)
    first = manager.update([{"box": [10, 10, 70, 34], "conf": 0.9, "source_vehicle_id": 6}])
    second = manager.update([{"box": [14, 11, 74, 35], "conf": 0.9, "source_vehicle_id": 10}])

    assert first[0]["id"] != second[0]["id"]


@pytest.mark.unit
def test_plate_track_manager_keeps_id_for_same_owner_low_iou_motion() -> None:
    from api.core.cascade_plate import PlateTrackManager

    manager = PlateTrackManager(iou_threshold=0.3, lost_buffer=3)
    first = manager.update([{"box": [10, 10, 50, 30], "conf": 0.9, "source_vehicle_id": 6}])
    second = manager.update([{"box": [36, 11, 76, 31], "conf": 0.9, "source_vehicle_id": 6}])

    assert first[0]["id"] == second[0]["id"]
