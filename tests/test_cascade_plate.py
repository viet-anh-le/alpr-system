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
    assert deduped[0]["id"] == 20


@pytest.mark.unit
def test_cascade_plate_module_does_not_expose_plate_track_manager() -> None:
    import api.core.cascade_plate as cascade_plate

    assert not hasattr(cascade_plate, "PlateTrackManager")


@pytest.mark.unit
def test_detect_plates_cascade_returns_vehicle_keyed_candidates(monkeypatch) -> None:
    import api.core.cascade_plate as cascade_plate

    frame = _frame()
    tracked = [{"id": 10, "box": [0, 0, 100, 80]}]
    crop = cascade_plate.VehicleCrop(
        vehicle_id=10,
        vehicle_box=(0, 0, 100, 80),
        crop_box=(0, 0, 100, 80),
        offset=(0, 0),
        image=frame,
    )
    raw_candidate = {
        "box": [20, 30, 70, 50],
        "pts": np.array([[20, 30], [70, 30], [70, 50], [20, 50]], dtype=np.float32),
        "crop": np.full((20, 50, 3), 100, dtype=np.uint8),
        "conf": 0.9,
        "source_vehicle_id": 10,
    }

    class FakeModel:
        def predict(self, images, **kwargs):
            assert images == [frame]
            assert kwargs["verbose"] is False
            return [object()]

    def fake_deduplicate(candidates, seen_tracked):
        assert candidates == [raw_candidate]
        assert seen_tracked == tracked
        return [{**raw_candidate, "source_vehicle_id": 20, "id": 20}]

    monkeypatch.setattr(cascade_plate, "crop_vehicle_regions", lambda *_args, **_kwargs: [crop])
    monkeypatch.setattr(
        cascade_plate,
        "_extract_obb_candidates",
        lambda _result, _vehicle_crop, _frame: [raw_candidate],
    )
    monkeypatch.setattr(cascade_plate, "deduplicate_plate_candidates", fake_deduplicate)

    detected = cascade_plate.detect_plates_cascade(
        frame,
        tracked,
        FakeModel(),
        use_half=False,
    )

    assert detected == [{**raw_candidate, "source_vehicle_id": 20, "id": 20}]


@pytest.mark.unit
def test_detect_plates_cascade_returns_no_unowned_candidates(monkeypatch) -> None:
    import api.core.cascade_plate as cascade_plate

    frame = _frame()
    tracked = [{"id": 10, "box": [0, 0, 100, 80]}]
    crop = cascade_plate.VehicleCrop(
        vehicle_id=10,
        vehicle_box=(0, 0, 100, 80),
        crop_box=(0, 0, 100, 80),
        offset=(0, 0),
        image=frame,
    )
    raw_candidate = {
        "box": [20, 30, 70, 50],
        "crop": np.full((20, 50, 3), 100, dtype=np.uint8),
        "conf": 0.9,
        "source_vehicle_id": 10,
    }

    class FakeModel:
        def predict(self, images, **_kwargs):
            return [object()]

    monkeypatch.setattr(cascade_plate, "crop_vehicle_regions", lambda *_args, **_kwargs: [crop])
    monkeypatch.setattr(
        cascade_plate,
        "_extract_obb_candidates",
        lambda _result, _vehicle_crop, _frame: [raw_candidate],
    )
    monkeypatch.setattr(
        cascade_plate,
        "deduplicate_plate_candidates",
        lambda _candidates, _tracked: [],
    )

    assert cascade_plate.detect_plates_cascade(
        frame,
        tracked,
        FakeModel(),
        use_half=False,
    ) == []


@pytest.mark.unit
def test_extract_obb_candidates_skips_missing_obb_payloads() -> None:
    from api.core.cascade_plate import VehicleCrop, _extract_obb_candidates

    frame = _frame()
    crop = VehicleCrop(
        vehicle_id=10,
        vehicle_box=(0, 0, 100, 80),
        crop_box=(0, 0, 100, 80),
        offset=(0, 0),
        image=frame,
    )
    missing_obb = type("MissingObbResult", (), {"obb": None})()
    missing_points = type(
        "MissingPointsResult",
        (),
        {"obb": type("Obb", (), {"xyxyxyxy": None})()},
    )()

    assert _extract_obb_candidates(missing_obb, crop, frame) == []
    assert _extract_obb_candidates(missing_points, crop, frame) == []
