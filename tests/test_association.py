from __future__ import annotations

import pytest


def _vehicle(track_id: int, box: list[int]) -> dict:
    return {"id": track_id, "box": box}


def _plate(track_id: int, box: list[int], source_vehicle_id: int | None = None) -> dict:
    plate = {"id": track_id, "box": box}
    if source_vehicle_id is not None:
        plate["source_vehicle_id"] = source_vehicle_id
    return plate


@pytest.mark.unit
def test_associator_prefers_visible_source_vehicle_for_unlocked_plate() -> None:
    from api.core.association import TrajectoryAssociator

    associator = TrajectoryAssociator(match_frames=1, agreement_ratio=1.0)
    vehicles = [
        _vehicle(6, [40, 30, 160, 90]),
        _vehicle(10, [0, 0, 220, 120]),
    ]

    matches = associator.process_frame(
        [_plate(1, [80, 50, 120, 70], source_vehicle_id=10)],
        vehicles,
    )

    assert matches == [(10, {"id": 1, "box": [80, 50, 120, 70], "source_vehicle_id": 10})]
    assert associator.plate_to_vehicle[1] == 10


@pytest.mark.unit
def test_associator_unlocks_stale_match_when_source_and_geometry_conflict() -> None:
    from api.core.association import TrajectoryAssociator

    associator = TrajectoryAssociator(
        match_frames=1,
        agreement_ratio=1.0,
        lock_conflict_frames=2,
    )
    initial_vehicles = [
        _vehicle(6, [0, 0, 120, 120]),
        _vehicle(10, [160, 0, 300, 120]),
    ]

    assert associator.process_frame(
        [_plate(1, [30, 60, 70, 80], source_vehicle_id=6)],
        initial_vehicles,
    ) == [(6, {"id": 1, "box": [30, 60, 70, 80], "source_vehicle_id": 6})]

    conflict_vehicles = [
        _vehicle(6, [0, 0, 120, 120]),
        _vehicle(10, [160, 0, 300, 120]),
    ]
    foreign_plate = _plate(1, [200, 60, 240, 80], source_vehicle_id=10)

    assert associator.process_frame([foreign_plate], conflict_vehicles) == []
    assert associator.process_frame([foreign_plate], conflict_vehicles) == [
        (10, foreign_plate)
    ]
    assert associator.plate_to_vehicle[1] == 10
