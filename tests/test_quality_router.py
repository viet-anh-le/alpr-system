from __future__ import annotations

import numpy as np
import pytest


def _crop(value: int = 128, h: int = 48, w: int = 96) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


@pytest.mark.unit
def test_router_maps_perfect_and_good_to_direct_route() -> None:
    from api.core.quality_router import PlateQualityRouter

    router = PlateQualityRouter(classifier=lambda crop: {"perfect": 0.8, "good": 0.1})
    result = router.route(_crop())

    assert result.legibility == "perfect"
    assert result.quality_bin == "suitable"
    assert result.route == "direct"
    assert result.router_conf == pytest.approx(0.8)


@pytest.mark.unit
def test_router_maps_poor_to_tracklet_fusion_route() -> None:
    from api.core.quality_router import PlateQualityRouter

    router = PlateQualityRouter(classifier=lambda crop: {"poor": 0.72, "good": 0.2})
    result = router.route(_crop())

    assert result.legibility == "poor"
    assert result.quality_bin == "unsuitable"
    assert result.route == "tracklet_fusion"
    assert result.tags.low_res is False


@pytest.mark.unit
def test_router_maps_illegible_to_unreadable_wait_route() -> None:
    from api.core.quality_router import DegradationTags, PlateQualityRouter

    router = PlateQualityRouter(
        classifier=lambda crop: {"illegible": 0.91, "poor": 0.05},
        diagnoser=lambda crop: DegradationTags(occluded=True),
    )
    result = router.route(_crop())

    assert result.legibility == "illegible"
    assert result.route == "unreadable_wait"
    assert result.tags.occluded is True


@pytest.mark.unit
def test_heuristic_router_marks_tiny_crop_as_low_res_tracklet() -> None:
    from api.core.quality_router import PlateQualityRouter

    result = PlateQualityRouter().route(_crop(h=8, w=18))

    assert result.tags.low_res is True
    assert result.route in {"tracklet_fusion", "unreadable_wait"}
