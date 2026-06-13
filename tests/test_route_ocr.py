from __future__ import annotations

import numpy as np
import pytest


def _crop(value: int = 128) -> np.ndarray:
    return np.full((48, 96, 3), value, dtype=np.uint8)


def _chars(text: str, conf: float = 0.95) -> list[tuple[str, float]]:
    return [(c, conf) for c in text]


@pytest.mark.unit
def test_direct_route_accepts_high_confidence_valid_original_ocr() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    tracker._cls[1] = "car"
    router = PlateQualityRouter(classifier=lambda crop: {"good": 0.96})

    jobs, active_tids = prepare_route_ocr_jobs([(1, _crop(), _crop())], tracker, router, 10)
    assert [job.candidate_method for job in jobs] == ["original"]
    ocr_results = [(_chars("30G-51827"), True) if job.candidate_method == "original" else (_chars("BAD"), False) for job in jobs]
    events: list[dict] = []

    consume_route_ocr_results(jobs, ocr_results, tracker, events.append)

    assert active_tids == {1}
    assert tracker._done[1] is True
    assert events[0]["type"] == "vehicle"
    assert events[0]["ocr_method"] == "single_frame_direct"
    assert events[0]["route"] == "direct"


@pytest.mark.unit
def test_direct_route_accepts_slot_corrected_ambiguous_ocr() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    tracker._cls[1] = "car"
    router = PlateQualityRouter(classifier=lambda crop: {"good": 0.96})

    jobs, _active_tids = prepare_route_ocr_jobs([(1, _crop(), _crop())], tracker, router, 10)
    ocr_results = [
        (_chars("30G-51B27"), True)
        if job.candidate_method == "original"
        else (_chars("BAD"), False)
        for job in jobs
    ]
    events: list[dict] = []

    consume_route_ocr_results(jobs, ocr_results, tracker, events.append)

    assert tracker.display_text(1) == "30G-51827"
    assert events[0]["plate"] == "30G-51827"
    assert events[0]["ocr_method"] == "single_frame_direct"


@pytest.mark.unit
def test_poor_route_buffers_for_tracklet_fusion_without_immediate_emit() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    router = PlateQualityRouter(classifier=lambda crop: {"poor": 0.88})

    jobs, _ = prepare_route_ocr_jobs([(2, _crop(), _crop())], tracker, router, 11)

    assert jobs == []
    assert tracker._done.get(2) is not True
    assert tracker._buffers[2].routes
    assert set(tracker._buffers[2].routes) == {"tracklet_fusion"}
    assert tracker._buffers[2].char_prob_lists == [[]]


@pytest.mark.unit
def test_illegible_route_buffers_unreadable_evidence_without_ocr_job() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    router = PlateQualityRouter(classifier=lambda crop: {"illegible": 0.91})

    jobs, active_tids = prepare_route_ocr_jobs([(3, _crop(), _crop())], tracker, router, 12)

    assert jobs == []
    assert active_tids == {3}
    assert tracker._buffers[3].routes == ["unreadable_wait"]
    assert tracker._buffers[3].char_prob_lists == [[]]


@pytest.mark.unit
def test_direct_route_finishes_track_and_leaves_existing_poor_illegible_buffer_unread() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    tracker._cls[4] = "car"
    tracker.buffer_crop(4, _crop(80), 0.35, 0.10, [], 1, route="tracklet_fusion")
    tracker.buffer_crop(4, _crop(20), 0.05, 0.10, [], 2, route="unreadable_wait")
    router = PlateQualityRouter(classifier=lambda crop: {"perfect": 0.98})

    jobs, _active_tids = prepare_route_ocr_jobs([(4, _crop(220), _crop(180))], tracker, router, 3)
    events: list[dict] = []
    consume_route_ocr_results(jobs, [(_chars("51G-12345"), True)], tracker, events.append)

    assert tracker._done[4] is True
    assert events[0]["ocr_method"] == "single_frame_direct"
    assert events[0]["plate"] == "51G-12345"
    assert tracker._buffers[4].routes[:2] == ["tracklet_fusion", "unreadable_wait"]
