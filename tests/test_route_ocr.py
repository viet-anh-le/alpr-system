from __future__ import annotations

import numpy as np
import pytest


def _crop(value: int = 128) -> np.ndarray:
    return np.full((48, 96, 3), value, dtype=np.uint8)


def _chars(text: str, conf: float = 0.95) -> list[tuple[str, float]]:
    return [(c, conf) for c in text]


def _chars_with_confidences(
    text: str,
    confidences: list[float],
) -> list[tuple[str, float]]:
    assert len(text) == len(confidences)
    return list(zip(text, confidences))


def _buffer_text(tracker, tid: int) -> str:
    from api.core.plate_format import chars_to_display_text

    return chars_to_display_text(tracker._buffers[tid].char_prob_lists[-1])


@pytest.mark.unit
def test_direct_route_uses_best_job_crop_when_rerank_selects_second_candidate() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    tracker._cls[7] = "motorcycle"
    router = PlateQualityRouter(classifier=lambda crop: {"good": 0.96})
    matches = [
        (7, _crop(77), _crop(10)),
        (7, _crop(66), _crop(20)),
    ]
    jobs, _active_tids = prepare_route_ocr_jobs(matches, tracker, router, 10)
    events: list[dict] = []

    consume_route_ocr_results(
        jobs,
        [(_chars("BADINPUT", conf=0.60), False), (_chars("66B-45851"), True)],
        tracker,
        events.append,
    )

    assert events[0]["plate"] == "66B-45851"
    assert events[0]["done"] is False
    assert tracker._done.get(7) is not True
    assert _buffer_text(tracker, 7) == "66B-45851"
    assert tracker._buffers[7].crops[0][0, 0, 0] == 66
    assert tracker._plate_img[7][0, 0, 0] == 66


@pytest.mark.unit
def test_direct_route_previews_all_char_confident_valid_original_ocr_without_finalising() -> None:
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

    record_calls: list[tuple] = []

    consume_route_ocr_results(
        jobs,
        ocr_results,
        tracker,
        events.append,
        session_id="session-1",
        loop=object(),
        record_save=lambda *args: record_calls.append(args),
    )

    assert active_tids == {1}
    assert tracker._done.get(1) is not True
    assert events[0]["type"] == "vehicle"
    assert events[0]["done"] is False
    assert events[0]["ocr_method"] == "single_frame_direct"
    assert events[0]["route"] == "direct"
    assert record_calls == []
    assert tracker._buffers[1].routes == ["direct"]
    assert _buffer_text(tracker, 1) == "30G-51827"


@pytest.mark.unit
def test_direct_route_buffers_low_char_confidence_ocr_without_immediate_emit() -> None:
    from api.core.config import CONF_THRESHOLD
    from api.core.plate_format import mean_confidence
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    tracker._cls[1] = "car"
    router = PlateQualityRouter(classifier=lambda crop: {"good": 0.96})
    char_probs = _chars_with_confidences(
        "30G-51827",
        [0.96, 0.96, 0.96, 0.96, 0.96, 0.89, 0.96, 0.96, 0.96],
    )

    jobs, _active_tids = prepare_route_ocr_jobs([(1, _crop(), _crop())], tracker, router, 10)
    events: list[dict] = []

    assert mean_confidence(char_probs) >= CONF_THRESHOLD
    assert any(conf < CONF_THRESHOLD for _, conf in char_probs)

    consume_route_ocr_results(jobs, [(char_probs, False)], tracker, events.append)

    assert events == []
    assert 1 not in tracker._best
    assert tracker._done.get(1) is not True
    assert tracker._buffers[1].routes == ["tracklet_fusion"]
    assert _buffer_text(tracker, 1) == "30G-51827"


@pytest.mark.unit
def test_direct_route_previews_all_char_confident_slot_corrected_ambiguous_ocr_without_finalising() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    tracker._cls[1] = "car"
    router = PlateQualityRouter(classifier=lambda crop: {"good": 0.96})

    jobs, _active_tids = prepare_route_ocr_jobs([(1, _crop(), _crop())], tracker, router, 10)
    ocr_results = [
        (_chars("30G-51B27", conf=0.99), True)
        if job.candidate_method == "original"
        else (_chars("BAD"), False)
        for job in jobs
    ]
    events: list[dict] = []

    consume_route_ocr_results(jobs, ocr_results, tracker, events.append)

    assert tracker.display_text(1) == "30G-51827"
    assert events[0]["plate"] == "30G-51827"
    assert events[0]["done"] is False
    assert tracker._done.get(1) is not True
    assert _buffer_text(tracker, 1) == "30G-51827"


@pytest.mark.unit
def test_poor_route_buffers_for_tracklet_fusion_without_immediate_emit() -> None:
    from api.core.quality_router import PlateQualityRouter
    from api.core.route_ocr import consume_route_ocr_results, prepare_route_ocr_jobs
    from api.core.tracker import WebTrackletManager

    tracker = WebTrackletManager()
    router = PlateQualityRouter(classifier=lambda crop: {"poor": 0.88})

    jobs, _ = prepare_route_ocr_jobs([(2, _crop(), _crop())], tracker, router, 11)
    events: list[dict] = []

    assert len(jobs) == 1
    consume_route_ocr_results(jobs, [(_chars("30G-51827"), True)], tracker, events.append)

    assert events == []
    assert tracker._done.get(2) is not True
    assert tracker._buffers[2].routes
    assert set(tracker._buffers[2].routes) == {"tracklet_fusion"}
    assert _buffer_text(tracker, 2) == "30G-51827"


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
def test_direct_route_previews_and_leaves_existing_poor_illegible_buffer_unread() -> None:
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

    assert tracker._done.get(4) is not True
    assert events[0]["ocr_method"] == "single_frame_direct"
    assert events[0]["plate"] == "51G-12345"
    assert events[0]["done"] is False
    assert tracker._buffers[4].routes == ["tracklet_fusion", "unreadable_wait", "direct"]
    assert _buffer_text(tracker, 4) == "51G-12345"
