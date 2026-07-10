"""Tests for the local BotSort override that enables ReID in both passes."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("boxmot")

from api.core import botsort_reid as botsort_reid_module
from api.core.botsort_reid import AlwaysReIDBotSort
from boxmot.trackers.botsort.basetrack import TrackState


class DummyReIDModel:
    def __init__(self) -> None:
        self.seen_boxes: np.ndarray | None = None

    def get_features(self, boxes: np.ndarray, img: np.ndarray) -> np.ndarray:
        self.seen_boxes = boxes.copy()
        features = np.ones((len(boxes), 4), dtype=np.float32)
        if len(boxes) > 0:
            features[:, 1] = np.arange(1, len(boxes) + 1, dtype=np.float32)
        return features


def make_tracker(reid_model: DummyReIDModel | None = None) -> AlwaysReIDBotSort:
    return AlwaysReIDBotSort(
        reid_model=reid_model or DummyReIDModel(),
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.7,
        match_thresh=0.7,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        cmc_method=None,
    )


@pytest.mark.unit
def test_split_detections_returns_embeddings_for_second_pass() -> None:
    tracker = make_tracker()
    dets = np.array(
        [
            [0, 0, 10, 10, 0.8, 2],
            [20, 20, 30, 30, 0.2, 2],
            [40, 40, 50, 50, 0.05, 2],
        ],
        dtype=np.float32,
    )
    embs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    _, dets_first, embs_first, dets_second, embs_second = tracker._split_detections(
        dets,
        embs,
    )

    assert dets_first.shape[0] == 1
    assert dets_second.shape[0] == 1
    np.testing.assert_array_equal(embs_first, embs[:1])
    np.testing.assert_array_equal(embs_second, embs[1:2])


@pytest.mark.unit
def test_update_extracts_reid_features_for_high_and_low_confidence_detections() -> None:
    model = DummyReIDModel()
    tracker = make_tracker(model)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    dets = np.array(
        [
            [0, 0, 10, 10, 0.8, 2],
            [20, 20, 30, 30, 0.2, 2],
        ],
        dtype=np.float32,
    )

    tracker._update_impl(dets, frame)

    assert model.seen_boxes is not None
    np.testing.assert_array_equal(model.seen_boxes, dets[:, :4])


@pytest.mark.unit
def test_second_association_uses_reid_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = make_tracker()
    dets = np.array([[20, 20, 30, 30, 0.2, 2]], dtype=np.float32)
    embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    _, _, _, dets_second, embs_second = tracker._split_detections(dets, embs)

    class FakeTrack:
        state = TrackState.Tracked

        def __init__(self) -> None:
            self.updated_with = None

        def update(self, det: object, frame_count: int) -> None:
            self.updated_with = det

        def re_activate(self, det: object, frame_count: int, new_id: bool = False) -> None:
            self.updated_with = det

        def mark_lost(self) -> None:
            self.state = TrackState.Lost

    fake_track = FakeTrack()
    captured: dict[str, object] = {}

    def fake_iou_distance(
        tracks: list[FakeTrack],
        detections: list[object],
        is_obb: bool = False,
    ) -> np.ndarray:
        assert tracks == [fake_track]
        assert len(detections) == 1
        return np.array([[0.4]], dtype=np.float32)

    def fake_embedding_distance(
        tracks: list[FakeTrack],
        detections: list[object],
    ) -> np.ndarray:
        assert tracks == [fake_track]
        assert len(detections) == 1
        assert detections[0].curr_feat is not None
        return np.array([[0.2]], dtype=np.float32)

    def fake_linear_assignment(
        dists: np.ndarray,
        thresh: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        captured["dists"] = dists.copy()
        captured["thresh"] = thresh
        return (
            np.array([[0, 0]], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    monkeypatch.setattr(botsort_reid_module, "iou_distance", fake_iou_distance)
    monkeypatch.setattr(
        botsort_reid_module,
        "embedding_distance",
        fake_embedding_distance,
    )
    monkeypatch.setattr(
        botsort_reid_module,
        "linear_assignment",
        fake_linear_assignment,
    )

    activated_stracks: list[FakeTrack] = []
    tracker._second_association(
        dets_second,
        embs_second,
        activated_stracks,
        [],
        [],
        np.array([0], dtype=np.int64),
        [fake_track],
    )

    np.testing.assert_array_equal(captured["dists"], np.array([[0.2]], dtype=np.float32))
    assert captured["thresh"] == tracker.match_thresh
    assert activated_stracks == [fake_track]
    assert fake_track.updated_with is not None
