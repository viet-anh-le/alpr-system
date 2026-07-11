from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module():
    path = ROOT / "scripts" / "eval_single_frame.py"
    spec = importlib.util.spec_from_file_location("scripts.eval_single_frame", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _char_probs(text: str, confidence: float = 0.95) -> list[tuple[str, float]]:
    return [(char, confidence) for char in text]


@pytest.mark.unit
def test_single_frame_selection_keeps_best_raw_crop_without_mutating_previous_state() -> None:
    module = _load_script_module()
    low_quality = np.full((8, 16, 3), 10, dtype=np.uint8)
    high_quality = np.full((8, 16, 3), 20, dtype=np.uint8)
    scores = {10: 0.25, 20: 0.90}

    first = module.select_best_single_frames(
        {},
        [(7, low_quality, np.empty((0, 0, 3), dtype=np.uint8))],
        frame_idx=3,
        score_fn=lambda crop: scores[int(crop[0, 0, 0])],
    )
    selected = module.select_best_single_frames(
        first,
        [(7, high_quality, np.empty((0, 0, 3), dtype=np.uint8))],
        frame_idx=8,
        score_fn=lambda crop: scores[int(crop[0, 0, 0])],
    )

    assert first[7].frame_idx == 3
    assert selected[7].frame_idx == 8
    assert selected[7].score == pytest.approx(0.90)
    assert np.array_equal(selected[7].crop, high_quality)
    assert not np.shares_memory(selected[7].crop, high_quality)


@pytest.mark.unit
def test_single_frame_ocr_runs_once_per_selected_track_and_returns_raw_decode() -> None:
    module = _load_script_module()
    candidates = {
        9: module.SingleFrameCandidate(np.full((4, 6, 3), 9, np.uint8), 0.8, 12),
        2: module.SingleFrameCandidate(np.full((4, 6, 3), 2, np.uint8), 0.7, 10),
    }
    seen_batches: list[torch.Tensor] = []

    def preprocess(_model, crop: np.ndarray) -> torch.Tensor:
        return torch.tensor([float(crop[0, 0, 0])])

    def infer(_model, tensors: torch.Tensor, _device):
        seen_batches.append(tensors.clone())
        return [
            (_char_probs("3OA-12345"), False),
            (_char_probs("51F-99999"), False),
        ]

    results = module.ocr_best_single_frames(
        candidates,
        ocr_model=object(),
        device="cpu",
        preprocess_fn=preprocess,
        ocr_batch_fn=infer,
    )

    assert len(seen_batches) == 1
    assert seen_batches[0].flatten().tolist() == [2.0, 9.0]
    assert "".join(char for char, _ in results[2]) == "3OA-12345"
    assert "".join(char for char, _ in results[9]) == "51F-99999"


@pytest.mark.unit
def test_single_frame_ocr_skips_inference_when_no_track_has_a_candidate() -> None:
    module = _load_script_module()

    def fail(*_args, **_kwargs):
        raise AssertionError("OCR must not run without a selected frame")

    assert module.ocr_best_single_frames(
        {},
        ocr_model=object(),
        device="cpu",
        preprocess_fn=fail,
        ocr_batch_fn=fail,
    ) == {}


@pytest.mark.unit
def test_multiframe_result_uses_top_ranked_buffer_entries_and_ctm_voting() -> None:
    module = _load_script_module()
    tracker = module.WebTrackletManager()
    crop = np.full((8, 24, 3), 128, dtype=np.uint8)

    tracker.buffer_crop(5, crop, 0.95, 0.95, _char_probs("30A-12345"), 1)
    tracker.buffer_crop(5, crop, 0.90, 0.95, _char_probs("30A-12345"), 2)
    tracker.buffer_crop(5, crop, 0.10, 0.95, _char_probs("30A-99999"), 3)

    results = module.fuse_multiframe_results(tracker, top_k=2)

    assert results[5][0].text == "30A-123.45"
    assert results[5][0].frame_count == 2


@pytest.mark.unit
def test_multiframe_result_keeps_every_valid_plate_cluster() -> None:
    module = _load_script_module()
    tracker = module.WebTrackletManager()
    crop = np.full((8, 24, 3), 128, dtype=np.uint8)

    for frame_idx, text in enumerate(
        ["30A-12345", "30A-12345", "51F-99999", "51F-99999"],
        start=1,
    ):
        tracker.buffer_crop(12, crop, 0.90, 0.95, _char_probs(text), frame_idx)

    results = module.fuse_multiframe_results(tracker, top_k=4)

    assert {result.text for result in results[12]} == {"30A-123.45", "51F-999.99"}
    assert [result.frame_count for result in results[12]] == [2, 2]


@pytest.mark.unit
def test_evaluation_table_prints_every_multiframe_cluster(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    candidate = module.SingleFrameCandidate(
        np.zeros((4, 8, 3), dtype=np.uint8),
        0.8,
        5,
    )
    multiframe = {
        12: [
            module.MultiFrameResult("30A-123.45", tuple(_char_probs("30A-123.45")), 3),
            module.MultiFrameResult("51F-999.99", tuple(_char_probs("51F-999.99")), 2),
        ]
    }

    module.print_evaluation_table(
        {12: candidate},
        {12: _char_probs("30A-12345")},
        multiframe,
    )

    output = capsys.readouterr().out
    assert "30A-123.45" in output
    assert "51F-999.99" in output
    assert "12 (split)" in output


@pytest.mark.unit
def test_run_eval_keeps_router_out_of_single_frame_branch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    plate_crop = np.indices((24, 80)).sum(axis=0) % 2
    plate_crop = np.repeat((plate_crop * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    vehicle_crop = np.full((48, 96, 3), 100, dtype=np.uint8)
    frame = np.full((64, 128, 3), 50, dtype=np.uint8)
    router_calls: list[np.ndarray] = []
    ocr_batch_sizes: list[int] = []

    router = module.PlateQualityRouter(
        classifier=lambda crop: router_calls.append(crop) or {"good": 0.99},
    )

    class FakeModels:
        quality_router = router
        ocr = object()
        plate = object()
        device = "cpu"

        @staticmethod
        def create_vehicle_tracker():
            return object()

    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [frame]
            self.released = False

        @staticmethod
        def isOpened() -> bool:
            return True

        def read(self):
            return (True, self.frames.pop(0)) if self.frames else (False, None)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()

    monkeypatch.setattr(module, "load_models", lambda: FakeModels())
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        module,
        "_build_vehicle_tracks",
        lambda *_args: [{"id": 4, "box": [0, 0, 100, 60]}],
    )
    monkeypatch.setattr(
        module,
        "detect_plate_tracks_cascade",
        lambda *_args, **_kwargs: [{"id": 8, "crop": plate_crop}],
    )
    monkeypatch.setattr(
        module,
        "_associate_plate_crops",
        lambda *_args: [(4, plate_crop, vehicle_crop)],
    )
    monkeypatch.setattr(
        module,
        "preprocess_plate_for_model",
        lambda _model, crop: torch.tensor([float(crop.mean())]),
    )

    def fake_ocr_batch(_model, tensors: torch.Tensor, _device):
        ocr_batch_sizes.append(int(tensors.shape[0]))
        return [(_char_probs("30A-12345"), True) for _ in range(tensors.shape[0])]

    monkeypatch.setattr(module, "ocr_batch", fake_ocr_batch)

    module.run_eval("sample.mp4")

    output = capsys.readouterr().out
    assert capture.released is True
    assert len(router_calls) == 1
    assert ocr_batch_sizes == [1, 1]
    assert "30A-12345" in output
    assert "30A-123.45" in output


@pytest.mark.unit
def test_build_vehicle_tracks_converts_detector_output_for_cascade() -> None:
    module = _load_script_module()

    class Boxes:
        xyxy = torch.tensor([[1.0, 2.0, 21.0, 32.0]])
        conf = torch.tensor([0.88])
        cls = torch.tensor([6.0])

        def __len__(self) -> int:
            return 1

    class VehicleModel:
        @staticmethod
        def predict(*_args, **_kwargs):
            return [type("Prediction", (), {"boxes": Boxes()})()]

    class Models:
        vehicle = VehicleModel()

    class VehicleTracker:
        @staticmethod
        def track(detections: np.ndarray, _frame: np.ndarray):
            assert detections.shape == (1, 6)
            return (
                np.array([[1.0, 2.0, 21.0, 32.0]]),
                np.array([17]),
                np.array([6]),
            )

    tracked = module._build_vehicle_tracks(
        Models(),
        VehicleTracker(),
        np.zeros((40, 50, 3), dtype=np.uint8),
    )

    assert tracked == [{"id": 17, "box": [1.0, 2.0, 21.0, 32.0], "class_id": 6}]


@pytest.mark.unit
def test_associate_plate_crops_uses_real_vehicle_crop() -> None:
    module = _load_script_module()
    frame = np.full((50, 80, 3), 123, dtype=np.uint8)
    plate_crop = np.full((8, 20, 3), 200, dtype=np.uint8)

    class Associator:
        vehicle_cache = {4: (20, 15, 60, 40)}

        @staticmethod
        def process_frame(_plate_tracks, _tracked):
            return [(4, {"crop": plate_crop}), (99, {"crop": plate_crop})]

    matched = module._associate_plate_crops(
        frame,
        [{"id": 4, "box": [20, 15, 60, 40]}],
        [{"id": 8, "crop": plate_crop}],
        Associator(),
    )

    assert len(matched) == 1
    assert matched[0][0] == 4
    assert matched[0][1] is plate_crop
    assert matched[0][2].shape == (50, 72, 3)
