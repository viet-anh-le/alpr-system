"""Tests for api/core/frame_source.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from api.core.frame_source import AdaptiveFrameSource, FileFrameSource, LiveBufferFrameSource


FIXTURE = str(Path(__file__).parent / "fixtures" / "short_clip.mp4")


@pytest.mark.unit
def test_file_source_yields_all_frames_when_unrestricted():
    src = FileFrameSource(FIXTURE)
    frames = list(src.iter_frames())
    assert len(frames) == 30
    idx, frame, ts = frames[0]
    assert idx == 0
    assert frame.shape == (360, 640, 3)
    assert ts == pytest.approx(0.0, abs=0.05)


@pytest.mark.unit
def test_file_source_reports_metadata():
    src = FileFrameSource(FIXTURE)
    assert src.fps == pytest.approx(30.0, abs=0.1)
    assert src.frame_size == (640, 360)
    assert src.total_frames == 30


@pytest.mark.unit
def test_file_source_reports_interval_frame_count_and_preserves_file_indices():
    src = FileFrameSource(FIXTURE, t_start=0.5, t_end=0.8)

    frames = list(src.iter_frames())

    assert src.total_frames == 9
    assert len(frames) == 9
    assert frames[0][0] >= 13
    assert frames[0][0] != 0


@pytest.mark.unit
def test_file_source_interval_total_clamps_to_eof():
    src = FileFrameSource(FIXTURE, t_start=0.5, t_end=999.0)

    assert src.total_frames == 15
    assert len(list(src.iter_frames())) == 15


@pytest.mark.unit
def test_file_source_respects_t_start():
    src = FileFrameSource(FIXTURE, t_start=0.5)  # start at frame ~15
    frames = list(src.iter_frames())
    assert 13 <= len(frames) <= 17  # allow ±2 frames for codec seek imprecision
    first_idx = frames[0][0]
    assert first_idx >= 13


@pytest.mark.unit
def test_file_source_respects_t_end():
    src = FileFrameSource(FIXTURE, t_start=0.0, t_end=0.5)  # ~first 15 frames
    frames = list(src.iter_frames())
    assert 13 <= len(frames) <= 17
    last_ts = frames[-1][2]
    assert last_ts <= 0.6


@pytest.mark.unit
def test_file_source_t_end_beyond_duration_clamps_to_eof():
    src = FileFrameSource(FIXTURE, t_start=0.0, t_end=999.0)
    frames = list(src.iter_frames())
    assert len(frames) == 30  # never errors, just stops at EOF


@pytest.mark.unit
def test_live_buffer_source_passthrough():
    fake_frames = [
        (i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
        for i in range(10)
    ]
    src = LiveBufferFrameSource(fake_frames, fps=30.0, frame_size=(640, 360))
    out = list(src.iter_frames())
    assert out == fake_frames
    assert src.fps == 30.0
    assert src.total_frames == 10


@pytest.mark.unit
def test_file_source_raises_on_missing_file():
    with pytest.raises(RuntimeError, match="Cannot open video"):
        FileFrameSource("tests/fixtures/does_not_exist.mp4")


@pytest.mark.unit
def test_live_buffer_source_empty():
    src = LiveBufferFrameSource([], fps=30.0, frame_size=(640, 360))
    assert list(src.iter_frames()) == []
    assert src.total_frames == 0


@pytest.mark.unit
def test_live_buffer_source_copies_input():
    fake_frames = [(0, np.zeros((10, 10, 3), dtype=np.uint8), 0.0)]
    src = LiveBufferFrameSource(fake_frames, fps=30.0, frame_size=(10, 10))
    fake_frames.clear()  # mutate the caller's list
    out = list(src.iter_frames())
    assert len(out) == 1  # buffer kept its own copy


@pytest.mark.unit
def test_adaptive_frame_source_samples_to_target_fps_and_preserves_source_indices():
    fake_frames = [
        (i, np.zeros((1080, 1920, 3), dtype=np.uint8), i / 60.0)
        for i in range(10)
    ]
    src = LiveBufferFrameSource(fake_frames, fps=60.0, frame_size=(1920, 1080))

    wrapped = AdaptiveFrameSource(src, target_fps=15.0, max_width=1280)
    out = list(wrapped.iter_frames())

    assert wrapped.sample_stride == 4
    assert wrapped.total_frames == 10
    assert wrapped.frame_size == (1280, 720)
    assert [idx for idx, _frame, _ts in out] == [0, 4, 8]
    assert out[0][1].shape == (720, 1280, 3)
    assert out[1][2] == pytest.approx(4 / 60.0)


@pytest.mark.unit
def test_adaptive_frame_source_keeps_all_frames_when_target_exceeds_source_fps():
    fake_frames = [
        (i, np.zeros((360, 640, 3), dtype=np.uint8), i / 30.0)
        for i in range(5)
    ]
    src = LiveBufferFrameSource(fake_frames, fps=30.0, frame_size=(640, 360))

    wrapped = AdaptiveFrameSource(src, target_fps=60.0, max_width=0)

    assert wrapped.sample_stride == 1
    assert wrapped.frame_size == (640, 360)
    assert [idx for idx, _frame, _ts in wrapped.iter_frames()] == list(range(5))


@pytest.mark.unit
def test_adaptive_frame_source_target_zero_disables_sampling_for_tracking():
    fake_frames = [
        (i, np.zeros((360, 640, 3), dtype=np.uint8), i / 30.0)
        for i in range(6)
    ]
    src = LiveBufferFrameSource(fake_frames, fps=30.0, frame_size=(640, 360))

    wrapped = AdaptiveFrameSource(src, target_fps=0.0, max_width=0)

    assert wrapped.sample_stride == 1
    assert [idx for idx, _frame, _ts in wrapped.iter_frames()] == list(range(6))
