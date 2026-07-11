from __future__ import annotations

import numpy as np
import pytest

from api.core.frame_source import LiveBufferFrameSource
from api.core.preprocessing import (
    PREPROCESS_MODES,
    PreprocessedFrameSource,
    apply_preprocessing,
    normalize_preprocess_mode,
)


def _gradient_frame() -> np.ndarray:
    x = np.tile(np.arange(0, 80, dtype=np.uint8), (40, 1))
    return np.dstack([x, x, x])


@pytest.mark.unit
def test_normalize_preprocess_mode_defaults_and_validates() -> None:
    assert normalize_preprocess_mode(None) == "none"
    assert normalize_preprocess_mode("") == "none"
    assert normalize_preprocess_mode(" NIGHT ") == "night"

    with pytest.raises(ValueError, match="Invalid preprocess_mode"):
        normalize_preprocess_mode("snow")


@pytest.mark.unit
@pytest.mark.parametrize("mode", sorted(PREPROCESS_MODES))
def test_apply_preprocessing_preserves_shape_dtype_and_input(mode: str) -> None:
    frame = _gradient_frame()
    original = frame.copy()

    out = apply_preprocessing(frame, mode)

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    np.testing.assert_array_equal(frame, original)


@pytest.mark.unit
def test_apply_preprocessing_none_returns_copy() -> None:
    frame = _gradient_frame()
    out = apply_preprocessing(frame, "none")

    assert out is not frame
    np.testing.assert_array_equal(out, frame)


@pytest.mark.unit
def test_night_preprocessing_brightens_dark_frame() -> None:
    frame = np.full((50, 80, 3), 24, dtype=np.uint8)

    out = apply_preprocessing(frame, "night")

    assert float(out.mean()) > float(frame.mean())


@pytest.mark.unit
def test_preprocessed_frame_source_preserves_metadata_and_indices() -> None:
    frames = [
        (0, np.full((20, 30, 3), 20, dtype=np.uint8), 0.0),
        (1, np.full((20, 30, 3), 30, dtype=np.uint8), 1 / 30),
    ]
    source = LiveBufferFrameSource(frames, fps=30.0, frame_size=(30, 20))
    wrapped = PreprocessedFrameSource(source, mode="night")

    out = list(wrapped.iter_frames())

    assert wrapped.fps == 30.0
    assert wrapped.frame_size == (30, 20)
    assert wrapped.total_frames == 2
    assert [idx for idx, _, _ in out] == [0, 1]
    assert [ts for _, _, ts in out] == [0.0, 1 / 30]
    assert out[0][1].shape == (20, 30, 3)
    assert float(out[0][1].mean()) > float(frames[0][1].mean())
