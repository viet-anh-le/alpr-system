"""FrameSource protocol and implementations.

A FrameSource yields (frame_idx, frame_bgr, timestamp_sec) tuples and exposes
fps / frame_size / total_frames metadata. Used by pipeline_core.process_frames
so the inference loop is decoupled from where frames come from.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class FrameSource(Protocol):
    fps: float
    frame_size: tuple[int, int]      # (width, height)
    total_frames: int | None         # None when unknown / unbounded

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        ...


class FileFrameSource:
    """A FrameSource backed by a video file on disk.

    Seeks to ``t_start`` and stops yielding when frame timestamp >= ``t_end``.
    ``t_end=None`` means "until end of file".
    """

    def __init__(self, path: str | Path, t_start: float = 0.0, t_end: float | None = None) -> None:
        self.path = str(path)
        self.t_start = float(t_start)
        self.t_end = None if t_end is None else float(t_end)

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.file_total_frames = total if total > 0 else None
        self.total_frames = _interval_total_frames(
            self.file_total_frames,
            self.fps,
            self.t_start,
            self.t_end,
        )
        cap.release()

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot re-open video for iteration: {self.path}")
        try:
            if self.t_start > 0.0:
                cap.set(cv2.CAP_PROP_POS_MSEC, self.t_start * 1000.0)
            # Read the actual frame position after seek so frame_idx reflects
            # the file-level frame number, not just a per-iteration counter.
            while True:
                # Capture position before read to get the frame's own index.
                file_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame = cap.read()
                if not ret:
                    break
                # POS_MSEC reports timestamp of the next frame on some codecs;
                # compute timestamp from file position for stability.
                ts = file_pos / self.fps
                if self.t_end is not None and ts >= self.t_end:
                    break
                yield file_pos, frame, ts
        finally:
            cap.release()


class LiveBufferFrameSource:
    """A FrameSource that wraps an already-decoded list of frames.

    Used by the event analyzer after snapshotting a LiveSession's rolling
    buffer. Iteration is a no-op pass-through.
    """

    def __init__(
        self,
        frames: list[tuple[int, np.ndarray, float]],
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self._frames = list(frames)
        self.fps = float(fps)
        self.frame_size = frame_size
        self.total_frames = len(frames)

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        yield from self._frames


def _interval_total_frames(
    file_total_frames: int | None,
    fps: float,
    t_start: float,
    t_end: float | None,
) -> int | None:
    if file_total_frames is None:
        return None
    if fps <= 0:
        return file_total_frames

    start_frame = max(0, int(math.floor(max(0.0, t_start) * fps)))
    end_frame = file_total_frames if t_end is None else int(math.ceil(max(0.0, t_end) * fps))
    start_frame = min(file_total_frames, start_frame)
    end_frame = min(file_total_frames, max(0, end_frame))
    return max(0, end_frame - start_frame)


class AdaptiveFrameSource:
    """Sample and optionally resize frames while preserving source frame IDs.

    ``total_frames`` stays equal to the original source total so progress bars
    still reflect the user's video.  ``fps`` and ``frame_size`` describe the
    processed stream that downstream detection actually sees.
    """

    def __init__(
        self,
        source: FrameSource,
        *,
        target_fps: float,
        max_width: int,
    ) -> None:
        self.source = source
        self.sample_stride = _sample_stride(source.fps, target_fps)
        self.fps = source.fps / self.sample_stride if self.sample_stride > 0 else source.fps
        self.total_frames = source.total_frames
        self.max_width = int(max_width)
        self.frame_size = _resized_size(source.frame_size, self.max_width)

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        for ordinal, (frame_idx, frame, ts) in enumerate(self.source.iter_frames()):
            if ordinal % self.sample_stride != 0:
                continue
            yield frame_idx, _resize_frame(frame, self.max_width), ts


def _sample_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0 or source_fps <= 0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def _resized_size(frame_size: tuple[int, int], max_width: int) -> tuple[int, int]:
    width, height = frame_size
    if max_width <= 0 or width <= max_width or width <= 0:
        return frame_size
    scale = max_width / float(width)
    return max_width, max(1, int(round(height * scale)))


def _resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
