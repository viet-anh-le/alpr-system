from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video I/O and frame buffering for ALPR pipeline."""

    def __init__(self, source: str | Path, buffer_size: int = 8) -> None:
        self.source = str(source)
        self.buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

    def read_frames(self):
        while self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()
