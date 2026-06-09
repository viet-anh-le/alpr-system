"""LiveSession — per-monitor-session RTSP decoder + rolling frame buffer.

One decoder thread per session. Frames are appended to a bounded deque so the
last N seconds are always available. The same frames are JPEG-encoded and
pushed into an asyncio.Queue for the MJPEG fallback endpoint.
"""
from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Callable

import cv2
import numpy as np

from . import mediamtx_client

logger = logging.getLogger(__name__)

_INTERNAL_RTSP_BASE = os.environ.get("MEDIAMTX_INTERNAL_RTSP_BASE", "rtsp://mediamtx:8554")
_BUFFER_SECONDS = 10.0
_RECONNECT_RETRIES = 3
_RECONNECT_BACKOFF_SEC = 1.0


class LiveSession:
    """One live monitoring session. Owns the MediaMTX path and decoder thread."""

    def __init__(self, session_id: str, mediamtx_path: str) -> None:
        self.session_id = session_id
        self.mediamtx_path = mediamtx_path
        self.fps: float = 30.0
        self.frame_size: tuple[int, int] = (0, 0)
        self.frame_buffer: collections.deque = collections.deque()
        self.mjpeg_queue = None         # set on start()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_error: Callable[[str], None] | None = None

    # ── Test hooks ──────────────────────────────────────────────────────────
    def _init_buffer(self, fps: float, frame_size: tuple[int, int], seconds: float = _BUFFER_SECONDS) -> None:
        self.fps = fps
        self.frame_size = frame_size
        maxlen = max(1, int(fps * seconds))
        self.frame_buffer = collections.deque(maxlen=maxlen)

    def _push_frame(self, idx: int, frame: np.ndarray, ts: float) -> None:
        self.frame_buffer.append((idx, frame, ts))

    # ── Public API ──────────────────────────────────────────────────────────
    def start(self, rtsp_url: str, mjpeg_queue, on_error: Callable[[str], None] | None = None) -> None:
        """Register the MediaMTX path and spawn the decoder thread."""
        mediamtx_client.add_path(self.mediamtx_path, rtsp_url)
        self.mjpeg_queue = mjpeg_queue
        self._on_error = on_error
        self._thread = threading.Thread(target=self._decoder_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        try:
            mediamtx_client.remove_path(self.mediamtx_path)
        except Exception:
            logger.exception("mediamtx remove_path failed for %s", self.mediamtx_path)

    def snapshot_window(self, seconds: float = _BUFFER_SECONDS) -> list[tuple[int, np.ndarray, float]]:
        """Return a chronologically-ordered shallow copy of the last `seconds` of buffered frames."""
        wanted = int(self.fps * seconds)
        snap = list(self.frame_buffer)
        if wanted < len(snap):
            snap = snap[-wanted:]
        return snap

    # ── Internal ────────────────────────────────────────────────────────────
    def _decoder_loop(self) -> None:
        url = f"{_INTERNAL_RTSP_BASE}/{self.mediamtx_path}"
        attempts = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                attempts += 1
                if attempts > _RECONNECT_RETRIES:
                    self._fail(f"Cannot open RTSP source via MediaMTX: {url}")
                    return
                time.sleep(_RECONNECT_BACKOFF_SEC)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            self._init_buffer(fps=fps, frame_size=(w, h), seconds=_BUFFER_SECONDS)

            idx = 0
            attempts = 0
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                ts = idx / fps
                self._push_frame(idx, frame, ts)
                if self.mjpeg_queue is not None:
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok and not self.mjpeg_queue.full():
                        try:
                            self.mjpeg_queue.put_nowait(bytes(jpg))
                        except Exception:
                            pass
                idx += 1

            cap.release()
            # Loop back to retry connection unless stopped
            attempts += 1
            if attempts > _RECONNECT_RETRIES:
                self._fail("RTSP stream lost (retry budget exhausted)")
                return
            time.sleep(_RECONNECT_BACKOFF_SEC)

    def _fail(self, msg: str) -> None:
        logger.error("LiveSession[%s]: %s", self.session_id, msg)
        if self._on_error is not None:
            try:
                self._on_error(msg)
            except Exception:
                logger.exception("LiveSession error callback raised")
