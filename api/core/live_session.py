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
from urllib.parse import urlparse

import cv2
import numpy as np

from . import mediamtx_client

logger = logging.getLogger(__name__)

_INTERNAL_RTSP_BASE = os.environ.get("MEDIAMTX_INTERNAL_RTSP_BASE", "rtsp://localhost:8554")
_BUFFER_SECONDS = 10.0
_RECONNECT_RETRIES = 3
_RECONNECT_BACKOFF_SEC = 1.0
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# ── MJPEG fallback tuning (env-overridable) ──────────────────────────────────
# MJPEG sends a full JPEG per frame (no inter-frame compression), so high-res or
# high-fps sources saturate bandwidth and stutter. Downscale + cap fps + lower
# quality for the MJPEG copy only; the detection buffer keeps the full-res frame.
_MJPEG_MAX_WIDTH = int(os.environ.get("MJPEG_MAX_WIDTH", "960"))  # 0 = no downscale
_MJPEG_FPS = float(os.environ.get("MJPEG_FPS", "12"))  # 0 = every frame
_MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "70"))


def _rtsp_port(parsed) -> int:
    return parsed.port or 554


def _normalized_host(host: str | None) -> str | None:
    if host is None:
        return None
    lowered = host.lower()
    return "loopback" if lowered in _LOOPBACK_HOSTS else lowered


def internal_mediamtx_path(rtsp_url: str) -> str | None:
    """Return the MediaMTX path when `rtsp_url` points at our internal server."""
    parsed = urlparse(rtsp_url)
    base = urlparse(_INTERNAL_RTSP_BASE)
    if parsed.scheme != base.scheme:
        return None
    if _rtsp_port(parsed) != _rtsp_port(base):
        return None
    if _normalized_host(parsed.hostname) != _normalized_host(base.hostname):
        return None
    path = parsed.path.strip("/")
    return path or None


class LiveSession:
    """One live monitoring session. Owns the MediaMTX path and decoder thread."""

    def __init__(
        self,
        session_id: str,
        mediamtx_path: str,
        *,
        owns_mediamtx_path: bool = True,
    ) -> None:
        self.session_id = session_id
        self.mediamtx_path = mediamtx_path
        self.owns_mediamtx_path = owns_mediamtx_path
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
        if self.owns_mediamtx_path:
            mediamtx_client.add_path(self.mediamtx_path, rtsp_url)
        self.mjpeg_queue = mjpeg_queue
        self._on_error = on_error
        self._thread = threading.Thread(target=self._decoder_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if not self.owns_mediamtx_path:
            return
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
            mjpeg_stride = max(1, round(fps / _MJPEG_FPS)) if _MJPEG_FPS > 0 else 1
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                ts = idx / fps
                self._push_frame(idx, frame, ts)
                if self.mjpeg_queue is not None and idx % mjpeg_stride == 0:
                    self._enqueue_mjpeg(frame)
                idx += 1

            cap.release()
            # Loop back to retry connection unless stopped
            attempts += 1
            if attempts > _RECONNECT_RETRIES:
                self._fail("RTSP stream lost (retry budget exhausted)")
                return
            time.sleep(_RECONNECT_BACKOFF_SEC)

    def _enqueue_mjpeg(self, frame: np.ndarray) -> None:
        """Downscale + JPEG-encode a frame for the MJPEG fallback queue.

        Drops the frame when the consumer is behind (full queue) so latency
        stays bounded instead of building an ever-growing backlog.
        """
        if self.mjpeg_queue is None or self.mjpeg_queue.full():
            return
        mframe = frame
        h, w = frame.shape[:2]
        if _MJPEG_MAX_WIDTH and w > _MJPEG_MAX_WIDTH:
            new_h = max(1, round(h * _MJPEG_MAX_WIDTH / w))
            mframe = cv2.resize(
                frame, (_MJPEG_MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA
            )
        ok, jpg = cv2.imencode(".jpg", mframe, [cv2.IMWRITE_JPEG_QUALITY, _MJPEG_QUALITY])
        if not ok:
            return
        try:
            self.mjpeg_queue.put_nowait(bytes(jpg))
        except Exception:
            pass

    def _fail(self, msg: str) -> None:
        logger.error("LiveSession[%s]: %s", self.session_id, msg)
        if self._on_error is not None:
            try:
                self._on_error(msg)
            except Exception:
                logger.exception("LiveSession error callback raised")
