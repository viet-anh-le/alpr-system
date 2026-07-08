"""Runtime artifacts for previewing preprocessed ALPR video.

The recorder wraps an already-preprocessed FrameSource and writes exactly the
frames consumed by the pipeline to a browser-playable MP4. Recording failures
are non-fatal: inference continues and the UI simply omits the artifact URL.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import (
    ALPR_PREPROCESSED_VIDEO_CLEANUP_INTERVAL_SEC,
    ALPR_PREPROCESSED_VIDEO_DIR,
    ALPR_PREPROCESSED_VIDEO_TTL_SEC,
)
from .frame_source import FrameSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessedVideoArtifact:
    job_id: str
    user_id: str
    path: Path
    created_at: float
    expires_at: float


_artifacts: dict[str, PreprocessedVideoArtifact] = {}
_artifacts_lock = threading.Lock()
_cleanup_task: asyncio.Task | None = None


def _safe_job_id(job_id: str) -> str:
    safe = "".join(ch for ch in str(job_id) if ch.isalnum() or ch in {"-", "_"})
    return safe or "job"


def build_preprocessed_video_path(job_id: str) -> Path:
    return ALPR_PREPROCESSED_VIDEO_DIR / f"{_safe_job_id(job_id)}.mp4"


def preprocessed_video_url(job_id: str) -> str:
    return f"/jobs/{job_id}/preprocessed-video"


def register_preprocessed_video_artifact(
    job_id: str,
    user_id: str,
    path: str | Path,
    *,
    ttl_sec: float = ALPR_PREPROCESSED_VIDEO_TTL_SEC,
) -> PreprocessedVideoArtifact:
    now = time.time()
    artifact = PreprocessedVideoArtifact(
        job_id=str(job_id),
        user_id=str(user_id),
        path=Path(path),
        created_at=now,
        expires_at=now + max(1.0, float(ttl_sec)),
    )
    with _artifacts_lock:
        _artifacts[artifact.job_id] = artifact
    return artifact


def get_preprocessed_video_artifact(
    job_id: str,
    user_id: str,
    *,
    now: float | None = None,
) -> PreprocessedVideoArtifact | None:
    current = time.time() if now is None else now
    with _artifacts_lock:
        artifact = _artifacts.get(str(job_id))
        if artifact is None:
            return None
        if artifact.expires_at <= current or not artifact.path.exists():
            _artifacts.pop(str(job_id), None)
            _delete_file(artifact.path)
            return None
        if artifact.user_id != str(user_id):
            return None
        return artifact


def cleanup_expired_preprocessed_video_artifacts(now: float | None = None) -> int:
    current = time.time() if now is None else now
    expired: list[PreprocessedVideoArtifact] = []
    with _artifacts_lock:
        for job_id, artifact in list(_artifacts.items()):
            if artifact.expires_at <= current or not artifact.path.exists():
                expired.append(artifact)
                _artifacts.pop(job_id, None)
    for artifact in expired:
        _delete_file(artifact.path)
    return len(expired)


def clear_preprocessed_video_artifacts(*, delete_files: bool = True) -> None:
    with _artifacts_lock:
        artifacts = list(_artifacts.values())
        _artifacts.clear()
    if delete_files:
        for artifact in artifacts:
            _delete_file(artifact.path)


def start_preprocessed_video_cleanup_task() -> None:
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop())


async def stop_preprocessed_video_cleanup_task() -> None:
    global _cleanup_task
    task = _cleanup_task
    _cleanup_task = None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _cleanup_loop() -> None:
    interval = max(30.0, float(ALPR_PREPROCESSED_VIDEO_CLEANUP_INTERVAL_SEC))
    while True:
        await asyncio.sleep(interval)
        cleanup_expired_preprocessed_video_artifacts()


def _delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove preprocessed artifact %s", path)


class RecordingFrameSource:
    """FrameSource wrapper that mirrors consumed frames into an MP4 artifact."""

    def __init__(
        self,
        source: FrameSource,
        output_path: str | Path,
        *,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.source = source
        self.output_path = Path(output_path)
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = source.fps
        self.frame_size = source.frame_size
        self.total_frames = source.total_frames
        self.available = False
        self.error: str | None = None
        self._process: subprocess.Popen | None = None
        self._frames_written = 0
        self._failed = False

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray, float]]:
        try:
            for frame_idx, frame, ts in self.source.iter_frames():
                self.record_frame(frame)
                yield frame_idx, frame, ts
        finally:
            self.finish()

    def record_frame(self, frame: np.ndarray) -> None:
        self._write_frame(frame)

    def finish(self) -> None:
        self._finish()

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._failed:
            return
        if not _is_recordable_frame(frame):
            self._mark_failed("Frame is not uint8 BGR")
            return

        if self._process is None:
            height, width = frame.shape[:2]
            self._start(width, height)
        if self._process is None or self._process.stdin is None:
            return

        try:
            contiguous = np.ascontiguousarray(frame)
            self._process.stdin.write(contiguous.tobytes())
            self._frames_written += 1
        except (BrokenPipeError, OSError, ValueError) as exc:
            self._mark_failed(f"ffmpeg write failed: {exc}")
            self._terminate_process()

    def _start(self, width: int, height: int) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._mark_failed(f"Could not create artifact directory: {exc}")
            return

        fps = self.fps if self.fps and self.fps > 0 else 30.0
        command = [
            self.ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            f"{float(fps):.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._mark_failed(f"Could not start ffmpeg: {exc}")

    def _finish(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            if self._failed:
                _delete_file(self.output_path)
            return

        try:
            if process.stdin is not None:
                process.stdin.close()
            returncode = process.wait(timeout=30)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as exc:
            self._mark_failed(f"ffmpeg finalization failed: {exc}")
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                process.wait(timeout=5)
            _delete_file(self.output_path)
            return

        if (
            returncode == 0
            and self._frames_written > 0
            and self.output_path.exists()
            and self.output_path.stat().st_size > 0
        ):
            self.available = True
            return

        self._mark_failed(f"ffmpeg exited with code {returncode}")
        _delete_file(self.output_path)

    def _terminate_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        with contextlib.suppress(OSError):
            if process.stdin is not None:
                process.stdin.close()
        with contextlib.suppress(OSError):
            process.terminate()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=5)
        _delete_file(self.output_path)

    def _mark_failed(self, message: str) -> None:
        if not self._failed:
            logger.warning("Preprocessed video recording unavailable: %s", message)
        self._failed = True
        self.error = message


def _is_recordable_frame(frame: np.ndarray) -> bool:
    return (
        isinstance(frame, np.ndarray)
        and frame.dtype == np.uint8
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.size > 0
    )
