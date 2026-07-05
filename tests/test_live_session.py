"""Tests for api/core/live_session.py — covers rolling buffer logic only.
Actual RTSP decoding is integration-tested manually (see E2E checklist)."""
from __future__ import annotations

import numpy as np
import pytest

from api.core.live_session import LiveSession


@pytest.mark.unit
def test_buffer_evicts_old_frames_at_maxlen():
    sess = LiveSession(session_id="t1", mediamtx_path="t1")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=1.0)  # maxlen=30
    for i in range(50):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    assert len(sess.frame_buffer) == 30
    # Oldest remaining frame should be index 20 (50 pushed, 30 retained)
    assert sess.frame_buffer[0][0] == 20


@pytest.mark.unit
def test_snapshot_returns_chronological_copy():
    sess = LiveSession(session_id="t2", mediamtx_path="t2")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=10.0)
    for i in range(10):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    snap = sess.snapshot_window(seconds=10.0)
    assert len(snap) == 10
    assert [f[0] for f in snap] == list(range(10))


@pytest.mark.unit
def test_snapshot_clamps_to_available():
    sess = LiveSession(session_id="t3", mediamtx_path="t3")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=10.0)
    for i in range(5):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    snap = sess.snapshot_window(seconds=10.0)
    assert len(snap) == 5  # only 5 available


@pytest.mark.unit
def test_snapshot_is_decoupled_from_buffer():
    sess = LiveSession(session_id="t4", mediamtx_path="t4")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=10.0)
    for i in range(10):
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
    snap = sess.snapshot_window(seconds=10.0)
    # Mutating snap must not affect the live buffer or vice versa
    snap.clear()
    assert len(sess.frame_buffer) == 10

@pytest.mark.unit
def test_decoder_loop_error_path(monkeypatch):
    import cv2
    import api.core.live_session as live_session_mod
    monkeypatch.setattr(live_session_mod, "_RECONNECT_RETRIES", 0)
    monkeypatch.setattr(live_session_mod, "_RECONNECT_BACKOFF_SEC", 0)

    sess = LiveSession(session_id="test_err", mediamtx_path="test_err")

    class MockCap:
        def isOpened(self): return False
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: MockCap())

    errors = []
    sess._on_error = lambda m: errors.append(m)
    sess._decoder_loop()

    assert len(errors) == 1
    assert "Cannot open RTSP source" in errors[0]


@pytest.mark.unit
def test_start_registers_mediamtx_path_and_spawns_thread(monkeypatch):
    """Lines 56-60: start() calls mediamtx_client.add_path and starts decoder thread."""
    import asyncio
    import api.core.live_session as ls_mod

    added = []
    monkeypatch.setattr(ls_mod.mediamtx_client, "add_path", lambda name, url: added.append((name, url)))

    sess = LiveSession(session_id="start_test", mediamtx_path="start_path")
    sess._stop.set()  # pre-stop so the decoder thread exits immediately

    q = asyncio.Queue()
    sess.start("rtsp://cam/main", mjpeg_queue=q)

    assert added == [("start_path", "rtsp://cam/main")]
    assert sess._thread is not None
    sess._thread.join(timeout=2.0)


@pytest.mark.unit
def test_internal_mediamtx_path_detects_loopback_source():
    from api.core.live_session import internal_mediamtx_path

    assert internal_mediamtx_path("rtsp://127.0.0.1:8554/alpr_demo") == "alpr_demo"
    assert internal_mediamtx_path("rtsp://localhost:8554/alpr_demo") == "alpr_demo"
    assert internal_mediamtx_path("rtsp://10.0.0.5:8554/alpr_demo") is None


@pytest.mark.unit
def test_existing_mediamtx_path_skips_registration_and_removal(monkeypatch):
    import asyncio
    import api.core.live_session as ls_mod

    added = []
    removed = []
    monkeypatch.setattr(ls_mod.mediamtx_client, "add_path", lambda name, url: added.append((name, url)))
    monkeypatch.setattr(ls_mod.mediamtx_client, "remove_path", lambda name: removed.append(name))

    sess = LiveSession(
        session_id="existing_path",
        mediamtx_path="alpr_demo",
        owns_mediamtx_path=False,
    )
    sess._stop.set()

    sess.start("rtsp://127.0.0.1:8554/alpr_demo", mjpeg_queue=asyncio.Queue())
    sess.stop()

    assert added == []
    assert removed == []


@pytest.mark.unit
def test_stop_calls_mediamtx_remove_path(monkeypatch):
    """Lines 63-69: stop() calls mediamtx_client.remove_path."""
    import api.core.live_session as ls_mod

    removed = []
    monkeypatch.setattr(ls_mod.mediamtx_client, "remove_path", lambda name: removed.append(name))

    sess = LiveSession(session_id="stop_test", mediamtx_path="stop_path")
    sess.stop()

    assert removed == ["stop_path"]


@pytest.mark.unit
def test_stop_handles_remove_path_exception(monkeypatch):
    """Lines 68-69: stop() swallows exceptions from remove_path."""
    import api.core.live_session as ls_mod

    def raising_remove(name):
        raise RuntimeError("mediamtx unreachable")

    monkeypatch.setattr(ls_mod.mediamtx_client, "remove_path", raising_remove)

    sess = LiveSession(session_id="stop_err", mediamtx_path="err_path")
    sess.stop()  # must not raise


@pytest.mark.unit
def test_snapshot_window_trims_to_requested_seconds():
    """Line 76: snapshot_window trims when requested window < buffer size."""
    sess = LiveSession(session_id="trim_test", mediamtx_path="trim")
    sess._init_buffer(fps=30.0, frame_size=(640, 360), seconds=10.0)
    for i in range(90):  # 3 seconds of frames
        sess._push_frame(i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)

    snap = sess.snapshot_window(seconds=1.0)  # request only 1s = 30 frames
    assert len(snap) == 30
    assert snap[0][0] == 60  # last 30 of 90


@pytest.mark.unit
def test_default_internal_rtsp_base_targets_localhost_for_local_dev():
    import api.core.live_session as live_session_mod

    assert live_session_mod._INTERNAL_RTSP_BASE == "rtsp://localhost:8554"


@pytest.mark.unit
def test_decoder_loop_reads_frames_and_appends_to_buffer(monkeypatch):
    """Lines 90-121: decoder_loop happy path — reads frames from opened cap."""
    import cv2
    import api.core.live_session as ls_mod

    monkeypatch.setattr(ls_mod, "_RECONNECT_RETRIES", 0)
    monkeypatch.setattr(ls_mod, "_RECONNECT_BACKOFF_SEC", 0)

    frames_to_emit = [np.zeros((360, 640, 3), dtype=np.uint8)] * 5
    call_idx = [0]

    class MockCap:
        def isOpened(self): return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640.0
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 360.0
            return 0.0

        def read(self):
            i = call_idx[0]
            call_idx[0] += 1
            if i < len(frames_to_emit):
                return True, frames_to_emit[i]
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: MockCap())

    sess = LiveSession(session_id="read_test", mediamtx_path="read")
    errors = []
    sess._on_error = lambda m: errors.append(m)
    sess._decoder_loop()

    assert len(sess.frame_buffer) == 5
    assert errors == ["RTSP stream lost (retry budget exhausted)"]


@pytest.mark.unit
def test_fail_triggers_error_callback_exception_is_swallowed(monkeypatch):
    """Lines 127-129: _fail swallows exceptions raised by on_error callback."""

    def bad_callback(msg):
        raise RuntimeError("callback blew up")

    sess = LiveSession(session_id="fail_test", mediamtx_path="fail")
    sess._on_error = bad_callback
    sess._fail("some error")  # must not propagate the RuntimeError
