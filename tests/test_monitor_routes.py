"""Tests for /monitor/* HTTP routes."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


pytestmark = pytest.mark.anyio


@pytest.fixture
async def client(monkeypatch):
    """Spin up a lightweight app with only monitor routes mounted."""
    from fastapi import FastAPI

    from api import routes_monitor

    app = FastAPI()
    app.state.models = MagicMock()
    app.include_router(routes_monitor.router)

    transport = httpx.ASGITransport(app=app)
    c = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    try:
        yield c
    finally:
        await c.aclose()
        routes_monitor.cleanup_all_upload_sessions()
        routes_monitor.monitor_sessions.clear()
        routes_monitor.event_queues.clear()


@pytest.mark.integration
async def test_monitor_upload_returns_session_id(client, tmp_path):
    fixture = Path("tests/fixtures/short_clip.mp4")
    with open(fixture, "rb") as f:
        resp = await client.post(
            "/monitor/upload",
            files={"file": ("short_clip.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert "video_url" in body
    assert body["video_url"].startswith("/monitor/upload/")


@pytest.mark.integration
async def test_monitor_live_connect_rejects_non_rtsp_scheme(client):
    resp = await client.post("/monitor/live/connect", json={"rtsp_url": "http://evil/path"})
    assert resp.status_code == 400


@pytest.mark.integration
async def test_monitor_live_connect_returns_urls(client, monkeypatch):
    """Mock LiveSession.start so we don't actually hit a camera."""
    from api import routes_monitor

    def fake_start(self, rtsp_url, mjpeg_queue, on_error=None):
        pass

    from api.core.live_session import LiveSession
    monkeypatch.setattr(LiveSession, "start", fake_start)

    resp = await client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://10.0.0.5/main"})
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert "whep_url" in body
    assert "mjpeg_url" in body


@pytest.mark.integration
async def test_monitor_live_mjpeg_returns_multipart(client, monkeypatch):
    from api import routes_monitor
    from api.core.live_session import LiveSession

    monkeypatch.setattr(LiveSession, "start", lambda self, *a, **kw: None)

    # Connect, then push a single JPEG into the queue
    resp = await client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://x/y"})
    sid = resp.json()["session_id"]
    routes_monitor.monitor_sessions[sid]["mjpeg_queue"].put_nowait(b"\xff\xd8fake")
    routes_monitor.monitor_sessions[sid]["mjpeg_queue"].put_nowait(None)

    async with client.stream("GET", f"/monitor/live/{sid}/mjpeg") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        # Just read one chunk and bail
        async for chunk in r.aiter_bytes():
            assert b"image/jpeg" in chunk
            break


@pytest.mark.integration
async def test_mark_upload_validates_window_too_long(client):
    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = await client.post(f"/monitor/{sid}/mark",
        json={"mode": "upload", "t_start": 0.0, "t_end": 999.0})
    assert resp.status_code == 400


@pytest.mark.integration
async def test_mark_upload_accepts_valid_window(client, monkeypatch):
    from api import routes_monitor

    monkeypatch.setattr(routes_monitor, "_dispatch_event", lambda *a, **kw: None)

    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = await client.post(f"/monitor/{sid}/mark",
        json={"mode": "upload", "t_start": 0.0, "t_end": 1.0})
    assert resp.status_code == 200
    assert "event_id" in resp.json()


@pytest.mark.integration
async def test_event_stream_returns_event_stream_headers(client, monkeypatch):
    """SSE endpoint returns 200 + text/event-stream.

    Push None sentinel BEFORE streaming so the generator exits immediately
    after the initial keep-alive comment — avoids blocking on queue.get()."""
    from api import routes_monitor

    monkeypatch.setattr(routes_monitor, "_dispatch_event", lambda *a, **kw: None)

    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    # Sentinel closes the generator after the keep-alive comment is sent.
    routes_monitor.event_queues[sid].put_nowait(None)

    async with client.stream("GET", f"/monitor/{sid}/events/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("cache-control") == "no-cache"
        body = await r.aread()
        assert b"keep-alive" in body


@pytest.mark.integration
async def test_event_stream_404_for_unknown_session(client):
    resp = await client.get("/monitor/unknown_session_xyz/events/stream")
    assert resp.status_code == 404


# ── Upload video endpoint ─────────────────────────────────────────────────────


@pytest.mark.integration
async def test_monitor_upload_video_returns_file(client):
    from api import routes_monitor

    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = await routes_monitor.monitor_upload_video(sid)
    assert resp.status_code == 200
    assert "video" in resp.media_type


@pytest.mark.integration
async def test_monitor_upload_video_404_unknown_session(client):
    resp = await client.get("/monitor/upload/no_such_session/video")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_monitor_upload_disconnect_removes_session(client):
    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]
    from api import routes_monitor

    upload_path = Path(routes_monitor.monitor_sessions[sid]["path"])
    assert upload_path.exists()

    resp = await client.delete(f"/monitor/upload/{sid}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["cleanup"] == "deleted"
    assert not upload_path.exists()

    # Session is gone — video endpoint now 404
    resp = await client.get(f"/monitor/upload/{sid}/video")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_monitor_upload_disconnect_404_unknown_session(client):
    resp = await client.delete("/monitor/upload/no_such_session")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_monitor_upload_disconnect_defers_cleanup_when_event_active(client):
    from api import routes_monitor

    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]
    sess = routes_monitor.monitor_sessions[sid]
    upload_path = Path(sess["path"])
    sess["active_events"] = 1

    resp = await client.delete(f"/monitor/upload/{sid}")

    assert resp.status_code == 200
    assert resp.json()["cleanup"] == "deferred"
    assert upload_path.exists()
    assert routes_monitor.monitor_sessions[sid]["cleanup_requested"] is True

    routes_monitor.cleanup_upload_session(sid, force=True)


@pytest.mark.unit
def test_upload_lifecycle_release_runs_deferred_cleanup(tmp_path):
    from api import routes_monitor

    upload_path = tmp_path / "active.mp4"
    upload_path.write_bytes(b"fake")
    sid = "mon_active_cleanup"
    routes_monitor.monitor_sessions[sid] = {
        "kind": "upload",
        "path": str(upload_path),
        "filename": "active.mp4",
        "preprocess_mode": "none",
        "created_at": 0.0,
        "last_access_at": 0.0,
        "active_events": 1,
        "cleanup_requested": True,
    }
    routes_monitor.event_queues[sid] = asyncio.Queue()

    routes_monitor._release_upload_event(sid)

    assert sid not in routes_monitor.monitor_sessions
    assert sid not in routes_monitor.event_queues
    assert not upload_path.exists()


@pytest.mark.unit
def test_upload_lifecycle_wrapper_emits_error_when_runner_crashes(tmp_path):
    from api import routes_monitor

    upload_path = tmp_path / "active.mp4"
    upload_path.write_bytes(b"fake")
    sid = "mon_wrapper_error"
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.new_event_loop()
    routes_monitor.monitor_sessions[sid] = {
        "kind": "upload",
        "path": str(upload_path),
        "filename": "active.mp4",
        "preprocess_mode": "none",
        "created_at": 0.0,
        "last_access_at": 0.0,
        "active_events": 1,
        "cleanup_requested": False,
    }

    def failing_runner(**_kwargs):
        raise TypeError("unexpected kw")

    try:
        routes_monitor._run_event_with_upload_lifecycle(
            upload_session_id=sid,
            run_event_fn=failing_runner,
            event_id="evt_wrapper_error",
            queue=queue,
            loop=loop,
        )

        async def drain_one():
            return await asyncio.wait_for(queue.get(), timeout=0.2)

        ev = loop.run_until_complete(drain_one())
    finally:
        routes_monitor.monitor_sessions.pop(sid, None)
        loop.close()

    assert ev["type"] == "event_error"
    assert ev["event_id"] == "evt_wrapper_error"
    assert "unexpected kw" in ev["message"]


@pytest.mark.unit
def test_ttl_cleanup_removes_expired_inactive_uploads_only(tmp_path, monkeypatch):
    from api import routes_monitor

    now = 10_000.0
    monkeypatch.setattr(routes_monitor, "MONITOR_UPLOAD_TTL_SEC", 100.0)
    expired_path = tmp_path / "expired.mp4"
    active_path = tmp_path / "active.mp4"
    expired_path.write_bytes(b"old")
    active_path.write_bytes(b"active")
    expired_sid = "mon_expired"
    active_sid = "mon_active"

    routes_monitor.monitor_sessions[expired_sid] = {
        "kind": "upload",
        "path": str(expired_path),
        "filename": "expired.mp4",
        "preprocess_mode": "none",
        "created_at": now - 200.0,
        "last_access_at": now - 200.0,
        "active_events": 0,
        "cleanup_requested": False,
    }
    routes_monitor.monitor_sessions[active_sid] = {
        "kind": "upload",
        "path": str(active_path),
        "filename": "active.mp4",
        "preprocess_mode": "none",
        "created_at": now - 200.0,
        "last_access_at": now - 200.0,
        "active_events": 1,
        "cleanup_requested": False,
    }
    routes_monitor.event_queues[expired_sid] = asyncio.Queue()
    routes_monitor.event_queues[active_sid] = asyncio.Queue()

    removed = routes_monitor.cleanup_expired_upload_sessions(now=now)

    assert removed == [expired_sid]
    assert expired_sid not in routes_monitor.monitor_sessions
    assert active_sid in routes_monitor.monitor_sessions
    assert not expired_path.exists()
    assert active_path.exists()

    routes_monitor.cleanup_upload_session(active_sid, force=True)


@pytest.mark.unit
def test_stale_orphan_cleanup_only_removes_monitor_upload_prefixed_files(tmp_path, monkeypatch):
    from api import routes_monitor

    now = 20_000.0
    monkeypatch.setattr(routes_monitor, "MONITOR_UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(routes_monitor, "MONITOR_UPLOAD_TTL_SEC", 100.0)
    orphan = tmp_path / f"{routes_monitor.MONITOR_UPLOAD_PREFIX}orphan.mp4"
    unrelated = tmp_path / "other.tmp"
    orphan.write_bytes(b"old")
    unrelated.write_bytes(b"old")
    old = now - 200.0
    os.utime(orphan, (old, old))
    os.utime(unrelated, (old, old))

    removed = routes_monitor.cleanup_stale_upload_files(now=now)

    assert removed == [orphan]
    assert not orphan.exists()
    assert unrelated.exists()


# ── Live connect error path ───────────────────────────────────────────────────


@pytest.mark.integration
async def test_monitor_live_connect_error_returns_502(client, monkeypatch):
    from api.core.live_session import LiveSession

    def failing_start(self, rtsp_url, mjpeg_queue, on_error=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(LiveSession, "start", failing_start)
    resp = await client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://bad/stream"})
    assert resp.status_code == 502
    assert "Could not connect" in resp.json()["detail"]


# ── Live disconnect ───────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_monitor_live_disconnect_success(client, monkeypatch):
    from api.core.live_session import LiveSession

    monkeypatch.setattr(LiveSession, "start", lambda self, *a, **kw: None)
    stopped = []
    monkeypatch.setattr(LiveSession, "stop", lambda self: stopped.append(True))

    resp = await client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://x/y"})
    sid = resp.json()["session_id"]

    resp = await client.delete(f"/monitor/live/{sid}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert stopped


@pytest.mark.integration
async def test_monitor_live_disconnect_404_unknown_session(client):
    resp = await client.delete("/monitor/live/no_such_session")
    assert resp.status_code == 404


# ── MJPEG endpoint ────────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_monitor_live_mjpeg_404_unknown_session(client):
    resp = await client.get("/monitor/live/no_such_session/mjpeg")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_monitor_live_mjpeg_terminates_on_none_sentinel(client, monkeypatch):
    from api import routes_monitor
    from api.core.live_session import LiveSession

    monkeypatch.setattr(LiveSession, "start", lambda self, *a, **kw: None)

    resp = await client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://x/y"})
    sid = resp.json()["session_id"]

    # None sentinel → generator breaks without yielding any frames
    routes_monitor.monitor_sessions[sid]["mjpeg_queue"].put_nowait(None)

    async with client.stream("GET", f"/monitor/live/{sid}/mjpeg") as r:
        assert r.status_code == 200
        body = await r.aread()
        assert b"--frame" not in body


# ── Mark route validation ─────────────────────────────────────────────────────


@pytest.mark.integration
async def test_mark_404_unknown_session(client):
    resp = await client.post("/monitor/no_such_session/mark",
        json={"mode": "upload", "t_start": 0.0, "t_end": 1.0})
    assert resp.status_code == 404


@pytest.mark.integration
async def test_mark_upload_mode_on_live_session_returns_400(client, monkeypatch):
    from api.core.live_session import LiveSession

    monkeypatch.setattr(LiveSession, "start", lambda self, *a, **kw: None)
    resp = await client.post("/monitor/live/connect", json={"rtsp_url": "rtsp://x/y"})
    sid = resp.json()["session_id"]

    resp = await client.post(f"/monitor/{sid}/mark",
        json={"mode": "upload", "t_start": 0.0, "t_end": 1.0})
    assert resp.status_code == 400
    assert "live" in resp.json()["detail"].lower()


@pytest.mark.integration
async def test_mark_upload_missing_times_returns_400(client):
    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = await client.post(f"/monitor/{sid}/mark", json={"mode": "upload"})
    assert resp.status_code == 400


@pytest.mark.integration
async def test_mark_upload_invalid_interval_t_start_gt_t_end(client):
    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = await client.post(f"/monitor/{sid}/mark",
        json={"mode": "upload", "t_start": 5.0, "t_end": 3.0})
    assert resp.status_code == 400


@pytest.mark.integration
async def test_mark_live_on_upload_session_returns_400(client):
    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    resp = await client.post(f"/monitor/{sid}/mark", json={"mode": "live"})
    assert resp.status_code == 400
    assert "upload" in resp.json()["detail"].lower()


# ── _dispatch_event direct unit tests ─────────────────────────────────────


@pytest.mark.unit
def test_dispatch_event_upload_mode_submits_to_executor(monkeypatch):
    """Lines 179-212: _dispatch_event upload mode path."""
    from api import routes_monitor
    from api.routes_monitor import _dispatch_event, MarkBody

    submitted = []

    def fake_submit(fn, **kwargs):
        submitted.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(routes_monitor._event_executor, "submit", fake_submit)

    # _dispatch_event calls get_running_loop(); supply a real one
    fake_loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: fake_loop)

    mock_request = MagicMock()
    mock_request.app.state.models = MagicMock()

    sess = {
        "kind": "upload",
        "path": "tests/fixtures/short_clip.mp4",
        "filename": "short_clip.mp4",
        "preprocess_mode": "night",
        "created_at": 0.0,
        "last_access_at": 0.0,
        "active_events": 0,
        "cleanup_requested": False,
    }
    routes_monitor.monitor_sessions["ses_disp"] = dict(sess)

    try:
        _dispatch_event(
            event_id="evt_disp_upload",
            session_id="ses_disp",
            sess=sess,
            body=MarkBody(mode="upload", t_start=0.0, t_end=1.0),
            queue=asyncio.Queue(),
            request=mock_request,
        )
    finally:
        routes_monitor.monitor_sessions.pop("ses_disp", None)
        fake_loop.close()

    assert len(submitted) == 1
    assert submitted[0]["event_id"] == "evt_disp_upload"
    assert submitted[0]["source_type"] == "upload"
    assert submitted[0]["source_ref"] == "short_clip.mp4"
    assert submitted[0]["source"].mode == "night"


@pytest.mark.unit
def test_dispatch_event_live_mode_submits_to_executor(monkeypatch):
    """Lines 190-212: _dispatch_event live mode path."""
    from api import routes_monitor
    from api.routes_monitor import _dispatch_event, MarkBody

    submitted = []

    def fake_submit(fn, **kwargs):
        submitted.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(routes_monitor._event_executor, "submit", fake_submit)

    fake_loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: fake_loop)

    mock_live = MagicMock()
    mock_live.fps = 30.0
    mock_live.frame_size = (640, 360)
    mock_live.snapshot_window.return_value = [
        (i, np.zeros((360, 640, 3), dtype=np.uint8), float(i) / 30.0)
        for i in range(60)
    ]

    sess = {
        "kind": "live",
        "live_session": mock_live,
        "rtsp_url": "rtsp://cam/main",
    }

    mock_request = MagicMock()
    mock_request.app.state.models = MagicMock()

    _dispatch_event(
        event_id="evt_disp_live",
        session_id="ses_disp_live",
        sess=sess,
        body=MarkBody(mode="live"),
        queue=asyncio.Queue(),
        request=mock_request,
    )
    fake_loop.close()

    assert len(submitted) == 1
    assert submitted[0]["source_type"] == "live"
    assert submitted[0]["source_ref"] == "rtsp://cam/main"


@pytest.mark.integration
async def test_mark_live_buffer_warmup_returns_409(client):
    """Lines 193-194: buffer too small → 409."""
    from api import routes_monitor

    mock_live = MagicMock()
    mock_live.fps = 30.0
    mock_live.snapshot_window.return_value = []

    sid = "mon_warmup_test"
    routes_monitor.monitor_sessions[sid] = {
        "kind": "live",
        "live_session": mock_live,
        "mediamtx_path": "warmup_path",
        "mjpeg_queue": asyncio.Queue(),
        "rtsp_url": "rtsp://cam/warmup",
    }
    routes_monitor.event_queues[sid] = asyncio.Queue()

    try:
        resp = await client.post(f"/monitor/{sid}/mark", json={"mode": "live"})
        assert resp.status_code == 409
        assert "warming up" in resp.json()["detail"]
    finally:
        routes_monitor.monitor_sessions.pop(sid, None)
        routes_monitor.event_queues.pop(sid, None)


# ── SSE event data yield ──────────────────────────────────────────────────────


@pytest.mark.integration
async def test_event_stream_yields_event_data(client, monkeypatch):
    """Line 263: gen() yields data line when event dict is in queue."""
    from api import routes_monitor

    monkeypatch.setattr(routes_monitor, "_dispatch_event", lambda *a, **kw: None)

    resp = await client.post("/monitor/upload",
        files={"file": ("short.mp4", open("tests/fixtures/short_clip.mp4", "rb"), "video/mp4")})
    sid = resp.json()["session_id"]

    routes_monitor.event_queues[sid].put_nowait({"type": "event_started", "event_id": "x"})
    routes_monitor.event_queues[sid].put_nowait(None)

    async with client.stream("GET", f"/monitor/{sid}/events/stream") as r:
        body = await r.aread()
        assert b"event_started" in body


# ── /events GET routes ─────────────────────────────────────────────────────


@pytest.mark.integration
async def test_get_event_503_when_db_not_configured(client, monkeypatch):
    """Lines 278-279: GET /events/{id} → 503 when DB not configured."""
    import api.database.mongodb as mongodb_mod
    monkeypatch.setattr(mongodb_mod, "is_db_configured", lambda: False)

    resp = await client.get("/events/some_event_id")
    assert resp.status_code == 503


@pytest.mark.integration
async def test_list_events_returns_empty_items_when_no_db(client, monkeypatch):
    """Lines 294-295: GET /events → {"items": []} when DB not configured."""
    import api.database.mongodb as mongodb_mod
    monkeypatch.setattr(mongodb_mod, "is_db_configured", lambda: False)

    resp = await client.get("/events")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


def test_get_track_record_uses_initialised_api_database_module(monkeypatch):
    """GET /records must use the same mongodb module initialised by app startup."""
    import api.main as main_mod
    import api.database.mongodb as mongodb_mod
    from api.database.models import User
    from bson import ObjectId

    class FakeRecord:
        def model_dump(self, mode):
            return {"session_id": "job-1", "track_id": 1, "mode": mode}

    user = User(
        id=ObjectId(),
        email="owner@example.com",
        name="Owner",
        password_hash="hash",
    )

    async def fake_get_record_by_track(job_id, track_id, user_id):
        assert job_id == "job-1"
        assert track_id == 1
        assert user_id == str(user.id)
        return FakeRecord()

    monkeypatch.setattr(mongodb_mod, "is_db_configured", lambda: True)
    monkeypatch.setattr(mongodb_mod, "get_record_by_track_for_user", fake_get_record_by_track)

    result = asyncio.run(main_mod.get_track_record("job-1", 1, user))

    assert result == {"session_id": "job-1", "track_id": 1, "mode": "json"}
