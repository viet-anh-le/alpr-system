from __future__ import annotations

import httpx
import pytest


@pytest.mark.unit
def test_get_supabase_uses_http1_client(monkeypatch):
    from api.core import database

    captured = {}

    class FakeSupabase:
        pass

    def fake_create_client(url, key, options=None):
        captured["url"] = url
        captured["key"] = key
        captured["options"] = options
        return FakeSupabase()

    monkeypatch.setattr(database, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(database, "SUPABASE_KEY", "service-key")
    monkeypatch.setattr(database, "create_client", fake_create_client)
    monkeypatch.setattr(database, "_supabase", None)

    client = database.get_supabase()

    assert isinstance(client, FakeSupabase)
    options = captured["options"]
    assert options is not None
    assert isinstance(options.httpx_client, httpx.Client)
    assert options.httpx_client._transport._pool._http2 is False
    options.httpx_client.close()


@pytest.mark.unit
def test_upload_image_retries_remote_protocol_error(monkeypatch):
    from api.core import database

    attempts = {"count": 0}

    class FakeBucket:
        def upload(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.RemoteProtocolError("connection terminated")
            return {"ok": True}

        def get_public_url(self, path):
            return f"https://cdn.example/{path}"

    class FakeStorage:
        def from_(self, bucket):
            assert bucket == "evidence"
            return FakeBucket()

    class FakeSupabase:
        storage = FakeStorage()

    monkeypatch.setattr(database, "get_supabase", lambda: FakeSupabase())
    monkeypatch.setattr(database.time, "sleep", lambda _: None)

    url = database.upload_image("evidence", "session/plate.jpg", b"jpg")

    assert url == "https://cdn.example/session/plate.jpg"
    assert attempts["count"] == 2
