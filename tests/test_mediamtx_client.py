"""Tests for api/core/mediamtx_client.py."""
from __future__ import annotations

import httpx
import pytest

from api.core import mediamtx_client


@pytest.fixture
def mock_api(monkeypatch):
    """Patch the module-level httpx.Client with a MockTransport."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and "/v3/config/paths/add/" in str(request.url):
            return httpx.Response(200, json={"ok": True})
        if request.method == "DELETE" and "/v3/config/paths/delete/" in str(request.url):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://mediamtx:9997")
    monkeypatch.setattr(mediamtx_client, "_client", client)
    return requests


@pytest.mark.unit
def test_default_api_url_targets_localhost_for_local_dev():
    assert mediamtx_client._DEFAULT_API_URL == "http://localhost:9997"


@pytest.mark.unit
def test_add_path_posts_correct_json(mock_api):
    mediamtx_client.add_path("live_abc", "rtsp://10.0.0.5/main")
    assert len(mock_api) == 1
    req = mock_api[0]
    assert req.method == "POST"
    assert str(req.url).endswith("/v3/config/paths/add/live_abc")
    import json as _json
    body = _json.loads(req.content.decode())
    assert body == {"source": "rtsp://10.0.0.5/main", "rtspTransport": "tcp"}


@pytest.mark.unit
def test_remove_path_sends_delete(mock_api):
    mediamtx_client.remove_path("live_abc")
    assert len(mock_api) == 1
    assert mock_api[0].method == "DELETE"


@pytest.mark.unit
def test_remove_path_is_idempotent_on_404(monkeypatch):
    def handler(request):
        return httpx.Response(404, json={"error": "not found"})
    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://mediamtx:9997"
    )
    monkeypatch.setattr(mediamtx_client, "_client", client)
    mediamtx_client.remove_path("nonexistent")  # must NOT raise


@pytest.mark.unit
def test_add_path_raises_on_5xx(monkeypatch):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})
    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://mediamtx:9997"
    )
    monkeypatch.setattr(mediamtx_client, "_client", client)
    with pytest.raises(mediamtx_client.MediaMTXError):
        mediamtx_client.add_path("x", "rtsp://x/y")
