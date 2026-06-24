"""Thin HTTP client for the MediaMTX control API.

The MediaMTX API lets us add/remove paths at runtime, which is how we make
each Event-Monitor session a separately-addressable stream.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://localhost:9997"
_API_URL = os.environ.get("MEDIAMTX_API_URL", _DEFAULT_API_URL)
_TIMEOUT = httpx.Timeout(5.0)

_client = httpx.Client(base_url=_API_URL, timeout=_TIMEOUT)


class MediaMTXError(RuntimeError):
    """Raised when the MediaMTX API returns an unexpected status."""


def add_path(name: str, source: str) -> None:
    """Register a new path that pulls from `source` (an RTSP URL)."""
    resp = _client.post(f"/v3/config/paths/add/{name}", json={"source": source})
    if resp.status_code >= 300:
        # Mask credentials in the logged URL
        safe = source.split("@")[-1] if "@" in source else source
        raise MediaMTXError(
            f"add_path({name}) failed: HTTP {resp.status_code} — source=…@{safe}"
        )


def remove_path(name: str) -> None:
    """Remove a path. Idempotent: 404 is ignored."""
    resp = _client.delete(f"/v3/config/paths/delete/{name}")
    if resp.status_code == 404:
        logger.debug("mediamtx: remove_path(%s) returned 404 (already gone)", name)
        return
    if resp.status_code >= 300:
        raise MediaMTXError(f"remove_path({name}) failed: HTTP {resp.status_code}")
