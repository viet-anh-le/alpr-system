from __future__ import annotations

import logging
import os
import time

import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase.lib.client_options import SyncClientOptions

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


SUPABASE_STORAGE_HTTP2 = _env_bool("SUPABASE_STORAGE_HTTP2", False)
SUPABASE_UPLOAD_MAX_ATTEMPTS = max(1, _env_int("SUPABASE_UPLOAD_MAX_ATTEMPTS", 3))
SUPABASE_UPLOAD_RETRY_BACKOFF_SEC = max(
    0.0,
    _env_float("SUPABASE_UPLOAD_RETRY_BACKOFF_SEC", 0.25),
)
SUPABASE_HTTP_TIMEOUT_SEC = max(1.0, _env_float("SUPABASE_HTTP_TIMEOUT_SEC", 30.0))

_supabase: Client | None = None

_TRANSIENT_UPLOAD_ERRORS = (
    httpx.ConnectError,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
    httpx.WriteError,
)


def _build_client_options() -> SyncClientOptions:
    return SyncClientOptions(
        httpx_client=httpx.Client(
            timeout=SUPABASE_HTTP_TIMEOUT_SEC,
            follow_redirects=True,
            http2=SUPABASE_STORAGE_HTTP2,
        )
    )


def _discard_supabase_client() -> None:
    global _supabase
    _supabase = None


def get_supabase() -> Client | None:
    global _supabase
    if _supabase is None:
        if SUPABASE_URL and SUPABASE_KEY:
            _supabase = create_client(
                SUPABASE_URL,
                SUPABASE_KEY,
                options=_build_client_options(),
            )
    return _supabase


def upload_image(bucket: str, path: str, image_bytes: bytes, content_type: str = "image/jpeg") -> str | None:
    """Upload image bytes to Supabase Storage and return the public URL, or None on failure."""
    for attempt in range(1, SUPABASE_UPLOAD_MAX_ATTEMPTS + 1):
        client = get_supabase()
        if not client:
            return None

        try:
            bucket_api = client.storage.from_(bucket)
            bucket_api.upload(
                file=image_bytes,
                path=path,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            return bucket_api.get_public_url(path)
        except _TRANSIENT_UPLOAD_ERRORS as exc:
            if attempt >= SUPABASE_UPLOAD_MAX_ATTEMPTS:
                logger.exception("Supabase: failed to upload %s/%s", bucket, path)
                return None
            logger.warning(
                "Supabase: transient upload failure for %s/%s (attempt %d/%d): %s",
                bucket,
                path,
                attempt,
                SUPABASE_UPLOAD_MAX_ATTEMPTS,
                exc,
            )
            _discard_supabase_client()
            time.sleep(SUPABASE_UPLOAD_RETRY_BACKOFF_SEC * attempt)
        except Exception:
            logger.exception("Supabase: failed to upload %s/%s", bucket, path)
            return None

    return None
