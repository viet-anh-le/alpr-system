from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_supabase: Client | None = None


def get_supabase() -> Client | None:
    global _supabase
    if _supabase is None:
        if SUPABASE_URL and SUPABASE_KEY:
            _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def upload_image(bucket: str, path: str, image_bytes: bytes, content_type: str = "image/jpeg") -> str | None:
    """Upload image bytes to Supabase Storage and return the public URL, or None on failure."""
    client = get_supabase()
    if not client:
        return None

    try:
        client.storage.from_(bucket).upload(
            file=image_bytes,
            path=path,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return client.storage.from_(bucket).get_public_url(path)
    except Exception:
        logger.exception("Supabase: failed to upload %s/%s", bucket, path)
        return None
