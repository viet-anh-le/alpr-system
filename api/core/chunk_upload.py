"""Shared chunked-upload store for endpoints behind a body-size-limited proxy.

Clients split large files into sub-limit chunks (Cloudflare free caps request
bodies at 100 MB); the server persists each part on disk and reassembles them
into a single file. Parts are removed on completion, on explicit abort, on a TTL
sweep of abandoned uploads, and on shutdown. What happens to the *reassembled*
file is the caller's concern (a processing job, or a playback session).

This module is framework-agnostic: methods return plain values/booleans and the
route layer maps them to HTTP responses.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

_COPY_BUF = 1024 * 1024


class ChunkUploadStore:
    def __init__(self, base_dir: str | Path, ttl_sec: int, max_chunks: int = 100_000) -> None:
        self.base_dir = Path(base_dir)
        self.ttl_sec = ttl_sec
        self.max_chunks = max_chunks
        self._uploads: dict[str, dict] = {}

    def _dir(self, upload_id: str) -> Path:
        return self.base_dir / upload_id

    def validate_params(self, upload_id: str, chunk_index: int, total_chunks: int) -> str | None:
        """Return an error message for bad params, else None."""
        if total_chunks < 1 or total_chunks > self.max_chunks:
            return "total_chunks không hợp lệ"
        if chunk_index < 0 or chunk_index >= total_chunks:
            return "chunk_index nằm ngoài phạm vi"
        if not upload_id.isalnum() or len(upload_id) > 64:
            return "upload_id không hợp lệ"
        return None

    def get(self, upload_id: str) -> dict | None:
        return self._uploads.get(upload_id)

    def begin_or_get(self, upload_id: str, owner: str, filename: str, suffix: str) -> dict:
        meta = self._uploads.get(upload_id)
        if meta is None:
            directory = self._dir(upload_id)
            directory.mkdir(parents=True, exist_ok=True)
            meta = {
                "owner": owner,
                "filename": filename,
                "suffix": suffix,
                "dir": str(directory),
                "created": time.time(),
                "bytes": 0,
            }
            self._uploads[upload_id] = meta
        return meta

    def write_chunk(self, meta: dict, chunk_index: int, data: bytes, max_bytes: int) -> bool:
        """Persist one chunk. Return False (without writing) if it would push the
        running total over max_bytes."""
        part = Path(meta["dir"]) / f"{chunk_index:06d}.part"
        prev = part.stat().st_size if part.exists() else 0
        running = meta["bytes"] - prev + len(data)
        if running > max_bytes:
            return False
        part.write_bytes(data)
        meta["bytes"] = running
        return True

    def received_count(self, meta: dict) -> int:
        return len(list(Path(meta["dir"]).glob("*.part")))

    def missing_chunks(self, meta: dict, total_chunks: int) -> list[int]:
        directory = Path(meta["dir"])
        return [i for i in range(total_chunks) if not (directory / f"{i:06d}.part").exists()]

    def assemble_into(self, meta: dict, total_chunks: int, target, max_bytes: int) -> int:
        """Concatenate parts 0..total_chunks-1 into the open binary file `target`.
        Return total bytes written; raise ValueError if it exceeds max_bytes."""
        directory = Path(meta["dir"])
        written = 0
        for i in range(total_chunks):
            part = directory / f"{i:06d}.part"
            written += part.stat().st_size
            if written > max_bytes:
                raise ValueError("assembled size exceeds limit")
            with open(part, "rb") as pf:
                shutil.copyfileobj(pf, target, length=_COPY_BUF)
        return written

    def discard(self, upload_id: str) -> None:
        """Remove an upload's parts + metadata. Idempotent."""
        meta = self._uploads.pop(upload_id, None)
        target = Path(meta["dir"]) if meta else self._dir(upload_id)
        shutil.rmtree(target, ignore_errors=True)

    def cleanup_expired(self) -> None:
        """Drop uploads abandoned mid-way (older than the TTL), including orphan
        dirs left on disk by a previous process."""
        now = time.time()
        for upload_id, meta in list(self._uploads.items()):
            if now - meta["created"] > self.ttl_sec:
                self.discard(upload_id)
        try:
            for directory in self.base_dir.iterdir():
                if directory.name in self._uploads:
                    continue
                try:
                    if now - directory.stat().st_mtime > self.ttl_sec:
                        shutil.rmtree(directory, ignore_errors=True)
                except OSError:
                    pass
        except FileNotFoundError:
            pass

    def cleanup_all(self) -> None:
        shutil.rmtree(self.base_dir, ignore_errors=True)
        self._uploads.clear()
