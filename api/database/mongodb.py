"""
api/database/mongodb.py — Async Motor client, index management, and CRUD helpers.

Usage in FastAPI lifespan:

    from api.database.mongodb import init_db, close_db, get_db

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db(MONGODB_URI, MONGODB_DB_NAME)
        yield
        await close_db()
"""
from __future__ import annotations

import logging
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from .models import Incident, RecognitionRecord, RecognitionSession

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None

SESSIONS_COL = "recognition_sessions"
RECORDS_COL = "recognition_records"
INCIDENTS_COL = "incidents"


# ── Connection lifecycle ──────────────────────────────────────────────────────

async def init_db(uri: str, db_name: str) -> None:
    """Connect to MongoDB Atlas and ensure all indexes exist."""
    global _client, _db
    _client = AsyncIOMotorClient(uri)
    _db = _client[db_name]
    await _ensure_indexes(_db)
    logger.info("MongoDB connected — db=%s", db_name)


async def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB not initialised. Call init_db() first.")
    return _db


# ── Index strategy ────────────────────────────────────────────────────────────
#
# recognition_sessions
#   - session_id       unique  — primary lookup key from API routes
#   - status                   — dashboard/admin "show processing jobs"
#   - created_at desc          — history feed, pagination
#
# recognition_records
#   - session_id               — "all plates for job X"
#   - plate_text               — search by plate number
#   - vehicle_class            — filter by vehicle type
#   - created_at desc          — recent recognitions
#   - (session_id, track_id)   unique compound — dedup / upsert guard

async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    await db[SESSIONS_COL].create_indexes([
        IndexModel([("session_id", ASCENDING)], unique=True, name="uq_session_id"),
        IndexModel([("status", ASCENDING)], name="ix_status"),
        IndexModel([("created_at", DESCENDING)], name="ix_created_at_desc"),
    ])

    await db[RECORDS_COL].create_indexes([
        IndexModel([("session_id", ASCENDING)], name="ix_session_id"),
        IndexModel([("plate_text", ASCENDING)], name="ix_plate_text"),
        IndexModel([("vehicle_class", ASCENDING)], name="ix_vehicle_class"),
        IndexModel([("created_at", DESCENDING)], name="ix_created_at_desc"),
        IndexModel(
            [("session_id", ASCENDING), ("track_id", ASCENDING)],
            unique=True,
            name="uq_session_track",
        ),
    ])

    await db[INCIDENTS_COL].create_indexes([
        IndexModel([("incident_id", ASCENDING)], unique=True, name="uq_incident_id"),
        IndexModel([("session_id", ASCENDING)], name="ix_session_id"),
        IndexModel([("marked_at", DESCENDING)], name="ix_marked_at_desc"),
        IndexModel([("status", ASCENDING)], name="ix_status"),
        IndexModel([("source_type", ASCENDING)], name="ix_source_type"),
    ])
    logger.info("MongoDB indexes ensured.")


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def is_db_configured() -> bool:
    """Return True when the Motor client is initialised and ready."""
    return _db is not None


async def insert_session(session: RecognitionSession) -> str:
    """Insert a new session document. Returns the inserted _id as hex string."""
    db = get_db()
    doc = session.model_dump(by_alias=True, exclude={"id"})
    result = await db[SESSIONS_COL].insert_one(doc)
    return str(result.inserted_id)


async def upsert_session(session: RecognitionSession) -> None:
    """Insert or update a session document matched by session_id."""
    from datetime import datetime, timezone

    db = get_db()
    doc = session.model_dump(by_alias=True, exclude={"id"})
    doc["updated_at"] = datetime.now(timezone.utc)
    await db[SESSIONS_COL].update_one(
        {"session_id": session.session_id},
        {"$set": doc},
        upsert=True,
    )


async def update_session(session_id: str, patch: dict[str, Any]) -> None:
    """Partially update a session by session_id."""
    from datetime import datetime, timezone

    db = get_db()
    patch.setdefault("updated_at", datetime.now(timezone.utc))
    await db[SESSIONS_COL].update_one(
        {"session_id": session_id},
        {"$set": patch},
    )


async def get_session(session_id: str) -> RecognitionSession | None:
    db = get_db()
    doc = await db[SESSIONS_COL].find_one({"session_id": session_id})
    if doc is None:
        return None
    return RecognitionSession.model_validate(doc)


async def insert_record(record: RecognitionRecord) -> str:
    """Insert a recognition record. Returns the inserted _id as hex string."""
    db = get_db()
    doc = record.model_dump(by_alias=True, exclude={"id"})
    result = await db[RECORDS_COL].insert_one(doc)
    return str(result.inserted_id)


async def upsert_record(record: RecognitionRecord) -> None:
    """
    Insert or replace a record matched by (session_id, track_id).

    Safe to call multiple times as a track is finalised — idempotent.
    """
    from datetime import datetime, timezone

    db = get_db()
    doc = record.model_dump(by_alias=True, exclude={"id"})
    doc["updated_at"] = datetime.now(timezone.utc)
    await db[RECORDS_COL].update_one(
        {"session_id": record.session_id, "track_id": record.track_id},
        {"$set": doc},
        upsert=True,
    )


async def get_record_by_track(session_id: str, track_id: int) -> RecognitionRecord | None:
    """Return a single recognition record by (session_id, track_id), or None."""
    db = get_db()
    doc = await db[RECORDS_COL].find_one({"session_id": session_id, "track_id": track_id})
    return RecognitionRecord.model_validate(doc) if doc else None


async def get_records_for_session(session_id: str) -> list[RecognitionRecord]:
    """Return all recognition records belonging to a session."""
    db = get_db()
    cursor = db[RECORDS_COL].find({"session_id": session_id})
    return [RecognitionRecord.model_validate(doc) async for doc in cursor]


async def search_by_plate(plate_text: str) -> list[RecognitionRecord]:
    """
    Exact-match search on plate_text.

    For prefix / partial search, callers should use a $regex query directly
    on the collection (not $where — avoids NoSQL injection risk).
    """
    db = get_db()
    cursor = db[RECORDS_COL].find({"plate_text": plate_text})
    return [RecognitionRecord.model_validate(doc) async for doc in cursor]


# ── Incident CRUD ─────────────────────────────────────────────────────────────


async def upsert_incident(incident: Incident) -> None:
    """Insert or replace an incident document, matched by incident_id."""
    from datetime import datetime, timezone

    db = get_db()
    doc = incident.model_dump(by_alias=True)
    doc["updated_at"] = datetime.now(timezone.utc)
    await db[INCIDENTS_COL].update_one(
        {"incident_id": incident.incident_id},
        {"$set": doc},
        upsert=True,
    )


async def get_incident(incident_id: str) -> Incident | None:
    db = get_db()
    doc = await db[INCIDENTS_COL].find_one({"incident_id": incident_id})
    return Incident.model_validate(doc) if doc else None


async def list_incidents(
    *,
    session_id: str | None = None,
    source_type: str | None = None,
    limit: int = 50,
) -> list[Incident]:
    db = get_db()
    query: dict = {}
    if session_id is not None:
        query["session_id"] = session_id
    if source_type is not None:
        query["source_type"] = source_type
    cursor = db[INCIDENTS_COL].find(query).sort("marked_at", DESCENDING).limit(limit)
    return [Incident.model_validate(doc) async for doc in cursor]

