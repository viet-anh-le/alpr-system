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
import re
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from .models import AuthSession, MonitorEvent, RecognitionRecord, RecognitionSession, User

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None

SESSIONS_COL = "recognition_sessions"
RECORDS_COL = "recognition_records"
EVENTS_COL = "events"
USERS_COL = "users"
AUTH_SESSIONS_COL = "auth_sessions"


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
    await db[USERS_COL].create_indexes([
        IndexModel([("email", ASCENDING)], unique=True, name="uq_email"),
        IndexModel([("created_at", DESCENDING)], name="ix_created_at_desc"),
    ])

    await db[AUTH_SESSIONS_COL].create_indexes([
        IndexModel([("session_id", ASCENDING)], unique=True, name="uq_session_id"),
        IndexModel([("user_id", ASCENDING)], name="ix_user_id"),
        IndexModel([("expires_at", ASCENDING)], name="ix_expires_at"),
    ])

    await db[SESSIONS_COL].create_indexes([
        IndexModel([("session_id", ASCENDING)], unique=True, name="uq_session_id"),
        IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="ix_user_created_at"),
        IndexModel([("status", ASCENDING)], name="ix_status"),
        IndexModel([("created_at", DESCENDING)], name="ix_created_at_desc"),
    ])

    await db[RECORDS_COL].create_indexes([
        IndexModel([("session_id", ASCENDING)], name="ix_session_id"),
        IndexModel([("user_id", ASCENDING), ("session_id", ASCENDING)], name="ix_user_session"),
        IndexModel([("plate_text", ASCENDING)], name="ix_plate_text"),
        IndexModel([("vehicle_track_id", ASCENDING)], name="ix_vehicle_track_id"),
        IndexModel([("plate_track_id", ASCENDING)], name="ix_plate_track_id"),
        IndexModel([("vehicle_class", ASCENDING)], name="ix_vehicle_class"),
        IndexModel([("created_at", DESCENDING)], name="ix_created_at_desc"),
        IndexModel(
            [("session_id", ASCENDING), ("track_id", ASCENDING)],
            unique=True,
            name="uq_session_track",
        ),
    ])

    await db[EVENTS_COL].create_indexes([
        IndexModel([("event_id", ASCENDING)], unique=True, name="uq_event_id"),
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


async def create_user(user: User) -> str:
    """Insert a user and return the inserted _id as hex string."""
    db = get_db()
    doc = user.model_dump(by_alias=True, exclude={"id"})
    result = await db[USERS_COL].insert_one(doc)
    return str(result.inserted_id)


async def get_user_by_email(email: str) -> User | None:
    db = get_db()
    doc = await db[USERS_COL].find_one({"email": email.strip().lower()})
    return User.model_validate(doc) if doc else None


async def get_user_by_id(user_id: str) -> User | None:
    db = get_db()
    if not ObjectId.is_valid(user_id):
        return None
    doc = await db[USERS_COL].find_one({"_id": ObjectId(user_id)})
    return User.model_validate(doc) if doc else None


async def create_auth_session(session: AuthSession) -> str:
    db = get_db()
    doc = session.model_dump(by_alias=True, exclude={"id"})
    result = await db[AUTH_SESSIONS_COL].insert_one(doc)
    return str(result.inserted_id)


async def get_auth_session(session_id: str) -> AuthSession | None:
    db = get_db()
    doc = await db[AUTH_SESSIONS_COL].find_one({"session_id": session_id})
    return AuthSession.model_validate(doc) if doc else None


async def revoke_auth_session(session_id: str) -> None:
    from datetime import datetime, timezone

    db = get_db()
    await db[AUTH_SESSIONS_COL].update_one(
        {"session_id": session_id},
        {"$set": {"revoked": True, "updated_at": datetime.now(timezone.utc)}},
    )


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


async def get_session_for_user(session_id: str, user_id: str) -> RecognitionSession | None:
    db = get_db()
    doc = await db[SESSIONS_COL].find_one({"session_id": session_id, "user_id": user_id})
    return RecognitionSession.model_validate(doc) if doc else None


async def count_sessions_for_user(user_id: str) -> int:
    db = get_db()
    return int(await db[SESSIONS_COL].count_documents({"user_id": user_id}))


async def list_sessions_for_user(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[RecognitionSession]:
    db = get_db()
    safe_limit = max(1, min(int(limit), 100))
    safe_offset = max(0, int(offset))
    cursor = (
        db[SESSIONS_COL]
        .find({"user_id": user_id})
        .sort("created_at", DESCENDING)
        .skip(safe_offset)
        .limit(safe_limit)
    )
    return [RecognitionSession.model_validate(doc) async for doc in cursor]


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


async def get_record_by_track_for_user(
    session_id: str,
    track_id: int,
    user_id: str,
) -> RecognitionRecord | None:
    """Return one recognition record only when owned by user_id."""
    db = get_db()
    doc = await db[RECORDS_COL].find_one({
        "session_id": session_id,
        "track_id": track_id,
        "user_id": user_id,
    })
    return RecognitionRecord.model_validate(doc) if doc else None


async def get_records_for_session(session_id: str) -> list[RecognitionRecord]:
    """Return all recognition records belonging to a session."""
    db = get_db()
    cursor = db[RECORDS_COL].find({"session_id": session_id})
    return [RecognitionRecord.model_validate(doc) async for doc in cursor]


async def get_records_for_session_for_user(
    session_id: str,
    user_id: str,
) -> list[RecognitionRecord]:
    """Return all recognition records for a user-owned session."""
    db = get_db()
    cursor = (
        db[RECORDS_COL]
        .find({"session_id": session_id, "user_id": user_id})
        .sort("track_id", ASCENDING)
    )
    return [RecognitionRecord.model_validate(doc) async for doc in cursor]

def _records_query(
    user_id: str,
    *,
    session_id: str | None = None,
    plate: str | None = None,
    vehicle_class: str | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    if vehicle_class:
        query["vehicle_class"] = vehicle_class
    if plate:
        query["plate_text"] = {
            "$regex": re.escape(plate.strip()),
            "$options": "i",
        }
    return query


async def list_records_for_user(
    user_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
    plate: str | None = None,
    vehicle_class: str | None = None,
) -> list[RecognitionRecord]:
    db = get_db()
    safe_limit = max(1, min(int(limit), 100))
    safe_offset = max(0, int(offset))
    cursor = (
        db[RECORDS_COL]
        .find(
            _records_query(
                user_id,
                session_id=session_id,
                plate=plate,
                vehicle_class=vehicle_class,
            )
        )
        .sort([("created_at", DESCENDING), ("session_id", ASCENDING), ("track_id", ASCENDING)])
        .skip(safe_offset)
        .limit(safe_limit)
    )
    return [RecognitionRecord.model_validate(doc) async for doc in cursor]


async def count_records_for_user(
    user_id: str,
    *,
    session_id: str | None = None,
    plate: str | None = None,
    vehicle_class: str | None = None,
) -> int:
    db = get_db()
    return int(await db[RECORDS_COL].count_documents(
        _records_query(
            user_id,
            session_id=session_id,
            plate=plate,
            vehicle_class=vehicle_class,
        )
    ))


async def summarize_records_for_user(
    user_id: str,
    *,
    session_id: str | None = None,
    plate: str | None = None,
    vehicle_class: str | None = None,
    top_limit: int = 8,
) -> dict[str, Any]:
    db = get_db()
    query = _records_query(
        user_id,
        session_id=session_id,
        plate=plate,
        vehicle_class=vehicle_class,
    )
    safe_top_limit = max(1, min(int(top_limit), 20))
    total_records = int(await db[RECORDS_COL].count_documents(query))
    vehicle_counts_raw = await db[RECORDS_COL].aggregate([
        {"$match": query},
        {"$group": {"_id": "$vehicle_class", "count": {"$sum": 1}}},
        {"$sort": {"count": -1, "_id": 1}},
    ]).to_list(length=None)
    top_plates_raw = await db[RECORDS_COL].aggregate([
        {"$match": query},
        {
            "$group": {
                "_id": "$plate_text",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$plate_text_confidence"},
            }
        },
        {"$sort": {"count": -1, "_id": 1}},
        {"$limit": safe_top_limit},
    ]).to_list(length=None)
    unique_plates = await db[RECORDS_COL].aggregate([
        {"$match": query},
        {"$group": {"_id": "$plate_text"}},
        {"$count": "value"},
    ]).to_list(length=1)

    return {
        "total_records": total_records,
        "unique_plates": int(unique_plates[0]["value"]) if unique_plates else 0,
        "vehicle_counts": [
            {"vehicle_class": item["_id"], "count": int(item["count"])}
            for item in vehicle_counts_raw
        ],
        "top_plates": [
            {
                "plate_text": item["_id"],
                "count": int(item["count"]),
                "avg_confidence": round(float(item.get("avg_confidence") or 0.0), 4),
            }
            for item in top_plates_raw
        ],
    }

async def search_by_plate(plate_text: str) -> list[RecognitionRecord]:
    """
    Exact-match search on plate_text.

    For prefix / partial search, callers should use a $regex query directly
    on the collection (not $where — avoids NoSQL injection risk).
    """
    db = get_db()
    cursor = db[RECORDS_COL].find({"plate_text": plate_text})
    return [RecognitionRecord.model_validate(doc) async for doc in cursor]


# ── Event CRUD ─────────────────────────────────────────────────────────────


async def upsert_event(event: MonitorEvent) -> None:
    """Insert or replace an event document, matched by event_id."""
    from datetime import datetime, timezone

    db = get_db()
    doc = event.model_dump(by_alias=True)
    doc["updated_at"] = datetime.now(timezone.utc)
    await db[EVENTS_COL].update_one(
        {"event_id": event.event_id},
        {"$set": doc},
        upsert=True,
    )


async def get_event(event_id: str) -> MonitorEvent | None:
    db = get_db()
    doc = await db[EVENTS_COL].find_one({"event_id": event_id})
    return MonitorEvent.model_validate(doc) if doc else None


async def list_events(
    *,
    session_id: str | None = None,
    source_type: str | None = None,
    limit: int = 50,
) -> list[MonitorEvent]:
    db = get_db()
    query: dict = {}
    if session_id is not None:
        query["session_id"] = session_id
    if source_type is not None:
        query["source_type"] = source_type
    cursor = db[EVENTS_COL].find(query).sort("marked_at", DESCENDING).limit(limit)
    return [MonitorEvent.model_validate(doc) async for doc in cursor]
