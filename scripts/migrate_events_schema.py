"""Migrate MongoDB monitor records from incidents schema to events schema.

The migration copies documents from the legacy `incidents` collection to the
new `events` collection, renaming `incident_id` to `event_id`. It deliberately
keeps the old collection intact as a rollback backup and does not move any
Supabase/Object Storage files; existing image URLs remain valid.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, IndexModel, MongoClient
from pymongo.collection import Collection

OLD_COLLECTION = "incidents"
NEW_COLLECTION = "events"


def transform_legacy_event_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one legacy document using the new events schema."""
    migrated = dict(doc)
    legacy_id = migrated.pop("incident_id", None)
    if not legacy_id:
        raise ValueError("legacy document is missing incident_id")
    migrated["event_id"] = legacy_id
    return migrated


def ensure_event_indexes(collection: Collection) -> None:
    collection.create_indexes([
        IndexModel([("event_id", ASCENDING)], unique=True, name="uq_event_id"),
        IndexModel([("session_id", ASCENDING)], name="ix_session_id"),
        IndexModel([("marked_at", DESCENDING)], name="ix_marked_at_desc"),
        IndexModel([("status", ASCENDING)], name="ix_status"),
        IndexModel([("source_type", ASCENDING)], name="ix_source_type"),
    ])


def migrate_collection(source: Collection, target: Collection, *, dry_run: bool = False) -> int:
    migrated_count = 0
    for doc in source.find({}):
        new_doc = transform_legacy_event_document(doc)
        existing = target.find_one({"event_id": new_doc["event_id"]}, {"_id": 1})
        if existing is not None and existing.get("_id") != new_doc.get("_id"):
            new_doc["_id"] = existing["_id"]
        if not dry_run:
            target.replace_one({"event_id": new_doc["event_id"]}, new_doc, upsert=True)
        migrated_count += 1
    if not dry_run:
        ensure_event_indexes(target)
    return migrated_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy legacy incidents collection to events.")
    parser.add_argument("--dry-run", action="store_true", help="count documents without writing to MongoDB")
    parser.add_argument("--db-name", default=None, help="override MONGODB_DB_NAME")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    load_dotenv(project_root / "api" / ".env")
    uri = os.getenv("MONGODB_URI")
    db_name = args.db_name or os.getenv("MONGODB_DB_NAME")
    if not uri:
        raise SystemExit("MONGODB_URI is required")
    if not db_name:
        raise SystemExit("MONGODB_DB_NAME is required, or pass --db-name")

    client = MongoClient(uri)
    try:
        db = client[db_name]
        count = migrate_collection(
            db[OLD_COLLECTION],
            db[NEW_COLLECTION],
            dry_run=args.dry_run,
        )
        action = "Would migrate" if args.dry_run else "Migrated"
        print(f"{action} {count} documents from {OLD_COLLECTION} to {NEW_COLLECTION}.")
        if not args.dry_run:
            print(f"Kept legacy collection {OLD_COLLECTION} unchanged as backup.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
