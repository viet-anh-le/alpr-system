"""Integration tests for events collection CRUD.

Skipped automatically when MONGODB_URI is unset."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from api.database import mongodb
from api.database.models import MonitorEvent

pytestmark = pytest.mark.skipif(
    "MONGODB_URI" not in os.environ, reason="MONGODB_URI not set"
)


@pytest.fixture(scope="module")
async def db_initialised():
    await mongodb.init_db(os.environ["MONGODB_URI"], "alpr_test")
    yield
    await mongodb.close_db()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_and_get_event(db_initialised):
    now = datetime.now(timezone.utc)
    event = MonitorEvent(
        event_id="evt_test_1",
        session_id="ses_test",
        source_type="live",
        source_ref="rtsp://localhost/test",
        marked_at=now,
        window_start_sec=0.0,
        window_end_sec=10.0,
        duration_sec=10.0,
        status="processing",
        created_at=now,
        updated_at=now,
    )
    await mongodb.upsert_event(event)
    fetched = await mongodb.get_event("evt_test_1")
    assert fetched is not None
    assert fetched.event_id == "evt_test_1"
    assert fetched.status == "processing"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_events_filters(db_initialised):
    items = await mongodb.list_events(source_type="live", limit=10)
    assert all(i.source_type == "live" for i in items)
