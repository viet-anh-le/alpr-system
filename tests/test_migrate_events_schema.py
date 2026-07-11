from __future__ import annotations

import importlib.util
from pathlib import Path

from bson import ObjectId


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_events_schema.py"
_SPEC = importlib.util.spec_from_file_location("migrate_events_schema", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
transform_legacy_event_document = _MODULE.transform_legacy_event_document


def test_transform_legacy_event_document_renames_id_without_mutating_input():
    original = {
        "_id": ObjectId(),
        "incident_id": "inc_legacy",
        "session_id": "mon_123",
        "vehicles": [],
    }

    migrated = transform_legacy_event_document(original)

    assert original["incident_id"] == "inc_legacy"
    assert "incident_id" not in migrated
    assert migrated["event_id"] == "inc_legacy"
    assert migrated["session_id"] == "mon_123"


def test_transform_legacy_event_document_requires_legacy_id():
    try:
        transform_legacy_event_document({"session_id": "mon_123"})
    except ValueError as exc:
        assert "incident_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")
