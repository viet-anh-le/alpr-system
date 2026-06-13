"""Tests for the Incident / IncidentVehicle Pydantic models."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from api.database.models import Incident, IncidentVehicle


@pytest.mark.unit
def test_incident_vehicle_minimal_construction():
    v = IncidentVehicle(
        track_id=7,
        plate_text="30A-12345",
        plate_text_confidence=0.94,
        chars=[("3", 0.99), ("0", 0.97)],
        vehicle_class="car",
        plate_image_url=None,
        vehicle_image_url=None,
        ocr_method="segment_vote",
        ocr_frames=18,
        first_seen_frame=4,
        last_seen_frame=142,
    )
    assert v.track_id == 7
    assert v.ocr_method == "segment_vote"


@pytest.mark.unit
def test_incident_default_status_and_lists():
    now = datetime.now(timezone.utc)
    i = Incident(
        incident_id="inc_abc",
        session_id="ses_xyz",
        source_type="live",
        source_ref="rtsp://10.0.0.5/main",
        marked_at=now,
        window_start_sec=0.0,
        window_end_sec=10.0,
        duration_sec=10.0,
        status="processing",
        created_at=now,
        updated_at=now,
    )
    assert i.vehicles == []
    assert i.total_vehicles == 0
    assert i.error_message is None


@pytest.mark.unit
def test_incident_rejects_bad_source_type():
    now = datetime.now(timezone.utc)
    with pytest.raises(Exception):
        Incident(
            incident_id="inc_abc",
            session_id="ses_xyz",
            source_type="invalid",  # not "live" or "upload"
            source_ref="x",
            marked_at=now,
            window_start_sec=0.0,
            window_end_sec=1.0,
            duration_sec=1.0,
            status="processing",
            created_at=now,
            updated_at=now,
        )
