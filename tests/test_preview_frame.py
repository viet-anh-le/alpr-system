from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest


@pytest.mark.unit
def test_make_preview_frame_event_resizes_and_scales_boxes() -> None:
    from api.core.preview_frame import make_preview_frame_event

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    event = make_preview_frame_event(
        frame,
        [
            {
                "id": 7,
                "box": [20, 10, 120, 60],
                "state": "active",
                "cls": "car",
                "plate": "30A-12345",
                "kind": "vehicle",
            }
        ],
        frame_index=42,
        max_width=100,
        jpeg_quality=70,
    )

    assert event["type"] == "frame"
    assert event["frame"] == 42
    assert event["source_width"] == 200
    assert event["source_height"] == 100
    assert event["image_width"] == 100
    assert event["image_height"] == 50
    assert event["boxes"] == [
        {
            "id": 7,
            "kind": "vehicle",
            "box": [10, 5, 60, 30],
            "state": "active",
            "cls": "car",
            "plate": "30A-12345",
            "label": "car #7 30A-12345",
        }
    ]


@pytest.mark.unit
def test_make_preview_frame_event_emits_decodable_jpeg() -> None:
    from api.core.preview_frame import make_preview_frame_event

    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    event = make_preview_frame_event(frame, [], frame_index=1)

    raw = base64.b64decode(event["b64"])
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[:2] == (24, 32)
