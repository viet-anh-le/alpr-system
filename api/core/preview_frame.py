"""Build SSE preview-frame events for frontend bounding-box overlays."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from .config import ALPR_PREVIEW_JPEG_QUALITY, ALPR_PREVIEW_MAX_WIDTH


def make_preview_frame_event(
    frame: np.ndarray,
    boxes: list[dict[str, Any]],
    *,
    frame_index: int,
    max_width: int = ALPR_PREVIEW_MAX_WIDTH,
    jpeg_quality: int = ALPR_PREVIEW_JPEG_QUALITY,
) -> dict[str, Any]:
    """Return a compact SSE event containing a preview JPEG and scaled boxes."""
    if frame is None or frame.size == 0:
        raise ValueError("Preview frame is empty")

    source_height, source_width = frame.shape[:2]
    preview = frame
    scale = 1.0

    if max_width > 0 and source_width > max_width:
        scale = max_width / float(source_width)
        image_width = int(round(source_width * scale))
        image_height = int(round(source_height * scale))
        preview = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    else:
        image_width = int(source_width)
        image_height = int(source_height)

    ok, jpg = cv2.imencode(
        ".jpg",
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not ok:
        raise ValueError("Could not encode preview frame")

    return {
        "type": "frame",
        "frame": int(frame_index),
        "b64": base64.b64encode(jpg).decode("ascii"),
        "image_width": image_width,
        "image_height": image_height,
        "source_width": int(source_width),
        "source_height": int(source_height),
        "boxes": [_scale_box(box, scale) for box in boxes],
    }


def _scale_box(box: dict[str, Any], scale: float) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(round(float(coord) * scale)) for coord in box["box"]]
    cls = str(box.get("cls") or "")
    track_id = box.get("id", "")
    plate = str(box.get("plate") or "")
    label = f"{cls} #{track_id}".strip()
    if plate:
        label = f"{label} {plate}".strip()

    return {
        "id": int(track_id) if _is_int_like(track_id) else track_id,
        "kind": str(box.get("kind") or "vehicle"),
        "box": [x1, y1, x2, y2],
        "state": str(box.get("state") or "tracked"),
        "cls": cls,
        "plate": plate,
        "label": label,
    }


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True
