"""
core/video_processor.py — Frame-level OpenCV utilities.

Handles video I/O, vehicle cropping, and server-side bounding box rendering
for MJPEG streaming.
"""

from __future__ import annotations

import cv2
import numpy as np

_VEHICLE_PAD = 16

# BGR colours matching the React canvas overlay palette
_BOX_COLORS: dict[str, tuple[int, int, int]] = {
    "active":  (0, 210, 255),   # yellow  #FFD200
    "done":    (60, 220, 0),    # green   #00DC3C
    "tracked": (180, 180, 180), # grey    #B4B4B4
}
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICK = 1


def crop_vehicle(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return the vehicle region from frame, expanded by _VEHICLE_PAD pixels."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (int(c) for c in box)
    x1 = max(0, x1 - _VEHICLE_PAD)
    y1 = max(0, y1 - _VEHICLE_PAD)
    x2 = min(W, x2 + _VEHICLE_PAD)
    y2 = min(H, y2 + _VEHICLE_PAD)
    return frame[y1:y2, x1:x2]


def warp_plate_crop(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Return a perspective-corrected plate crop from 4 OBB corner points.

    Uses the OBB polygon directly instead of an axis-aligned bounding rect,
    so tilted plates are de-rotated and tightly cropped without background
    corners contaminating the OCR input.

    pts: shape (4, 2) integer pixel coordinates in any winding order,
         as returned by Ultralytics OBB xyxyxyxy.

    Returns an empty array (size == 0) when pts are degenerate.
    """
    src = pts.astype(np.float32)

    # Sort corners into TL, TR, BR, BL using the sum/diff trick:
    #   TL has the smallest (x+y), BR has the largest (x+y)
    #   TR has the smallest (y-x), BL has the largest (y-x)
    s = src.sum(axis=1)
    d = np.diff(src, axis=1).ravel()
    tl = src[np.argmin(s)]
    br = src[np.argmax(s)]
    tr = src[np.argmin(d)]
    bl = src[np.argmax(d)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    w = int(round(max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl),
    )))
    h = int(round(max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr),
    )))
    if w < 1 or h < 1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(frame, M, (w, h))


def draw_annotated_frame(frame: np.ndarray, boxes: list[dict]) -> bytes:
    """Draw bounding boxes + labels onto frame and return JPEG bytes.

    Each box dict: {id, box: [x1,y1,x2,y2], state, plate, cls}
    """
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(c) for c in b["box"])
        color      = _BOX_COLORS.get(b.get("state", "tracked"), _BOX_COLORS["tracked"])
        line_width = 3 if b.get("state") == "active" else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        label = f"{b.get('cls', '')} #{b.get('id', '')}"
        if b.get("plate"):
            label += f"  {b['plate']}"

        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        lx = x1
        ly = max(y1 - th - baseline - 6, 0)
        cv2.rectangle(out, (lx, ly), (lx + tw + 8, ly + th + baseline + 4), color, -1)
        cv2.putText(out, label, (lx + 4, ly + th + 2), _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK, cv2.LINE_AA)

    _, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return bytes(jpg)
