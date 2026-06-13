# Perspective-Warp Plate Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the axis-aligned bounding-rect plate crop with a perspective-warp crop so that tilted plates are de-rotated and tightly cropped, eliminating the background-corner contamination that degrades OCR accuracy.

**Architecture:** A new `warp_plate_crop(frame, pts)` function in `api/core/video_processor.py` uses the four OBB corner points from the plate detector to compute a `getPerspectiveTransform` → `warpPerspective` crop. Both `api/core/pipeline_core.py` and `scripts/diag_ocr_frames.py` call this instead of `cv2.boundingRect` + rectangular slice. The `"box"` field (used by `TrajectoryAssociator` for IoU) still comes from `cv2.boundingRect` so association logic is unaffected.

**Tech Stack:** OpenCV (`cv2.getPerspectiveTransform`, `cv2.warpPerspective`), NumPy, pytest.

---

## Background

The plate OBB detector returns four corner points. The current code calls `cv2.boundingRect(pts)` to get an axis-aligned rectangle, adds `PLATE_PAD` padding, and slices the frame. When the plate is rotated, the axis-aligned rect is larger than the plate and includes background triangles at the corners. These extra pixels confuse SmallLPR — confirmed by running inference on a tight hand-crop of Track 15's plate (`29-M1-99973`, all_confident=True) vs the loose pipeline crop (garbage output, min_char_conf 0.14–0.28).

---

## File Structure

| File | Change |
|------|--------|
| `api/core/video_processor.py` | Add `warp_plate_crop(frame, pts) -> np.ndarray` |
| `api/core/pipeline_core.py` | Import + call `warp_plate_crop`; remove `PLATE_PAD` from crop path |
| `scripts/diag_ocr_frames.py` | Same import + call replacement |
| `tests/test_warp_plate_crop.py` | New: 6 unit tests for `warp_plate_crop` |
| `tests/test_pipeline_core_parity.py` | Re-capture golden if plate text changes |

---

## Task 1: Add `warp_plate_crop` to `video_processor.py`

**Files:**
- Modify: `api/core/video_processor.py`
- Create: `tests/test_warp_plate_crop.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_warp_plate_crop.py`:

```python
from __future__ import annotations
import numpy as np
import pytest
from api.core.video_processor import warp_plate_crop


def _frame() -> np.ndarray:
    return np.zeros((300, 400, 3), dtype=np.uint8)


def test_axis_aligned_rect_returns_correct_dimensions():
    # Axis-aligned box [x=50..150, y=60..80] → width≈100, height≈20
    pts = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert abs(result.shape[1] - 100) <= 2
    assert abs(result.shape[0] - 20) <= 2


def test_tilted_box_returns_non_empty():
    pts = np.array([[100, 50], [140, 40], [145, 60], [105, 70]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.size > 0
    assert result.ndim == 3


def test_degenerate_pts_returns_empty():
    pts = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.size == 0


def test_order_invariant():
    # Same rect given in reverse winding order → same output shape
    pts_a = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    pts_b = np.array([[150, 80], [50, 80], [50, 60], [150, 60]], dtype=np.int32)
    a = warp_plate_crop(_frame(), pts_a)
    b = warp_plate_crop(_frame(), pts_b)
    assert a.shape == b.shape


def test_output_dtype_is_uint8():
    pts = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    result = warp_plate_crop(_frame(), pts)
    assert result.dtype == np.uint8


def test_pixel_values_preserved():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[60:80, 50:150] = (0, 255, 0)   # green plate region
    pts = np.array([[50, 60], [150, 60], [150, 80], [50, 80]], dtype=np.int32)
    result = warp_plate_crop(frame, pts)
    assert result[:, :, 1].mean() > 200   # green channel dominates
```

- [ ] **Step 2: Run the tests — verify they all FAIL**

```bash
pytest tests/test_warp_plate_crop.py -v 2>&1 | tail -15
```

Expected: `ImportError: cannot import name 'warp_plate_crop'`

- [ ] **Step 3: Implement `warp_plate_crop` in `api/core/video_processor.py`**

Append after the existing `crop_vehicle` function (after line 34, before the `draw_annotated_frame` function):

```python
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
```

- [ ] **Step 4: Run the tests — verify all 6 PASS**

```bash
pytest tests/test_warp_plate_crop.py -v 2>&1 | tail -15
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add api/core/video_processor.py tests/test_warp_plate_crop.py
git commit -m "feat(crop): add warp_plate_crop — perspective-correct OBB crops"
```

---

## Task 2: Update `pipeline_core.py` to use `warp_plate_crop`

**Files:**
- Modify: `api/core/pipeline_core.py:34-40` (imports) and `pipeline_core.py:229-249` (crop section)
- Modify: `tests/test_pipeline_core_parity.py` (re-capture golden if plate text changes)

- [ ] **Step 1: Update the import in `pipeline_core.py`**

Find lines 37–40 (the `from .video_processor import` block):

```python
from .video_processor import (
    crop_vehicle as _crop_vehicle,
    draw_annotated_frame as _draw_annotated_frame,
)
```

Replace with:

```python
from .video_processor import (
    crop_vehicle as _crop_vehicle,
    draw_annotated_frame as _draw_annotated_frame,
    warp_plate_crop as _warp_plate_crop,
)
```

Also remove `PLATE_PAD` from the `from .config import` block (lines 22–30), since it is no longer used in this file:

```python
from .config import (
    FRAME_STRIDE,
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
    TOP_K_FRAMES,
    VEHICLE_CLASSES,
)
```

- [ ] **Step 2: Replace the crop section in `process_frames`**

Find lines 229–249 in `api/core/pipeline_core.py`:

```python
            for pts, det_conf, p_tid in zip(obb_pts, obb_conf, obb_ids):
                if float(det_conf) < PLATE_DET_CONF:
                    continue
                raw_rx, raw_ry, raw_rw, raw_rh = cv2.boundingRect(pts)
                if raw_rw < MIN_PLATE_W or raw_rh < MIN_PLATE_H:
                    continue
                rx = max(0, raw_rx - PLATE_PAD)
                ry = max(0, raw_ry - PLATE_PAD)
                rw = min(raw_rw + 2 * PLATE_PAD, W - rx)
                rh = min(raw_rh + 2 * PLATE_PAD, H - ry)
                plate_crop = frame[ry : ry + rh, rx : rx + rw]
                if plate_crop.size == 0:
                    continue
                if not is_sharp(plate_crop):
                    continue
                plate_tracks.append({
                    "id": int(p_tid),
                    "box": [rx, ry, rx + rw, ry + rh],
                    "crop": plate_crop,
                    "conf": float(det_conf),
                })
```

Replace with:

```python
            for pts, det_conf, p_tid in zip(obb_pts, obb_conf, obb_ids):
                if float(det_conf) < PLATE_DET_CONF:
                    continue
                raw_rx, raw_ry, raw_rw, raw_rh = cv2.boundingRect(pts)
                if raw_rw < MIN_PLATE_W or raw_rh < MIN_PLATE_H:
                    continue
                plate_crop = _warp_plate_crop(frame, pts)
                if plate_crop.size == 0:
                    continue
                if not is_sharp(plate_crop):
                    continue
                plate_tracks.append({
                    "id": int(p_tid),
                    "box": [raw_rx, raw_ry, raw_rx + raw_rw, raw_ry + raw_rh],
                    "crop": plate_crop,
                    "conf": float(det_conf),
                })
```

Key changes:
- `_warp_plate_crop(frame, pts)` replaces the 5-line boundingRect + padding + slice
- `"box"` now uses `raw_rx/raw_ry/raw_rw/raw_rh` (no padding) — TrajectoryAssociator uses this for IoU matching; raw is correct
- `PLATE_PAD` is entirely removed from this path

- [ ] **Step 3: Re-capture the golden file**

The warp produces tighter crops so OCR results may change for some tracks.

```bash
python tests/_capture_golden.py
```

Then inspect changes:

```bash
git diff tests/fixtures/golden_run_job_events.json
```

If the diff shows plate text changes (not just key reordering), confirm they are improvements or at worst neutral, then stage the updated golden.

- [ ] **Step 4: Run parity + warp tests**

```bash
pytest tests/test_pipeline_core_parity.py tests/test_warp_plate_crop.py tests/test_track_buffer.py -v --tb=short 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/core/pipeline_core.py tests/fixtures/golden_run_job_events.json
git commit -m "feat(pipeline): use perspective-warp crop instead of axis-aligned bbox

Replaces cv2.boundingRect + PLATE_PAD slice with warp_plate_crop(), which
uses the four OBB corner points to compute a tight, de-rotated plate image.
Eliminates background-corner contamination that degraded OCR on tilted plates
(confirmed on Track 15 / 29-M1-99973 in hn_oto_18.mp4).
"
```

---

## Task 3: Update `diag_ocr_frames.py` and verify Track 15

**Files:**
- Modify: `scripts/diag_ocr_frames.py`

- [ ] **Step 1: Update the imports in `diag_ocr_frames.py`**

Find the import block at the top of `scripts/diag_ocr_frames.py`. It currently imports:

```python
from api.core.config import (
    CHARS,
    CONF_THRESHOLD,
    FRAME_STRIDE,
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
    PLATE_PAD,
    TOP_K_FRAMES,
    VEHICLE_CLASSES,
)
```

Remove `PLATE_PAD` (no longer used after the crop change):

```python
from api.core.config import (
    CHARS,
    CONF_THRESHOLD,
    FRAME_STRIDE,
    MIN_PLATE_H,
    MIN_PLATE_W,
    PLATE_DET_CONF,
    TOP_K_FRAMES,
    VEHICLE_CLASSES,
)
```

Add the `warp_plate_crop` import below the existing `from api.core.video_processor import crop_vehicle` line:

```python
from api.core.video_processor import crop_vehicle, warp_plate_crop
```

- [ ] **Step 2: Replace the crop section in `run()`**

Find the plate crop loop in `run()` (currently around lines 155–178):

```python
                raw_rx, raw_ry, raw_rw, raw_rh = cv2.boundingRect(pts)
                if raw_rw < MIN_PLATE_W or raw_rh < MIN_PLATE_H:
                    gate_counts["size_fail"] += 1
                    continue
                rx = max(0, raw_rx - PLATE_PAD)
                ry = max(0, raw_ry - PLATE_PAD)
                rw = min(raw_rw + 2 * PLATE_PAD, W - rx)
                rh = min(raw_rh + 2 * PLATE_PAD, H - ry)
                plate_crop = frame[ry:ry+rh, rx:rx+rw]
                if plate_crop.size == 0:
                    continue
```

Replace with:

```python
                raw_rx, raw_ry, raw_rw, raw_rh = cv2.boundingRect(pts)
                if raw_rw < MIN_PLATE_W or raw_rh < MIN_PLATE_H:
                    gate_counts["size_fail"] += 1
                    continue
                plate_crop = warp_plate_crop(frame, pts)
                if plate_crop.size == 0:
                    continue
```

Also update the `plate_tracks.append` call's `"box"` field (a few lines below) to use raw coordinates (no padding):

```python
                plate_tracks.append({
                    "id": int(p_tid),
                    "box": [raw_rx, raw_ry, raw_rx+raw_rw, raw_ry+raw_rh],
                    "crop": plate_crop,
                    "conf": float(det_conf),
                })
```

- [ ] **Step 3: Run the diagnostic and confirm Track 15 improves**

```bash
python scripts/diag_ocr_frames.py data/realworld-videos/chunks/hn_oto_18.mp4 2>/dev/null \
  | grep -A 30 "Track  *15 \b"
```

Expected: Track 15 per-frame OCR results should now show higher character confidences and consistent plate text (previously all chars had min_char_conf 0.14–0.28 with 5 different plate texts across 5 frames).

Also verify Track 6 still reads correctly (should still get `30G-51827`):

```bash
python scripts/diag_ocr_frames.py data/realworld-videos/chunks/hn_oto_18.mp4 2>/dev/null \
  | grep -A 5 "FINALISE track 6"
```

Expected: `Voted result: '30G-51827'  valid=True`

- [ ] **Step 4: Commit**

```bash
git add scripts/diag_ocr_frames.py
git commit -m "fix(diag): use warp_plate_crop in diagnostic script"
```

---

## Self-Review

**Spec coverage:**
- [x] `warp_plate_crop` uses OBB corner points — Task 1
- [x] Perspective transform de-rotates tilted plates — Task 1 (`getPerspectiveTransform` + `warpPerspective`)
- [x] `pipeline_core.py` uses the new crop — Task 2
- [x] `PLATE_PAD` removed from crop path (no longer needed) — Task 2
- [x] `"box"` field for TrajectoryAssociator uses raw bounding rect (correct for IoU) — Task 2
- [x] `diag_ocr_frames.py` updated — Task 3
- [x] Track 15 verified — Task 3, Step 3
- [x] Track 6 regression check — Task 3, Step 3

**Placeholder scan:** None found.

**Type consistency:** `warp_plate_crop(frame: np.ndarray, pts: np.ndarray) -> np.ndarray` used identically in all three tasks.
