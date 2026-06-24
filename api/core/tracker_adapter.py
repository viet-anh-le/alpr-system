"""
core/tracker_adapter.py — Thin wrapper around boxmot.BotSort.

Why: Ultralytics' built-in BoT-SORT wraps any ReID `model:` path with YOLO()
and forces 640x640 preprocessing. Our ReID (MobileNetV3-Small, 256x128 input)
is incompatible with that. BoxMOT decouples detection from tracking and
handles ReID crop/preprocess/embed natively.

Plate tracking (Ultralytics ByteTrack on OBB) is unaffected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from boxmot.reid.core.config import MODEL_TYPES
from boxmot.reid.core.reid import ReID

from .botsort_reid import AlwaysReIDBotSort

logger = logging.getLogger(__name__)


def _to_boxmot_device(device: str) -> str:
    """Convert torch-style device string to boxmot format.

    boxmot's select_device rejects "cuda" — it requires "0", "cpu", "0,1", etc.
      "cpu"    → "cpu"
      "cuda"   → "0"
      "cuda:0" → "0"
      "cuda:1" → "1"
    """
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device  # already a plain index or other valid form


# FIXME: boxmot (v12–v19) infers the ReID architecture from the ONNX filename
# by scanning MODEL_TYPES against Path.name.  Remove this workaround once
# boxmot supports explicit arch config (track upstream boxmot issue).
_FALLBACK_ARCH = "mobilenetv2_x1_0"


def _ensure_recognised_name(weights: Path) -> Path:
    """Return a path whose stem contains a boxmot-recognised architecture name.

    If the provided path already satisfies the requirement, it is returned as-is.
    Otherwise a relative symlink sibling is created (once) and that path is returned.
    The symlink has no effect on ONNX inference — boxmot's ONNX backend loads the
    session directly from the file; the architecture name is only used to build a
    reference PyTorch model for input-shape discovery (which is then discarded).
    """
    name = weights.name
    if any(arch in name for arch in MODEL_TYPES):
        return weights

    new_name = f"{_FALLBACK_ARCH}_{name}"
    link_path = weights.parent / new_name
    if not link_path.exists():
        os.symlink(weights.name, link_path)
        logger.debug("Created symlink %s → %s for boxmot arch detection", link_path.name, name)
    return link_path


class VehicleTracker:
    """Thin adapter around boxmot.BotSort for vehicle tracking with ReID.

    Accepts standard YOLO-style detections and returns per-frame track arrays.
    """

    # BotSort hyperparameters — defined as class constants to avoid magic numbers.
    _TRACK_HIGH_THRESH: float = 0.3
    _TRACK_LOW_THRESH: float = 0.1
    _NEW_TRACK_THRESH: float = 0.7
    _TRACK_BUFFER: int = 30
    _MATCH_THRESH: float = 0.7
    _PROXIMITY_THRESH: float = 0.5
    _APPEARANCE_THRESH: float = 0.25

    def __init__(
        self,
        reid_weights: Path,
        device: str = "cuda:0",
        half: bool = False,
    ) -> None:
        """Initialise BotSort with the given ReID weights.

        Args:
            reid_weights: Path to the ReID ONNX model (expects (B, 3, 256, 128) input).
            device: Torch device string, e.g. "cuda:0" or "cpu".
            half: Whether to use FP16 inference for the ReID model.
        """
        self._reid_weights = _ensure_recognised_name(Path(reid_weights))
        self._device = _to_boxmot_device(device)
        self._half = half

        # BotSort expects the backend model (ReID(...).model), not the ReID wrapper
        reid_model = ReID(path=self._reid_weights, device=self._device, half=half)
        self._tracker = AlwaysReIDBotSort(
            reid_model=reid_model.model,
            track_high_thresh=self._TRACK_HIGH_THRESH,
            track_low_thresh=self._TRACK_LOW_THRESH,
            new_track_thresh=self._NEW_TRACK_THRESH,
            track_buffer=self._TRACK_BUFFER,
            match_thresh=self._MATCH_THRESH,
            proximity_thresh=self._PROXIMITY_THRESH,
            appearance_thresh=self._APPEARANCE_THRESH,
        )
        logger.info(
            "VehicleTracker initialised with ReID weights: %s on device: %s",
            self._reid_weights,
            device,
        )

    def reset(self) -> None:
        """Reset tracker state between video jobs to avoid stale tracklets."""
        self._tracker.reset()

    def track(
        self,
        dets: np.ndarray,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update tracker with detections for the current frame.

        Args:
            dets: Detection array of shape (N, 6) = [x1, y1, x2, y2, conf, cls] float32.
            frame: BGR image as (H, W, 3) uint8 numpy array.

        Returns:
            Tuple of:
              - boxes: (M, 4) int32 array of [x1, y1, x2, y2] bounding boxes.
              - ids: (M,) int64 array of track IDs.
              - classes: (M,) int32 array of class labels.
        """
        _empty = (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )

        if dets.size == 0:
            return _empty

        # TrackResults is a numpy subclass: cols [x1,y1,x2,y2, id, conf, cls, det_ind]
        result = self._tracker.update(dets.astype(np.float32), frame)

        if result is None or len(result) == 0:
            return _empty

        boxes = np.asarray(result[:, :4], dtype=np.int32)
        ids = np.asarray(result[:, 4], dtype=np.int64)
        classes = np.asarray(result[:, 6], dtype=np.int32)
        return boxes, ids, classes
