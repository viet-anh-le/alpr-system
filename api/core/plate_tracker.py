"""
core/plate_tracker.py — ByteTrack wrapper for license-plate tracking.

Tracks plate detections (in global frame coordinates) across frames using
Ultralytics' built-in BYTETracker.  Provides stable plate track IDs so the
downstream TrajectoryAssociator can match plate *trajectories* to vehicle
*trajectories* instead of doing single-frame spatial matching.

Why ByteTrack (not BoT-SORT)?
  Plates are small, texturally uniform objects — ReID features add little
  value.  ByteTrack's two-stage IoU association is lightweight and handles
  low-confidence detections gracefully, which is ideal for plates that
  frequently dip below the detection threshold.
"""

from __future__ import annotations

import logging
from argparse import Namespace

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker

from .config import (
    PLATE_TRACK_BUFFER,
    PLATE_TRACK_HIGH_THRESH,
    PLATE_TRACK_LOW_THRESH,
    PLATE_TRACK_MATCH_THRESH,
    PLATE_TRACK_NEW_THRESH,
)

logger = logging.getLogger(__name__)


class _PlateDetections:
    """Minimal Results-like object expected by BYTETracker.update().

    BYTETracker.init_track() reads .xywh, .conf, .cls and supports
    indexing via __getitem__ (boolean mask).
    """

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self._xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        """Convert xyxy → xywh (center-x, center-y, width, height)."""
        x1, y1, x2, y2 = (
            self._xyxy[:, 0],
            self._xyxy[:, 1],
            self._xyxy[:, 2],
            self._xyxy[:, 3],
        )
        return np.stack(
            [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1
        )

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    def __len__(self) -> int:
        return len(self._xyxy)

    def __getitem__(self, idx: np.ndarray | slice) -> "_PlateDetections":
        return _PlateDetections(
            xyxy=self._xyxy[idx],
            conf=self.conf[idx],
            cls=self.cls[idx],
        )


class PlateTracker:
    """Thin adapter around Ultralytics BYTETracker for plate tracking.

    Usage::

        tracker = PlateTracker()

        # Each frame:
        tracked = tracker.update(raw_candidates)
        # tracked: list of dicts with persistent "id" field
    """

    def __init__(
        self,
        *,
        track_buffer: int = PLATE_TRACK_BUFFER,
        track_high_thresh: float = PLATE_TRACK_HIGH_THRESH,
        track_low_thresh: float = PLATE_TRACK_LOW_THRESH,
        new_track_thresh: float = PLATE_TRACK_NEW_THRESH,
        match_thresh: float = PLATE_TRACK_MATCH_THRESH,
    ) -> None:
        args = Namespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        self._tracker = BYTETracker(args)

    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        self._tracker.frame_id = 0
        self._tracker.tracked_stracks = []
        self._tracker.lost_stracks = []
        self._tracker.removed_stracks = []
        self._tracker.reset_id()

    def update(self, candidates: list[dict]) -> list[dict]:
        """Feed plate detections and return tracked plates with persistent IDs.

        Args:
            candidates: list of dicts, each with at least:
                - "box": [x1, y1, x2, y2] in global frame coords
                - "conf": float detection confidence
                Optionally: "crop", "pts", "source_vehicle_id", "det_conf"

        Returns:
            list of dicts, same fields as input plus:
                - "id": int — persistent plate track ID from ByteTrack
        """
        if not candidates:
            # Still need to update tracker with empty detections so it can
            # age out lost tracks properly.
            empty = _PlateDetections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                conf=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.float32),
            )
            self._tracker.update(empty)
            return []

        # Build detection arrays
        xyxy = np.array([c["box"] for c in candidates], dtype=np.float32)
        conf = np.array(
            [c.get("conf", c.get("det_conf", 0.5)) for c in candidates],
            dtype=np.float32,
        )
        cls = np.zeros(len(candidates), dtype=np.float32)  # single class

        dets = _PlateDetections(xyxy=xyxy, conf=conf, cls=cls)
        # BYTETracker.update returns shape (M, 8):
        #   [x1, y1, x2, y2, track_id, score, cls, det_idx]
        raw = self._tracker.update(dets)

        if raw is None or len(raw) == 0:
            return []

        tracked: list[dict] = []
        for row in raw:
            track_id = int(row[4])
            det_idx = int(row[7])
            # det_idx maps back to the original candidate
            if 0 <= det_idx < len(candidates):
                entry = {**candidates[det_idx], "id": track_id}
            else:
                # Tracker re-activated a lost track without a matching
                # detection index — build from tracked box.
                entry = {
                    "box": [int(row[0]), int(row[1]), int(row[2]), int(row[3])],
                    "conf": float(row[5]),
                    "id": track_id,
                }
            tracked.append(entry)

        return tracked
