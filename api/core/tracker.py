"""
core/tracker.py — WebTrackletManager and TrackBuffer.

TrackBuffer accumulates plate crops per track and evicts the lowest-quality
frame when the buffer is full (max MAX_BUFFER entries).

WebTrackletManager now supports two modes:
  - Legacy per-frame update via update() + _fuse() (kept for backward compat)
  - Track-lifecycle mode: buffer_crop() → [track lost] → track-level voting,
    then update() once with all_confident=True.

Evidence images (best plate crop + best vehicle crop) are stored per track.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import (
    CLUSTER_SIMILARITY_THRESHOLD,
    CONF_THRESHOLD,
    LOST_THRESHOLD,
    MAX_BUFFER,
    MAX_CLUSTERS,
    MIN_FRAME_VOTES,
    MIN_FRAMES_FOR_OCR,
    TOP_K_FRAMES,
)
from .plate_format import chars_to_display_text

logger = logging.getLogger(__name__)

_VEHICLE_IMG_MAX_W = 320


# ── Track buffer ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackBufferEntry:
    crop: np.ndarray
    quality_score: float
    ocr_conf: float
    char_probs: list[tuple[str, float]]
    frame_idx: int
    candidate_method: str
    route: str
    router_result: dict

    @property
    def combined_score(self) -> float:
        return TrackBuffer._combined(self.quality_score, self.ocr_conf)


@dataclass
class TrackBuffer:
    """
    Per-track ring buffer of plate crops.

    Eviction policy: when full, the crop with the lowest
    *combined* score (visual quality × OCR confidence) is dropped.
    This prevents sharp-but-garbled crops from displacing softer-but-
    correctly-read ones — the root cause of the 30G-51827 mis-read.
    """

    max_size: int = field(default=MAX_BUFFER)
    crops: list[np.ndarray] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)
    ocr_confs: list[float] = field(default_factory=list)
    char_prob_lists: list[list[tuple[str, float]]] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    candidate_methods: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    router_results: list[dict] = field(default_factory=list)

    @staticmethod
    def _combined(quality: float, ocr_conf: float) -> float:
        return quality * max(ocr_conf, 0.10)

    def add(
        self,
        crop: np.ndarray,
        quality_score: float,
        ocr_conf: float,
        char_probs: list[tuple[str, float]],
        frame_idx: int,
        candidate_method: str = "original",
        route: str = "tracklet_fusion",
        router_result: dict | None = None,
    ) -> None:
        self.crops.append(crop)
        self.quality_scores.append(quality_score)
        self.ocr_confs.append(ocr_conf)
        self.char_prob_lists.append(char_probs)
        self.frame_indices.append(frame_idx)
        self.candidate_methods.append(candidate_method)
        self.routes.append(route)
        self.router_results.append(router_result or {})
        if len(self.crops) > self.max_size:
            worst = min(
                range(len(self.crops)),
                key=lambda i: (
                    1 if self.router_results[i].get("legibility") in ("perfect", "good") else 0,
                    self._combined(self.quality_scores[i], self.ocr_confs[i]),
                ),
            )
            del self.crops[worst]
            del self.quality_scores[worst]
            del self.ocr_confs[worst]
            del self.char_prob_lists[worst]
            del self.frame_indices[worst]
            del self.candidate_methods[worst]
            del self.routes[worst]
            del self.router_results[worst]

    def top_k(
        self, k: int = TOP_K_FRAMES
    ) -> tuple[list[np.ndarray], list[float], list[list[tuple[str, float]]]]:
        """Return up to k entries ranked by combined score descending.
        k=None (default) uses all buffered entries (bounded by max_size)."""
        if k is None:
            k = self.max_size
        if not self.crops:
            return [], [], []
        combined = [self._combined(q, c) for q, c in zip(self.quality_scores, self.ocr_confs)]
        is_prioritized = [
            1 if res.get("legibility") in ("perfect", "good") else 0 for res in self.router_results
        ]
        triples = sorted(
            zip(is_prioritized, combined, self.crops, self.char_prob_lists),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )[:k]
        _, scores, crops, prob_lists = zip(*triples)
        return list(crops), list(scores), list(prob_lists)

    def top_k_entries(self, k: int = TOP_K_FRAMES) -> list[TrackBufferEntry]:
        """Return up to k full entries ranked by combined score descending.
        k=None (default) uses all buffered entries (bounded by max_size)."""
        if k is None:
            k = self.max_size
        entries = [
            TrackBufferEntry(
                crop=crop,
                quality_score=q,
                ocr_conf=ocr_conf,
                char_probs=char_probs,
                frame_idx=frame_idx,
                candidate_method=candidate_method,
                route=route,
                router_result=router_result,
            )
            for crop, q, ocr_conf, char_probs, frame_idx, candidate_method, route, router_result in zip(
                self.crops,
                self.quality_scores,
                self.ocr_confs,
                self.char_prob_lists,
                self.frame_indices,
                self.candidate_methods,
                self.routes,
                self.router_results,
            )
        ]
        return sorted(
            entries,
            key=lambda entry: (
                1 if entry.router_result.get("legibility") in ("perfect", "good") else 0,
                entry.combined_score,
            ),
            reverse=True,
        )[:k]


# ── Tracklet manager ──────────────────────────────────────────────────────────


class WebTrackletManager:
    def __init__(self) -> None:
        self._done: dict[int, bool] = {}
        self._best: dict[int, list[tuple[str, float]]] = {}
        self._cls: dict[int, str] = {}
        self._prev_plate: dict[int, str] = {}
        self._ocr_count: dict[int, int] = {}

        # Evidence images
        self._plate_img: dict[int, np.ndarray] = {}
        self._plate_img_conf: dict[int, float] = {}
        self._vehicle_img: dict[int, np.ndarray] = {}
        self._vehicle_img_conf: dict[int, float] = {}

        # Track lifecycle voting path
        self._buffers: dict[int, TrackBuffer] = {}
        self._lost_count: dict[int, int] = {}

        # Multi-cluster results (populated by finalise_track_ocr when a track
        # contains OCR evidence for 2+ distinct licence plates).
        # Each entry: list of {"plate", "chars", "confidence", "plate_b64", "frame_count"}
        self._cluster_results: dict[int, list[dict]] = {}

        # Lifecycle: tids whose heavy state was released after finalize. Kept as a
        # lightweight tombstone so should_ocr() stays False and the same tid is
        # never OCR'd/finalised again.
        self._released_tids: set[int] = set()
        self._released_recognized_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def vehicle_track_id(self, tid: int) -> int:
        return int(tid)

    def plate_track_id(self, tid: int) -> int | None:
        return None

    def should_ocr(self, tid: int) -> bool:
        if tid in self._released_tids:
            return False
        return not self._done.get(tid, False)

    def release_track(self, tid: int, recognized: bool) -> None:
        """Drop all heavy per-track state after a track has been finalised, its
        event emitted, and its DB snapshot taken. Leaves only a lightweight
        tombstone so the same tid is never re-OCR'd/finalised."""
        if tid in self._released_tids:
            return
        for state in (
            self._done,
            self._best,
            self._cls,
            self._prev_plate,
            self._ocr_count,
            self._plate_img,
            self._plate_img_conf,
            self._vehicle_img,
            self._vehicle_img_conf,
            self._buffers,
            self._lost_count,
            self._cluster_results,
        ):
            state.pop(tid, None)
        self._released_tids.add(tid)
        if recognized:
            self._released_recognized_count += 1

    def recognized_vehicle_count(self) -> int:
        """Total vehicles with a valid recognised plate this session — survives
        release_track() cleanup (does not depend on _best)."""
        live = sum(1 for tid in self._best if tid not in self._released_tids)
        return self._released_recognized_count + live

    def update(
        self,
        tid: int,
        char_probs: list[tuple[str, float]],
        all_confident: bool,
    ) -> None:
        if not char_probs:
            return

        self._ocr_count[tid] = self._ocr_count.get(tid, 0) + 1

        self._best[tid] = (
            char_probs if tid not in self._best else self._fuse(self._best[tid], char_probs)
        )

        frames = self._ocr_count[tid]
        fully_confident = all(p >= CONF_THRESHOLD for _, p in self._best[tid])
        if (fully_confident or all_confident) and frames >= MIN_FRAME_VOTES:
            self._done[tid] = True

    def update_plate_img(
        self,
        tid: int,
        crop: np.ndarray,
        char_probs: list[tuple[str, float]],
    ) -> None:
        if not char_probs:
            return
        conf = sum(p for _, p in char_probs) / len(char_probs)
        if conf > self._plate_img_conf.get(tid, -1.0):
            self._plate_img_conf[tid] = conf
            self._plate_img[tid] = crop.copy()

    def set_plate_img(
        self,
        tid: int,
        crop: np.ndarray,
        confidence: float,
    ) -> None:
        if crop.size == 0:
            return
        self._plate_img_conf[tid] = confidence
        self._plate_img[tid] = crop.copy()

    def update_vehicle_img(
        self,
        tid: int,
        crop: np.ndarray,
        conf: float,
    ) -> None:
        if crop.size == 0:
            return
        if conf > self._vehicle_img_conf.get(tid, -1.0):
            self._vehicle_img_conf[tid] = conf
            self._vehicle_img[tid] = crop.copy()

    # ── Track lifecycle voting path ──────────────────────────────────────────

    def buffer_crop(
        self,
        tid: int,
        crop: np.ndarray,
        quality_score: float,
        ocr_conf: float,
        char_probs: list[tuple[str, float]],
        frame_idx: int,
        candidate_method: str = "original",
        route: str = "tracklet_fusion",
        router_result: dict | None = None,
    ) -> None:
        if tid not in self._buffers:
            self._buffers[tid] = TrackBuffer()
        self._buffers[tid].add(
            crop,
            quality_score,
            ocr_conf,
            char_probs,
            frame_idx,
            candidate_method=candidate_method,
            route=route,
            router_result=router_result,
        )

    def mark_lost(self, tid: int) -> bool:
        """
        Increment the lost-frame counter for tid.
        Returns True once the track has been absent for LOST_THRESHOLD strides,
        indicating it should be finalised.
        """
        self._lost_count[tid] = self._lost_count.get(tid, 0) + 1
        return self._lost_count[tid] >= LOST_THRESHOLD

    def reset_lost(self, tid: int) -> None:
        """Call when a previously-lost track reappears."""
        self._lost_count.pop(tid, None)

    def ready_for_track_ocr(self, tid: int) -> bool:
        buf = self._buffers.get(tid)
        if buf is None:
            return False
        has_direct_ocr = any(
            route == "direct" and bool(char_probs)
            for route, char_probs in zip(buf.routes, buf.char_prob_lists)
        )
        if has_direct_ocr:
            return True
        return len(set(buf.frame_indices)) >= MIN_FRAMES_FOR_OCR

    # ── Accessors ─────────────────────────────────────────────────────────────

    def plate_b64(self, tid: int) -> str | None:
        return self._encode(self._plate_img.get(tid), max_w=None, quality=90)

    def vehicle_b64(self, tid: int) -> str | None:
        return self._encode(
            self._vehicle_img.get(tid),
            max_w=_VEHICLE_IMG_MAX_W,
            quality=85,
        )

    def track_buffer_json(self, tid: int) -> list[dict]:
        buf = self._buffers.get(tid)
        if buf is None:
            return []

        frames: list[dict] = []
        for entry in buf.top_k_entries(k=buf.max_size):
            frames.append(
                {
                    "frame_index": int(entry.frame_idx),
                    "quality_score": round(float(entry.quality_score), 4),
                    "ocr_confidence": round(float(entry.ocr_conf), 4),
                    "candidate_method": entry.candidate_method,
                    "route": entry.route,
                    "image_b64": self._encode(entry.crop, max_w=None, quality=85),
                    **entry.router_result,
                }
            )
        return frames

    def cluster_results(self, tid: int) -> list[dict]:
        """Return multi-cluster OCR results for this track (if any)."""
        return self._cluster_results.get(tid, [])

    def set_cluster_results(self, tid: int, results: list[dict]) -> None:
        self._cluster_results[tid] = results

    def display_text(self, tid: int) -> str:
        return chars_to_display_text(self._best.get(tid, []))

    def chars_json(self, tid: int) -> list[list]:
        return [[c, round(p, 3)] for c, p in self._best.get(tid, [])]

    def ocr_frames(self, tid: int) -> int:
        return self._ocr_count.get(tid, 0)

    def plate_changed(self, tid: int) -> bool:
        cur = self.display_text(tid)
        if cur != self._prev_plate.get(tid):
            self._prev_plate[tid] = cur
            return True
        return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _encode(
        img: np.ndarray | None,
        max_w: int | None,
        quality: int,
    ) -> str | None:
        if img is None:
            return None
        if max_w is not None:
            h, w = img.shape[:2]
            if w > max_w:
                img = cv2.resize(
                    img,
                    (max_w, int(h * max_w / w)),
                    interpolation=cv2.INTER_AREA,
                )
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode()
