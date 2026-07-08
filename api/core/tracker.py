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

# ── Vietnamese plate segment patterns ─────────────────────────────────────────
# Order matters: most-specific first so _parse_plate_segments returns the right fmt.
#   fmt 0: DD[L]{1,2}-NNNNN   e.g. 30A-12345, 50LD-12345
#   fmt 1: DD-[LS]-NNNNN      e.g. 29-X1-12345, 43-AA-01234
#   fmt 2: DDL-NNNN            e.g. 31H-9999
#   fmt 3: DD-LN-NNNN          e.g. 29-F4-8888
_SEG_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(?P<province>\d{2})(?P<serial>[A-Z]{1,2})-(?P<number>\d{5})$"),
    re.compile(r"^(?P<province>\d{2})-(?P<serial>[A-Z][\dA-Z])-(?P<number>\d{5})$"),
    re.compile(r"^(?P<province>\d{2})(?P<serial>[A-Z])-(?P<number>\d{4})$"),
    re.compile(r"^(?P<province>\d{2})-(?P<serial>[A-Z]\d)-(?P<number>\d{4})$"),
]


@dataclass(frozen=True)
class _PlateSegments:
    province: str
    serial: str
    number: str
    fmt: int  # index into _SEG_PATTERNS

    @property
    def serial_start(self) -> int:
        """Character index where the serial begins in the full plate string."""
        return 3 if self.fmt in (1, 3) else 2

    @property
    def number_start(self) -> int:
        """Character index where the number begins in the full plate string."""
        return self.serial_start + len(self.serial) + 1  # +1 for middle hyphen


def _parse_plate_segments(text: str) -> _PlateSegments | None:
    for fmt, pat in enumerate(_SEG_PATTERNS):
        m = pat.match(text)
        if m:
            return _PlateSegments(
                province=m.group("province"),
                serial=m.group("serial"),
                number=m.group("number"),
                fmt=fmt,
            )
    return None


# _ParsedFrame = (segments, char_probs)
_ParsedFrame = tuple[_PlateSegments, list[tuple[str, float]]]


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
        """Return up to k entries ranked by combined score descending."""
        if not self.crops:
            return [], [], []
        combined = [self._combined(q, c) for q, c in zip(self.quality_scores, self.ocr_confs)]
        is_prioritized = [
            1 if res.get("legibility") in ("perfect", "good") else 0
            for res in self.router_results
        ]
        triples = sorted(
            zip(is_prioritized, combined, self.crops, self.char_prob_lists),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )[:k]
        _, scores, crops, prob_lists = zip(*triples)
        return list(crops), list(scores), list(prob_lists)

    def top_k_entries(self, k: int = TOP_K_FRAMES) -> list[TrackBufferEntry]:
        """Return up to k full entries ranked by combined score descending."""
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

    # ── Public API ────────────────────────────────────────────────────────────

    def vehicle_track_id(self, tid: int) -> int:
        return int(tid)

    def plate_track_id(self, tid: int) -> int | None:
        return None

    def should_ocr(self, tid: int) -> bool:
        return not self._done.get(tid, False)

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

    # ── Probability-voting fusion over OCR frames ────────────────────────────

    @staticmethod
    def _prob_vote(
        prob_lists: list[list[tuple[str, float]]],
    ) -> list[tuple[str, float]]:
        """
        Average per-position character confidence across T OCR results.

        Sequences are aligned by position (no edit-distance alignment).  Shorter
        sequences skip positions they don't cover.  The character with the highest
        mean confidence at each position is selected.
        """
        if not prob_lists:
            return []
        max_len = max(len(p) for p in prob_lists)
        result: list[tuple[str, float]] = []
        for pos in range(max_len):
            votes: dict[str, list[float]] = {}
            for probs in prob_lists:
                if pos < len(probs):
                    char, conf = probs[pos]
                    votes.setdefault(char, []).append(conf)
            if not votes:
                continue
            # Bầu chọn theo số lượng vote trước (frequency).
            # Nếu số lượng vote bằng nhau (ví dụ 50/50), lấy kí tự có độ tự tin cao nhất làm tie-breaker.
            best_char = max(votes, key=lambda c: (len(votes[c]), max(votes[c])))
            # Ensemble confidence: total confidence divided by total frames
            best_conf = sum(votes[best_char]) / len(prob_lists)
            result.append((best_char, best_conf))
        return result

    @staticmethod
    def _segment_vote(
        prob_lists: list[list[tuple[str, float]]],
    ) -> list[tuple[str, float]] | None:
        """
        Apply _prob_vote independently on each plate segment.

        Each OCR result is parsed into (province, serial, number) segments.
        Candidates that don't match a valid Vietnamese plate format are dropped.
        Remaining candidates are grouped by (serial_length, number_length) so
        that _prob_vote never receives misaligned sequences.  The dominant group
        is chosen and _prob_vote is run separately on province, serial, and
        number character probabilities.

        Returns None if no candidate parses as a valid plate format.
        """
        from collections import Counter

        parsed: list[_ParsedFrame] = []
        for probs in prob_lists:
            text = "".join(c for c, _ in probs)
            seg = _parse_plate_segments(text)
            if seg is not None:
                parsed.append((seg, probs))

        if not parsed:
            return None

        # Pick the dominant (serial_len, number_len) group to prevent
        # _prob_vote from receiving sequences of different lengths.
        target_serial_len, target_number_len = Counter(
            (len(seg.serial), len(seg.number)) for seg, _ in parsed
        ).most_common(1)[0][0]

        pool = [
            (seg, probs)
            for seg, probs in parsed
            if len(seg.serial) == target_serial_len and len(seg.number) == target_number_len
        ]

        prov_chars = WebTrackletManager._prob_vote([probs[0:2] for _, probs in pool])
        serial_chars = WebTrackletManager._prob_vote(
            [probs[seg.serial_start : seg.serial_start + target_serial_len] for seg, probs in pool]
        )
        number_chars = WebTrackletManager._prob_vote(
            [probs[seg.number_start : seg.number_start + target_number_len] for seg, probs in pool]
        )

        use_leading_hyphen = pool[0][0].fmt in (1, 3)

        result: list[tuple[str, float]] = []
        result.extend(prov_chars)
        if use_leading_hyphen:
            result.append(("-", 0.9))
        result.extend(serial_chars)
        result.append(("-", 0.9))
        result.extend(number_chars)
        return result

    # ── Levenshtein confidence fusion (legacy / single-frame path) ───────────

    def _fuse(
        self,
        seq1: list[tuple[str, float]],
        seq2: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        n, m = len(seq1), len(seq2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq1[i - 1][0] == seq2[j - 1][0] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )

        i, j = n, m
        a1: list = []
        a2: list = []
        while i > 0 or j > 0:
            if (
                i > 0
                and j > 0
                and dp[i][j] == dp[i - 1][j - 1] + (0 if seq1[i - 1][0] == seq2[j - 1][0] else 1)
            ):
                a1.append(seq1[i - 1])
                a2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                a1.append(seq1[i - 1])
                a2.append(None)
                i -= 1
            else:
                a1.append(None)
                a2.append(seq2[j - 1])
                j -= 1

        a1.reverse()
        a2.reverse()

        merged: list[tuple[str, float]] = []
        for a, b in zip(a1, a2):
            if a is not None and b is not None:
                merged.append(a if a[1] >= b[1] else b)
            elif a is not None:
                merged.append(a)
            elif b is not None and b[1] >= CONF_THRESHOLD:
                merged.append(b)
        return merged
