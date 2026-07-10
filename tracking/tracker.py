"""
core/tracker.py — WebTrackletManager.

Manages per-vehicle OCR state: Levenshtein-based confidence fusion (Layer 0)
and temporal consistency gating (Layer 3).  No disk I/O.

Stores two evidence images per vehicle:
  _plate_img   — best plate crop (highest avg OCR confidence)
  _vehicle_img — best full-vehicle crop (same confidence criterion)
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .config import CONF_THRESHOLD, MIN_FRAME_VOTES

_VEHICLE_IMG_MAX_W = 320


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

    # ── Public API ────────────────────────────────────────────────────────────

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

    def plate_b64(self, tid: int) -> str | None:
        return self._encode(self._plate_img.get(tid), max_w=None, quality=90)

    def vehicle_b64(self, tid: int) -> str | None:
        return self._encode(
            self._vehicle_img.get(tid),
            max_w=_VEHICLE_IMG_MAX_W,
            quality=85,
        )

    def display_text(self, tid: int) -> str:
        return "".join(c for c, p in self._best.get(tid, []))

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

    # ── Levenshtein confidence fusion (identical to inference_pipeline.py) ────

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
