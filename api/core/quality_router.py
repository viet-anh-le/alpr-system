"""Plate Quality Router inspired by the LPLCv2 legibility taxonomy."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .config import BLUR_THRESHOLD, MIN_PLATE_H, MIN_PLATE_W
from .quality_scorer import quality_score

Legibility = Literal["perfect", "good", "poor", "illegible"]
QualityBin = Literal["suitable", "unsuitable"]
RouteName = Literal["direct", "tracklet_fusion", "unreadable_wait"]

_LEGIBILITY_NAMES: tuple[Legibility, ...] = ("illegible", "poor", "good", "perfect")


@dataclass(frozen=True)
class DegradationTags:
    low_res: bool = False
    motion_blur: bool = False
    low_light: bool = False
    low_contrast: bool = False
    rain_or_haze: bool = False
    faulty_color: bool = False
    occluded: bool = False

    def any_active(self) -> bool:
        return any(self.as_dict().values())

    def as_dict(self) -> dict[str, bool]:
        return {
            "low_res": self.low_res,
            "motion_blur": self.motion_blur,
            "low_light": self.low_light,
            "low_contrast": self.low_contrast,
            "rain_or_haze": self.rain_or_haze,
            "faulty_color": self.faulty_color,
            "occluded": self.occluded,
        }

    @classmethod
    def merge(cls, tags: list["DegradationTags"]) -> "DegradationTags":
        if not tags:
            return cls()
        merged: dict[str, bool] = {}
        for key in cls.__dataclass_fields__:
            merged[key] = any(getattr(tag, key) for tag in tags)
        return cls(**merged)


@dataclass(frozen=True)
class PlateQualityResult:
    legibility: Legibility
    quality_bin: QualityBin
    router_conf: float
    tags: DegradationTags
    route: RouteName
    quality_numeric: float

    def as_event_fields(self) -> dict:
        return {
            "route": self.route,
            "legibility": self.legibility,
            "quality_bin": self.quality_bin,
            "degradation_tags": self.tags.as_dict(),
            "router_conf": round(float(self.router_conf), 4),
            "quality_score": round(float(self.quality_numeric), 4),
        }


ClassifierFn = Callable[[np.ndarray], Mapping[str | int, float]]
DiagnoserFn = Callable[[np.ndarray], DegradationTags]


class PlateQualityRouter:
    """Route rectified plate crops to direct OCR, fusion, or unreadable wait.

    A trained classifier can be supplied as a callable returning class scores
    for the LPLCv2 classes. When no classifier is available, deterministic
    visual diagnostics provide a conservative fallback for local inference and
    tests.
    """

    def __init__(
        self,
        *,
        classifier: ClassifierFn | None = None,
        diagnoser: DiagnoserFn | None = None,
        model_path: str | os.PathLike[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.classifier = classifier
        self.diagnoser = diagnoser
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self._model = None
        if self.classifier is None and self.model_path:
            self._model = self._load_ultralytics_classifier(self.model_path)

    @classmethod
    def from_env(cls, *, device: object | None = None) -> "PlateQualityRouter":
        model_path = os.environ.get("PLATE_QUALITY_ROUTER_MODEL", "").strip()
        return cls(model_path=model_path or None, device=str(device) if device is not None else None)

    def route(self, crop_bgr: np.ndarray) -> PlateQualityResult:
        q = quality_score(crop_bgr) if crop_bgr.size else 0.0
        tags = self.diagnoser(crop_bgr) if self.diagnoser else diagnose_degradation(crop_bgr)
        scores = self._predict_scores(crop_bgr)

        if scores:
            legibility, conf = _best_legibility(scores)
        else:
            legibility, conf = _heuristic_legibility(q, tags)

        quality_bin: QualityBin = "suitable" if legibility in {"perfect", "good"} else "unsuitable"
        if legibility == "illegible" or tags.occluded:
            route: RouteName = "unreadable_wait"
        elif quality_bin == "suitable":
            route = "direct"
        else:
            route = "tracklet_fusion"

        return PlateQualityResult(
            legibility=legibility,
            quality_bin=quality_bin,
            router_conf=float(conf),
            tags=tags,
            route=route,
            quality_numeric=float(q),
        )

    def _predict_scores(self, crop_bgr: np.ndarray) -> dict[str, float]:
        if self.classifier is not None:
            return _normalize_scores(self.classifier(crop_bgr))
        if self._model is None:
            return {}
        return self._predict_ultralytics(crop_bgr)

    def _load_ultralytics_classifier(self, model_path: Path):
        try:
            from ultralytics import YOLO

            return YOLO(str(model_path))
        except Exception:
            return None

    def _predict_ultralytics(self, crop_bgr: np.ndarray) -> dict[str, float]:
        if self._model is None:
            return {}
        try:
            result = self._model.predict(crop_bgr, verbose=False, device=self.device)[0]
            probs = result.probs.data.detach().cpu().numpy()
            names = getattr(result, "names", None) or getattr(self._model, "names", {})
            return _normalize_scores({names.get(i, i): float(score) for i, score in enumerate(probs)})
        except Exception:
            return {}


def diagnose_degradation(crop_bgr: np.ndarray) -> DegradationTags:
    if crop_bgr.size == 0:
        return DegradationTags(occluded=True)

    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    channel_means = crop_bgr.reshape(-1, 3).mean(axis=0)
    color_spread = float(channel_means.max() - channel_means.min())
    saturation = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

    low_res = h < MIN_PLATE_H * 2 or w < MIN_PLATE_W * 2 or (h * w) < (MIN_PLATE_H * MIN_PLATE_W * 2)
    low_light = brightness < 70.0
    low_contrast = contrast < 24.0
    motion_blur = lap_var < BLUR_THRESHOLD
    rain_or_haze = brightness > 125.0 and contrast < 34.0 and float(saturation.mean()) < 70.0
    faulty_color = color_spread > 55.0
    occluded = brightness < 18.0 and contrast < 8.0

    return DegradationTags(
        low_res=low_res,
        motion_blur=motion_blur,
        low_light=low_light,
        low_contrast=low_contrast,
        rain_or_haze=rain_or_haze,
        faulty_color=faulty_color,
        occluded=occluded,
    )


def _normalize_scores(scores: Mapping[str | int, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in scores.items():
        if isinstance(key, int):
            label = _LEGIBILITY_NAMES[key] if 0 <= key < len(_LEGIBILITY_NAMES) else str(key)
        else:
            label = str(key).strip().lower()
        if label in _LEGIBILITY_NAMES:
            normalized[label] = float(value)
    return normalized


def _best_legibility(scores: Mapping[str | int, float]) -> tuple[Legibility, float]:
    normalized = _normalize_scores(scores)
    if not normalized:
        return "poor", 0.0
    label = max(normalized, key=normalized.get)
    return label, float(normalized[label])  # type: ignore[return-value]


def _heuristic_legibility(q: float, tags: DegradationTags) -> tuple[Legibility, float]:
    if tags.occluded or q < 0.08:
        return "illegible", max(0.50, 1.0 - q)
    if tags.low_res or tags.motion_blur or tags.low_light or tags.low_contrast:
        return "poor", max(0.50, 1.0 - q)
    if q >= 0.70:
        return "perfect", min(0.99, q)
    if q >= 0.35:
        return "good", max(0.55, q)
    return "poor", max(0.50, 1.0 - q)
