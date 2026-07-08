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
class PlateQualityResult:
    legibility: Legibility
    quality_bin: QualityBin
    router_conf: float
    route: RouteName
    quality_numeric: float

    def as_event_fields(self) -> dict:
        return {
            "route": self.route,
            "legibility": self.legibility,
            "quality_bin": self.quality_bin,
            "router_conf": round(float(self.router_conf), 4),
            "quality_score": round(float(self.quality_numeric), 4),
        }


ClassifierFn = Callable[[np.ndarray], Mapping[str | int, float]]


class PlateQualityRouter:
    """Route rectified plate crops to direct OCR, fusion, or unreadable wait."""

    def __init__(
        self,
        *,
        classifier: ClassifierFn | None = None,
        model_path: str | os.PathLike[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.classifier = classifier
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
        scores = self._predict_scores(crop_bgr)

        if scores:
            legibility, conf = _best_legibility(scores)
        else:
            legibility, conf = "illegible", 0.0

        quality_bin: QualityBin = "suitable" if legibility in {"perfect", "good"} else "unsuitable"
        if legibility == "illegible":
            route: RouteName = "unreadable_wait"
        elif quality_bin == "suitable":
            route = "direct"
        else:
            route = "tracklet_fusion"

        return PlateQualityResult(
            legibility=legibility,
            quality_bin=quality_bin,
            router_conf=float(conf),
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


