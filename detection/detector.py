from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """YOLO-based license plate detector."""

    def __init__(self, weights: Path, conf_threshold: float = 0.5) -> None:
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.model = None

    def load(self) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(self.weights))
        logger.info(f"Detector loaded from {self.weights}")

    def detect(self, image: torch.Tensor) -> list[dict]:
        raise NotImplementedError
