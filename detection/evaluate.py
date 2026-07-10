from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate(weights: Path, data_yaml: Path) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    return model.val(data=str(data_yaml))
