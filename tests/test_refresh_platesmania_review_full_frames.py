from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "refresh_platesmania_review_full_frames.py"
SPEC = importlib.util.spec_from_file_location("refresh_platesmania_review_full_frames", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (20, 80, 120)
    assert cv2.imwrite(str(path), image)


@pytest.mark.unit
def test_refresh_low_confidence_review_image_with_full_frame(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "platesmania_vn"
    _write_image(dataset_dir / "downloads" / "full_frames" / "nomer1.jpg", width=200, height=100)
    _write_image(dataset_dir / "review" / "pending_review" / "nomer1.jpg", width=60, height=20)
    (dataset_dir / "review" / "pending_review" / "nomer1.txt").write_text(
        "low_detector_confidence:0.2991\n",
        encoding="utf-8",
    )

    stats = MODULE.refresh_review_full_frames(dataset_dir)

    refreshed = cv2.imread(str(dataset_dir / "review" / "pending_review" / "nomer1.jpg"))
    assert stats.refreshed == 1
    assert stats.skipped == 0
    assert stats.missing_full_frame == 0
    assert refreshed.shape[:2] == (100, 200)


@pytest.mark.unit
def test_refresh_skips_non_matching_reason_by_default(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "platesmania_vn"
    _write_image(dataset_dir / "downloads" / "full_frames" / "nomer1.jpg", width=200, height=100)
    _write_image(dataset_dir / "review" / "pending_review" / "nomer1.jpg", width=60, height=20)
    (dataset_dir / "review" / "pending_review" / "nomer1.txt").write_text(
        "no_plate_detection\n",
        encoding="utf-8",
    )

    stats = MODULE.refresh_review_full_frames(dataset_dir)

    refreshed = cv2.imread(str(dataset_dir / "review" / "pending_review" / "nomer1.jpg"))
    assert stats.refreshed == 0
    assert stats.skipped == 1
    assert refreshed.shape[:2] == (20, 60)
