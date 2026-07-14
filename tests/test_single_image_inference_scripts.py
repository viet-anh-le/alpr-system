from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "single_image_inference",
        ROOT / "scripts/single_image_inference.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prediction_kwargs_omits_auto_device_and_keeps_thresholds() -> None:
    module = _load_script_module()

    kwargs = module._prediction_kwargs(device="auto", imgsz=1280, conf=0.25)

    assert kwargs == {"verbose": False, "imgsz": 1280, "conf": 0.25}


def test_prediction_kwargs_accepts_explicit_device_and_half() -> None:
    module = _load_script_module()

    kwargs = module._prediction_kwargs(device="cpu", imgsz=64, half=False)

    assert kwargs == {"verbose": False, "imgsz": 64, "device": "cpu", "half": False}


def test_normalize_yolo_names_accepts_lists_and_dicts() -> None:
    module = _load_script_module()

    assert module._normalize_names(["poor", "good"]) == {0: "poor", 1: "good"}
    assert module._normalize_names({"0": "plate", 2: "other"}) == {
        0: "plate",
        2: "other",
    }


def test_plate_candidate_json_excludes_image_crop() -> None:
    module = _load_script_module()
    candidate = module.PlateCandidate(
        index=1,
        kind="obb",
        class_id=0,
        class_name="plate",
        confidence=0.91234567,
        box=[1, 2, 30, 12],
        points=[[1.1111, 2.2222], [30, 2], [30, 12], [1, 12]],
        crop=object(),
        crop_path=Path("data/outputs/crops/plate.jpg"),
    )

    payload = candidate.to_json()

    assert payload["confidence"] == 0.912346
    assert payload["points"][0] == [1.11, 2.22]
    assert payload["crop_path"] == "data/outputs/crops/plate.jpg"
    assert "crop" not in payload
