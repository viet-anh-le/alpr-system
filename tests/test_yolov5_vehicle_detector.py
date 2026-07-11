from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.unit
def test_default_vehicle_detector_points_to_ctm_object_checkpoint() -> None:
    from api.core import config

    assert str(config.VEHICLE_MODEL_PATH).endswith(
        "references/Character-Time-series-Matching/Vietnamese/object.pt"
    )
    assert config.VEHICLE_CLASSES == [1, 6, 7, 8, 9, 10]


@pytest.mark.unit
def test_load_models_reuses_yolov5_object_for_vehicle(monkeypatch) -> None:
    import api.core.models as models

    fake_vehicle = SimpleNamespace(kind="vehicle", names=["1"])

    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    fake_object_path = Path("/tmp/missing-object.pt")
    monkeypatch.setattr(models, "VEHICLE_MODEL_PATH", fake_object_path)
    monkeypatch.setattr(models, "YOLOV5_CHAR_CKPT_PATH", Path("/tmp/missing-char.pt"))
    monkeypatch.setattr(models, "YOLOV5_OBJECT_CKPT_PATH", fake_object_path)
    monkeypatch.setattr(
        models,
        "load_yolov5_vehicle_detector",
        lambda path, device: fake_vehicle,
    )
    monkeypatch.setattr(
        models,
        "VehicleTracker",
        lambda **kwargs: SimpleNamespace(kind="tracker", kwargs=kwargs),
    )
    monkeypatch.setattr(
        models.PlateQualityRouter,
        "from_env",
        classmethod(lambda cls, device=None: SimpleNamespace(kind="router", device=device)),
    )
    monkeypatch.setattr(
        models,
        "load_small_lpr_line_ctc_model",
        lambda path, device: models.SmallLprLineCtcOcrModel(
            model=torch.nn.Identity(),
            chars=["<blank>", "A"],
        ),
    )

    bundle = models.load_models()

    assert bundle.vehicle is fake_vehicle
    assert bundle.yolov5_object is fake_vehicle


@pytest.mark.unit
def test_yolov5_vehicle_detector_predict_exposes_ultralytics_like_boxes(monkeypatch) -> None:
    import numpy as np

    import api.core.yolov5_vehicle as yolov5_vehicle

    class FakeModel(torch.nn.Module):
        names = ["person", "motorbike", "bicycle", "face", "square license plate", "rectangle license plate", "car"]

        def forward(self, images, augment=False):
            return (torch.zeros((images.shape[0], 1, 6)),)

    detection = torch.tensor([[10.0, 20.0, 50.0, 80.0, 0.91, 6.0]])
    monkeypatch.setattr(
        yolov5_vehicle,
        "non_max_suppression",
        lambda *args, **kwargs: [detection.clone()],
    )
    monkeypatch.setattr(
        yolov5_vehicle,
        "scale_coords",
        lambda _from_shape, coords, _to_shape: coords,
    )

    detector = yolov5_vehicle.YOLOv5VehicleDetector(
        model=FakeModel(),
        names=FakeModel.names,
        device=torch.device("cpu"),
    )
    frame = np.zeros((96, 128, 3), dtype=np.uint8)

    result = detector.predict(frame, classes=[6], verbose=False)[0]

    assert len(result.boxes) == 1
    assert result.boxes.xyxy.tolist() == [[10.0, 20.0, 50.0, 80.0]]
    assert result.boxes.conf.tolist() == [pytest.approx(0.91)]
    assert result.boxes.cls.tolist() == [6.0]
