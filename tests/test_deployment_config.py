from __future__ import annotations

import importlib
from pathlib import Path


def test_runpod_model_paths_can_be_overridden_with_absolute_env_paths(monkeypatch, tmp_path):
    from api.core import config

    plate_path = tmp_path / "plate.pt"
    reid_path = tmp_path / "vehicle_reid.onnx"

    monkeypatch.setenv("PLATE_MODEL_PATH", str(plate_path))
    monkeypatch.setenv("REID_MODEL_PATH", str(reid_path))

    reloaded = importlib.reload(config)
    try:
        assert reloaded.PLATE_MODEL_PATH == plate_path
        assert reloaded.REID_MODEL_PATH == reid_path
    finally:
        monkeypatch.delenv("PLATE_MODEL_PATH", raising=False)
        monkeypatch.delenv("REID_MODEL_PATH", raising=False)
        importlib.reload(config)


def test_relative_runpod_model_paths_still_resolve_from_project_root(monkeypatch):
    from api.core import config

    monkeypatch.setenv("PLATE_MODEL_PATH", "models/plate.pt")
    monkeypatch.setenv("REID_MODEL_PATH", "models/reid.onnx")

    reloaded = importlib.reload(config)
    try:
        assert reloaded.PLATE_MODEL_PATH == Path(reloaded.ROOT / "models/plate.pt")
        assert reloaded.REID_MODEL_PATH == Path(reloaded.ROOT / "models/reid.onnx")
    finally:
        monkeypatch.delenv("PLATE_MODEL_PATH", raising=False)
        monkeypatch.delenv("REID_MODEL_PATH", raising=False)
        importlib.reload(config)


def test_vehicle_detector_backend_can_force_yolov5_on_runpod(monkeypatch):
    from api.core import config

    monkeypatch.setenv("VEHICLE_DETECTOR_BACKEND", "yolov5")

    reloaded = importlib.reload(config)
    try:
        assert reloaded.VEHICLE_DETECTOR_BACKEND == "yolov5"
    finally:
        monkeypatch.delenv("VEHICLE_DETECTOR_BACKEND", raising=False)
        importlib.reload(config)


def test_reid_device_can_force_cpu_tracking_on_runpod(monkeypatch):
    from api.core import config

    monkeypatch.setenv("REID_DEVICE", "cpu")

    reloaded = importlib.reload(config)
    try:
        assert reloaded.REID_DEVICE == "cpu"
    finally:
        monkeypatch.delenv("REID_DEVICE", raising=False)
        importlib.reload(config)


def test_runpod_docker_context_keeps_yolov5_source_available():
    dockerignore = Path(".dockerignore").read_text()

    assert "!references/Character-Time-series-Matching/yolov5/**" in dockerignore
    assert "!references/Character-Time-series-Matching/yolov5/models/**" in dockerignore
    assert "!references/Character-Time-series-Matching/yolov5/utils/**" in dockerignore


def test_runpod_keeps_setuptools_pkg_resources_for_yolov5():
    dockerfile = Path("Dockerfile.runpod").read_text()
    requirements = Path("requirements-runpod.txt").read_text()

    assert "setuptools<81" in dockerfile
    assert "setuptools<81" in requirements


def test_runpod_cpu_reid_uses_cpu_onnxruntime_package():
    requirements = Path("requirements-runpod.txt").read_text()

    assert "onnxruntime==1.20.0" in requirements
    assert "onnxruntime-gpu" not in requirements


def test_runpod_image_exposes_rtsp_ingest_port():
    dockerfile = Path("Dockerfile.runpod").read_text()

    assert "EXPOSE 8000 8889 8554" in dockerfile


def test_mediamtx_allows_local_demo_rtsp_publisher_path():
    config = Path("configs/mediamtx.yml").read_text()

    assert "alpr_demo:" in config
    assert "source: publisher" in config
