from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from bson import ObjectId
from fastapi import HTTPException

from api.database.models import User


@pytest.mark.unit
def test_recording_frame_source_writes_browser_playable_mp4(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for preprocessed video artifacts")

    from api.core.frame_source import LiveBufferFrameSource
    from api.core.preprocessed_video import RecordingFrameSource

    frames = [
        (idx, np.full((48, 64, 3), idx * 40, dtype=np.uint8), idx / 12.0)
        for idx in range(1, 4)
    ]
    source = LiveBufferFrameSource(frames, fps=12.0, frame_size=(64, 48))
    artifact_path = tmp_path / "job_abc123.mp4"

    recorder = RecordingFrameSource(source, artifact_path)

    out = list(recorder.iter_frames())

    assert [idx for idx, _, _ in out] == [1, 2, 3]
    assert recorder.available is True
    assert artifact_path.exists()
    assert artifact_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(artifact_path))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 3
    finally:
        cap.release()


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.integration
async def test_preprocessed_video_route_requires_matching_owner(tmp_path):
    from api import main
    from api.core.preprocessed_video import (
        clear_preprocessed_video_artifacts,
        register_preprocessed_video_artifact,
    )

    owner = User(
        id=ObjectId(),
        email="owner@example.com",
        name="Owner",
        password_hash="hash",
    )
    other = User(
        id=ObjectId(),
        email="other@example.com",
        name="Other",
        password_hash="hash",
    )
    artifact_path = tmp_path / "artifact.mp4"
    artifact_path.write_bytes(b"fake-mp4")
    register_preprocessed_video_artifact("job_owned", str(owner.id), artifact_path)

    try:
        ok = await main.get_preprocessed_video("job_owned", owner)
        assert ok.status_code == 200
        assert ok.media_type == "video/mp4"
        assert Path(ok.path) == artifact_path

        with pytest.raises(HTTPException) as missing:
            await main.get_preprocessed_video("no_such_job", owner)
        assert missing.value.status_code == 404

        with pytest.raises(HTTPException) as forbidden_as_missing:
            await main.get_preprocessed_video("job_owned", other)
        assert forbidden_as_missing.value.status_code == 404
    finally:
        clear_preprocessed_video_artifacts(delete_files=False)
