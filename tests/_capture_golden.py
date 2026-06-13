"""One-off: run the current run_job against the fixture and dump the event
stream. Run with: python tests/_capture_golden.py"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from api.core.models import load_models
from api.core.pipeline import run_job


def main() -> None:
    fixture_src = Path("tests/fixtures/short_clip.mp4")
    # Copy to a temp file because run_job will os.unlink() it at the end
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        shutil.copy(fixture_src, tmp.name)
        fixture = tmp.name

    captured: list[dict] = []

    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    async def drain() -> None:
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                break
            drop = {"plate_b64", "vehicle_b64", "detail"}
            captured.append({k: v for k, v in ev.items() if k not in drop})

    models = load_models()
    jobs: dict = {}
    run_job(
        video_path=fixture,
        job_id="golden",
        queue=queue,
        loop=loop,
        models=models,
        jobs=jobs,
        filename="short_clip.mp4",
        mjpeg_queue=None,
    )
    loop.run_until_complete(drain())

    out = Path("tests/fixtures/golden_run_job_events.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(captured, indent=2, default=str))
    print(f"wrote {len(captured)} events to {out}")


if __name__ == "__main__":
    main()
