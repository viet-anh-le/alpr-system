"""Upload a video to the ALPR backend and append vehicle events to a CSV log.

Run from the repository root while the FastAPI backend is running:

    /home/vietanh/anaconda3/envs/myenv/bin/python scripts/test_backend_upload_log.py \
        data/realworld-videos/chunks/đoạn_004.mp4 \
        --log data/outputs/backend_upload_log.csv

The script exercises the same web upload path:
  POST /upload -> GET /stream/{job_id}
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import requests


DEFAULT_BACKEND = "http://localhost:8000"
DEFAULT_LOG = Path("data/outputs/backend_upload_log.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ALPR backend upload flow and append detected vehicles to a CSV log."
    )
    parser.add_argument("video", type=Path, help="Path to the video file to upload.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Backend base URL. Default: {DEFAULT_BACKEND}",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"CSV log file to append. Default: {DEFAULT_LOG}",
    )
    parser.add_argument(
        "--preprocess-mode",
        default="none",
        help="Value sent as preprocess_mode in the upload form. Default: none",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Also log rejected_vehicle events.",
    )
    parser.add_argument(
        "--include-final-duplicates",
        action="store_true",
        help="Log final snapshot events even if the same track was already logged.",
    )
    return parser.parse_args()


def video_fps(video_path: Path) -> float | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0 else None
    finally:
        cap.release()


def upload_video(
    backend: str,
    video_path: Path,
    preprocess_mode: str,
) -> str:
    upload_url = f"{backend.rstrip('/')}/upload"
    with video_path.open("rb") as fh:
        files = {"file": (video_path.name, fh, "video/mp4")}
        data = {"preprocess_mode": preprocess_mode}
        response = requests.post(upload_url, files=files, data=data, timeout=(10, 300))
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Upload response did not contain job_id: {payload}")
    return str(job_id)


def iter_sse_events(backend: str, job_id: str):
    stream_url = f"{backend.rstrip('/')}/stream/{job_id}"
    with requests.get(stream_url, stream=True, timeout=(10, None)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if not raw:
                continue
            yield json.loads(raw)


def ensure_log_writer(log_path: Path) -> tuple[Any, csv.DictWriter]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    fh = log_path.open("a", newline="", encoding="utf-8")
    fieldnames = [
        "detected_at",
        "upload_file",
        "job_id",
        "track_id",
        "event_type",
        "plate",
        "vehicle_class",
        "approx_frame",
        "approx_video_time_sec",
        "ocr_frames",
        "confidence",
        "done",
        "final",
        "vote_summary",
    ]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if needs_header:
        writer.writeheader()
        fh.flush()
    return fh, writer


def build_log_row(
    event: dict[str, Any],
    *,
    upload_file: str,
    job_id: str,
    latest_progress_frame: int | None,
    fps: float | None,
) -> dict[str, Any]:
    approx_time = None
    if latest_progress_frame is not None and fps:
        approx_time = round(latest_progress_frame / fps, 3)

    return {
        "detected_at": datetime.now().isoformat(timespec="seconds"),
        "upload_file": upload_file,
        "job_id": job_id,
        "track_id": event.get("id", ""),
        "event_type": event.get("type", ""),
        "plate": event.get("plate", ""),
        "vehicle_class": event.get("cls", ""),
        "approx_frame": latest_progress_frame if latest_progress_frame is not None else "",
        "approx_video_time_sec": approx_time if approx_time is not None else "",
        "ocr_frames": event.get("ocr_frames", ""),
        "confidence": event.get("confidence", ""),
        "done": event.get("done", ""),
        "final": event.get("final", ""),
        "vote_summary": json.dumps(event.get("vote_summary", {}), ensure_ascii=False),
    }


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    backend = args.backend.rstrip("/")
    fps = video_fps(video_path)
    job_id = upload_video(backend, video_path, args.preprocess_mode)
    print(f"Uploaded {video_path.name} -> job_id={job_id}")
    print(f"Appending detections to {args.log}")

    latest_progress_frame: int | None = None
    logged_tracks: set[tuple[str, int]] = set()

    fh, writer = ensure_log_writer(args.log)
    with fh:
        for event in iter_sse_events(backend, job_id):
            event_type = event.get("type")

            if event_type == "progress":
                frame = event.get("frame")
                latest_progress_frame = int(frame) if frame is not None else latest_progress_frame
                continue

            if event_type == "error":
                raise RuntimeError(event.get("message", "Backend returned an error event"))

            should_log = event_type == "vehicle" or (
                args.include_rejected and event_type == "rejected_vehicle"
            )
            if should_log:
                track_id = int(event.get("id", -1))
                dedupe_key = (str(event_type), track_id)
                if args.include_final_duplicates or dedupe_key not in logged_tracks:
                    row = build_log_row(
                        event,
                        upload_file=video_path.name,
                        job_id=job_id,
                        latest_progress_frame=latest_progress_frame,
                        fps=fps,
                    )
                    writer.writerow(row)
                    fh.flush()
                    logged_tracks.add(dedupe_key)
                    print(
                        f"{row['detected_at']} | {row['upload_file']} | "
                        f"{row['vehicle_class']} | {row['plate']}"
                    )

            if event_type == "complete":
                print(f"Complete: total_vehicles={event.get('total_vehicles', '')}")
                break


if __name__ == "__main__":
    main()
