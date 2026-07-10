"""Run the ALPR pipeline locally on a video and append detections to a CSV log.

This does not call the FastAPI backend. It loads the models in-process and runs
the same pipeline code used by the application.

Example:

    /home/vietanh/anaconda3/envs/myenv/bin/python scripts/test_pipeline_log.py \
        data/realworld-videos/chunks/đoạn_004.mp4 \
        --log data/outputs/pipeline_log.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.core.frame_source import FileFrameSource  # noqa: E402
from api.core.models import load_models  # noqa: E402
from api.core.pipeline_async import process_frames_async  # noqa: E402
from api.core.pipeline_core import process_frames  # noqa: E402


DEFAULT_LOG = Path("data/outputs/pipeline_log.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local ALPR pipeline inference and append vehicle results to a CSV log."
    )
    parser.add_argument("video", type=Path, help="Path to the input video.")
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"CSV log file to append. Default: {DEFAULT_LOG}",
    )
    parser.add_argument(
        "--mode",
        choices=("async", "sync"),
        default="async",
        help="Pipeline implementation to run. Default: async",
    )
    parser.add_argument("--t-start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--t-end", type=float, default=None, help="End time in seconds.")
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


def ensure_log_writer(log_path: Path) -> tuple[Any, csv.DictWriter]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    fh = log_path.open("a", newline="", encoding="utf-8")
    fieldnames = [
        "detected_at",
        "input_file",
        "pipeline_mode",
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
    input_file: str,
    pipeline_mode: str,
    latest_progress_frame: int | None,
    fps: float,
) -> dict[str, Any]:
    approx_time = ""
    if latest_progress_frame is not None and fps > 0:
        approx_time = round(max(latest_progress_frame - 1, 0) / fps, 3)

    return {
        "detected_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": input_file,
        "pipeline_mode": pipeline_mode,
        "track_id": event.get("id", ""),
        "event_type": event.get("type", ""),
        "plate": event.get("plate", ""),
        "vehicle_class": event.get("cls", ""),
        "approx_frame": latest_progress_frame if latest_progress_frame is not None else "",
        "approx_video_time_sec": approx_time,
        "ocr_frames": event.get("ocr_frames", ""),
        "confidence": event.get("confidence", ""),
        "done": event.get("done", ""),
        "final": event.get("final", ""),
        "vote_summary": json.dumps(event.get("vote_summary", {}), ensure_ascii=False),
    }


def select_pipeline(mode: str) -> Callable:
    return process_frames_async if mode == "async" else process_frames


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    source = FileFrameSource(video_path, t_start=args.t_start, t_end=args.t_end)
    pipeline = select_pipeline(args.mode)
    models = load_models()

    latest_progress_frame: int | None = None
    logged_tracks: set[tuple[str, int]] = set()
    fh, writer = ensure_log_writer(args.log)

    def emit(event: dict[str, Any]) -> None:
        nonlocal latest_progress_frame
        event_type = event.get("type")

        if event_type == "progress":
            frame = event.get("frame")
            latest_progress_frame = int(frame) if frame is not None else latest_progress_frame
            return

        should_log = event_type == "vehicle" or (
            args.include_rejected and event_type == "rejected_vehicle"
        )
        if not should_log:
            return

        track_id = int(event.get("id", -1))
        dedupe_key = (str(event_type), track_id)
        if not args.include_final_duplicates and dedupe_key in logged_tracks:
            return

        row = build_log_row(
            event,
            input_file=video_path.name,
            pipeline_mode=args.mode,
            latest_progress_frame=latest_progress_frame,
            fps=source.fps,
        )
        writer.writerow(row)
        fh.flush()
        logged_tracks.add(dedupe_key)
        print(
            f"{row['detected_at']} | frame~{row['approx_frame']} | "
            f"{row['vehicle_class']} | {row['plate']}"
        )

    print(f"Running {args.mode} pipeline on {video_path}")
    print(f"Appending detections to {args.log}")
    with fh:
        summary = pipeline(source, emit=emit, models=models)

    print(
        "Complete: "
        f"processed_frames={summary['processed_frames']} "
        f"total_vehicles={summary['total_vehicles']}"
    )


if __name__ == "__main__":
    main()
