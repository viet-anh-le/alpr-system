"""run_benchmark_async.py — A/B speed benchmark: Synchronous vs Async pipeline.

Usage (conda env):
    conda activate myenv
    python scripts/run_benchmark_async.py [--videos N] [--mode sync|async|both]

Results are saved to:
    data/benchmark/results/pipeline_async/timing_ab_comparison.csv
    data/benchmark/results/pipeline_async/results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.core.frame_source import FileFrameSource
from api.core.models import ModelBundle, load_models
from api.core.pipeline_core import process_frames as process_sync
from api.core.pipeline_async import process_frames_async as process_async

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_VIDEOS_DIR = (
    "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/benchmark/videos"
)
RESULTS_DIR = (
    "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/benchmark/results/pipeline_async"
)
RESULTS_FILE      = os.path.join(RESULTS_DIR, "results.json")
AB_TIMING_FILE    = os.path.join(RESULTS_DIR, "timing_ab_comparison.csv")
DETAIL_TIMING_FILE = os.path.join(RESULTS_DIR, "timing_detail.csv")


# ── Per-video processor ────────────────────────────────────────────────────────

def _collect_emit(events: list) -> callable:
    """Return an emit() callback that collects final vehicle events."""
    def emit(event: dict) -> None:
        if event["type"] == "vehicle" and event.get("final"):
            events.append(event)
        elif event["type"] == "error":
            print(f"  [error] {event.get('message')}")
    return emit


def process_video(
    video_path: str,
    models: ModelBundle,
    mode: str,
    crops_dir: str,
) -> tuple[list, dict]:
    """Run one video through the chosen pipeline and return (results, metrics)."""
    video_name  = os.path.basename(video_path)
    os.makedirs(crops_dir, exist_ok=True)

    video_results: list[dict] = []
    emit = _collect_emit(video_results)

    def record_save(session_id, tid, tracker, char_probs, ocr_method, vote_summary, loop):
        buf = tracker._buffers.get(tid)
        if not buf or not buf.crops:
            return
        v_img = tracker._vehicle_img.get(tid)
        if v_img is not None:
            cv2.imwrite(os.path.join(crops_dir, f"veh_{tid}_thumbnail.jpg"), v_img)
        for i, (crop, q_score) in enumerate(zip(buf.crops, buf.quality_scores)):
            fname = f"veh_{tid}_plate_{i}_q{q_score:.2f}.jpg"
            cv2.imwrite(os.path.join(crops_dir, fname), crop)

    timings: dict[str, float] = {}
    source = FileFrameSource(video_path)

    wall_start = time.perf_counter()

    if mode == "async":
        summary = process_async(
            source=source,
            emit=emit,
            models=models,
            session_id=video_name,
            record_save=record_save,
            timings=timings,
        )
    else:  # sync
        summary = process_sync(
            source=source,
            emit=emit,
            models=models,
            session_id=video_name,
            loop=True,   # truthy dummy so record_save fires
            record_save=record_save,
            timings=timings,
        )

    wall_elapsed = time.perf_counter() - wall_start
    processed_frames = int(summary.get("processed_frames", 0))
    fps = processed_frames / wall_elapsed if wall_elapsed > 0 else 0.0

    # Annotate vehicle results
    for r in video_results:
        r.update({
            "video_name": video_name,
            "pipeline_mode": mode,
            "video_fps": round(fps, 2),
        })

    metrics = {
        "video_name": video_name,
        "pipeline_mode": mode,
        "processed_frames": processed_frames,
        "total_vehicles": summary.get("total_vehicles", 0),
        "wall_time_s": round(wall_elapsed, 4),
        "fps": round(fps, 2),
        **{k: round(v, 4) for k, v in timings.items()},
    }
    return video_results, metrics


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ALPR Pipeline Benchmark (Sync vs Async)")
    parser.add_argument(
        "--videos", type=int, default=None,
        help="Max number of videos to benchmark (default: all)",
    )
    parser.add_argument(
        "--mode", choices=["sync", "async", "both"], default="both",
        help="Which pipeline to benchmark (default: both for A/B comparison)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(BENCHMARK_VIDEOS_DIR):
        print(f"[ERROR] Videos dir not found: {BENCHMARK_VIDEOS_DIR}")
        print("Run scripts/create_benchmark_dataset.py first.")
        sys.exit(1)

    video_files = sorted(
        f for f in os.listdir(BENCHMARK_VIDEOS_DIR) if f.endswith(".mp4")
    )
    if args.videos:
        video_files = video_files[: args.videos]

    if not video_files:
        print(f"[ERROR] No .mp4 files found in {BENCHMARK_VIDEOS_DIR}")
        sys.exit(1)

    print(f"\nLoading AI models…")
    models = load_models()
    print(f"Models ready. Benchmarking {len(video_files)} video(s), mode={args.mode!r}\n")

    modes_to_run: list[str] = (
        ["sync", "async"] if args.mode == "both" else [args.mode]
    )

    all_results:  list[dict] = []
    timing_rows:  list[dict] = []

    # ── A/B loop ──────────────────────────────────────────────────────────────
    for video in video_files:
        video_path = os.path.join(BENCHMARK_VIDEOS_DIR, video)
        row: dict = {"video_name": video}

        for mode in modes_to_run:
            print(f"  [{mode:5s}] {video} …", end="", flush=True)
            crops_dir = os.path.join(RESULTS_DIR, "crops", mode, video)
            try:
                results, metrics = process_video(video_path, models, mode, crops_dir)
                fps = metrics["fps"]
                n_veh = metrics["total_vehicles"]
                print(f" {fps:6.2f} FPS  |  {n_veh} vehicles")
                all_results.extend(results)
                row[f"{mode}_fps"] = fps
                row[f"{mode}_frames"] = metrics["processed_frames"]
                row[f"{mode}_vehicles"] = n_veh
                row[f"{mode}_wall_s"] = metrics["wall_time_s"]
                timing_rows.append(metrics)

                # Print queue-stall diagnosis for async runs
                if mode == "async":
                    _print_stall_diagnosis(metrics)

            except Exception as exc:
                print(f" ERROR: {exc}")
                row[f"{mode}_fps"] = None

        if args.mode == "both" and "sync_fps" in row and "async_fps" in row:
            if row["sync_fps"] and row["async_fps"]:
                speedup = row["async_fps"] / row["sync_fps"]
                row["speedup_x"] = round(speedup, 2)
                print(f"  {'→':5s} Speedup: {speedup:.2f}× (async vs sync)")

        # Save incrementally so progress isn't lost on crash
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── A/B summary CSV ────────────────────────────────────────────────────────
    if timing_rows:
        # Per-video A/B comparison
        ab_cols = ["video_name"]
        for m in modes_to_run:
            ab_cols += [f"{m}_fps", f"{m}_frames", f"{m}_vehicles", f"{m}_wall_s"]
        if args.mode == "both":
            ab_cols.append("speedup_x")

        # Build ab_rows from timing_rows (one row per video per mode → pivot)
        ab_by_video: dict[str, dict] = {}
        for t in timing_rows:
            vn   = t["video_name"]
            mode = t["pipeline_mode"]
            if vn not in ab_by_video:
                ab_by_video[vn] = {"video_name": vn}
            ab_by_video[vn][f"{mode}_fps"]     = t["fps"]
            ab_by_video[vn][f"{mode}_frames"]  = t["processed_frames"]
            ab_by_video[vn][f"{mode}_vehicles"] = t["total_vehicles"]
            ab_by_video[vn][f"{mode}_wall_s"]  = t["wall_time_s"]

        if args.mode == "both":
            for row in ab_by_video.values():
                sf = row.get("sync_fps") or 0
                af = row.get("async_fps") or 0
                row["speedup_x"] = round(af / sf, 2) if sf > 0 else None

        with open(AB_TIMING_FILE, "w", newline="", encoding="utf-8") as f:
            valid_cols = [c for c in ab_cols if any(c in r for r in ab_by_video.values())]
            writer = csv.DictWriter(f, fieldnames=valid_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ab_by_video.values())

        # Detailed per-stage timing
        detail_keys = sorted({k for r in timing_rows for k in r})
        preferred = [
            "video_name", "pipeline_mode", "processed_frames", "total_vehicles",
            "wall_time_s", "fps", "total",
            "vehicle_detect", "vehicle_track", "crop_prep",
            "plate_cascade", "plate_postprocess", "association", "ocr",
            "s1_put_stall", "s2_get_stall", "s2_put_stall", "s3_get_stall",
        ]
        detail_cols = [k for k in preferred if k in detail_keys] + [
            k for k in detail_keys if k not in preferred
        ]
        with open(DETAIL_TIMING_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=detail_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(timing_rows)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"  Results JSON  : {RESULTS_FILE}")
    print(f"  A/B Comparison: {AB_TIMING_FILE}")
    print(f"  Stage Timings : {DETAIL_TIMING_FILE}")

    if args.mode == "both" and timing_rows:
        sync_fps_list  = [r["fps"] for r in timing_rows if r["pipeline_mode"] == "sync"]
        async_fps_list = [r["fps"] for r in timing_rows if r["pipeline_mode"] == "async"]
        if sync_fps_list and async_fps_list:
            avg_sync  = sum(sync_fps_list)  / len(sync_fps_list)
            avg_async = sum(async_fps_list) / len(async_fps_list)
            print(f"\n  Avg Sync  FPS : {avg_sync:.2f}")
            print(f"  Avg Async FPS : {avg_async:.2f}")
            print(f"  Overall Speedup: {avg_async / avg_sync:.2f}×")
    print("=" * 60)


if __name__ == "__main__":
    main()
