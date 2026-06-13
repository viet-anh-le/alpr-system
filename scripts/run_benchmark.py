import os
import json
import time
import sys
import csv
import cv2
# Add parent directory to path so we can import from api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.core.models import ModelBundle, load_models
from api.core.frame_source import FileFrameSource
from api.core.pipeline_core import process_frames

BENCHMARK_VIDEOS_DIR = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/benchmark/videos"
RESULTS_DIR = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/benchmark/results/pipeline_A"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.json")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "results_summary.csv")
TIMING_FILE = os.path.join(RESULTS_DIR, "timing_summary.csv")

def process_video(video_path: str, models: ModelBundle) -> tuple[list, dict]:
    video_results = []
    video_name = os.path.basename(video_path)
    
    # Create directory for saving crops for this video
    video_crops_dir = os.path.join(RESULTS_DIR, "crops", video_name)
    os.makedirs(video_crops_dir, exist_ok=True)
    
    def emit(event: dict) -> None:
        if event["type"] == "vehicle" and event.get("final"):
            res = {
                "video_name": video_name,
                "vehicle_id": event["id"],
                "vehicle_class": event.get("cls", "unknown"),
                "plate_text": event.get("plate", ""),
                "confidence": event.get("confidence", 0.0),
                "ocr_frames_processed": event.get("ocr_frames", 0),
                "char_details": event.get("chars", [])
            }
            video_results.append(res)
        elif event["type"] == "error":
            print(f"Error processing {video_name}: {event.get('message')}")

    def my_record_save(session_id, tid, tracker, char_probs, ocr_method, vote_summary, loop):
        # This callback is invoked when a track is finalized
        buf = tracker._buffers.get(tid)
        if not buf or not buf.crops:
            return
            
        # Optional: Save the full vehicle thumbnail
        v_img = tracker._vehicle_img.get(tid)
        if v_img is not None:
            cv2.imwrite(os.path.join(video_crops_dir, f"veh_{tid}_thumbnail.jpg"), v_img)
            
        # Save all plate crops buffered for this vehicle
        for i, (crop, q_score) in enumerate(zip(buf.crops, buf.quality_scores)):
            filename = f"veh_{tid}_plate_crop_{i}_q{q_score:.2f}.jpg"
            filepath = os.path.join(video_crops_dir, filename)
            cv2.imwrite(filepath, crop)
            
    timings: dict[str, float] = {}
    start_time = time.time()
    
    source = FileFrameSource(video_path)
    # process_frames is fully synchronous
    summary = process_frames(
        source=source,
        emit=emit,
        models=models,
        session_id=video_name,
        loop=True,  # Dummy value to trigger record_save
        record_save=my_record_save,
        timings=timings,
    )
    
    end_time = time.time()
    
    elapsed = end_time - start_time
    processed_frames = int(summary.get("processed_frames", 0))
    fps = processed_frames / elapsed if elapsed > 0 else 0.0
    metrics = {
        "video_name": video_name,
        "processed_frames": processed_frames,
        "elapsed_s": round(elapsed, 4),
        "fps": round(fps, 2),
        **{k: round(v, 4) for k, v in timings.items()},
    }

    # Calculate inference time per vehicle
    if video_results:
        time_per_vehicle = (elapsed * 1000) / len(video_results)
        for r in video_results:
            r["inference_time_ms"] = round(time_per_vehicle, 2)
            r["video_fps"] = metrics["fps"]
            
    return video_results, metrics

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading AI Models (this takes a moment)...")
    models = load_models()
    print("Models loaded successfully.")
    
    all_results = []
    timing_results = []
    processed_videos = set()
    
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            processed_videos = {r.get("video_name") for r in all_results if r.get("video_name")}
            if processed_videos:
                print(f"Resuming from previous run. Skipped {len(processed_videos)} videos already processed.")
        except Exception as e:
            print(f"Could not load previous results (starting fresh): {e}")
            all_results = []
    
    if not os.path.exists(BENCHMARK_VIDEOS_DIR):
        print(f"Directory {BENCHMARK_VIDEOS_DIR} not found. Did you run create_benchmark_dataset.py?")
        return

    videos = sorted([f for f in os.listdir(BENCHMARK_VIDEOS_DIR) if f.endswith('.mp4')])
    pending_videos = [v for v in videos if v not in processed_videos]
    print(f"Found {len(videos)} total videos, {len(pending_videos)} pending processing.")
    
    for i, video in enumerate(pending_videos):
        video_path = os.path.join(BENCHMARK_VIDEOS_DIR, video)
        print(f"[{i+1}/{len(pending_videos)}] Processing {video}...")
        try:
            results, metrics = process_video(video_path, models)
            print(f"  -> Found {len(results)} valid vehicles at {metrics['fps']} FPS.")
            all_results.extend(results)
            timing_results.append(metrics)
        except Exception as e:
            print(f"  -> Fatal error on video {video}: {e}")
        
        # Save intermediate results so progress isn't lost if it crashes
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
            
        # Also write the simplified summary CSV
        with open(SUMMARY_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "vehicle_class", "plate_text"])
            for r in all_results:
                writer.writerow([r["video_name"], r["vehicle_class"], r["plate_text"]])

        if timing_results:
            timing_keys = sorted({key for row in timing_results for key in row})
            preferred = ["video_name", "processed_frames", "elapsed_s", "fps", "total", "vehicle_detect", "vehicle_track", "crop_prep", "plate_cascade", "plate_postprocess", "association", "ocr"]
            columns = [key for key in preferred if key in timing_keys] + [key for key in timing_keys if key not in preferred]
            with open(TIMING_FILE, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(timing_results)
            
    print(f"\nAll done! Detailed results saved to {RESULTS_FILE}")
    print(f"Summary results saved to {SUMMARY_FILE}")
    print(f"Timing results saved to {TIMING_FILE}")

if __name__ == "__main__":
    main()
