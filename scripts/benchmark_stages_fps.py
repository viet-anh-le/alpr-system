"""
benchmark_stages_fps.py - Đánh giá FPS của từng giai đoạn trong pipeline
"""
import os
import sys
import time
import random
import glob
import cv2
from collections import defaultdict

# Bật ALPR_DEBUG_TIMINGS trước khi import config
os.environ["ALPR_DEBUG_TIMINGS"] = "1"
os.environ["SMALL_LPR_CTC_CKPT_PATH"] = "weights/ocr/small_lpr_ctc/ctc_finetune_ep55_lr1e4/small_lpr_ctc-epoch=007-val_acc=0.9356.ckpt"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.core.frame_source import FileFrameSource
from api.core.models import load_models
from api.core.pipeline_async import process_frames_async
from api.core.config import FRAME_STRIDE

def main():
    video_dir = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/realworld-videos/chunks"
    all_videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if not all_videos:
        print(f"Không tìm thấy video nào trong {video_dir}")
        return
        
    # Chọn ngẫu nhiên tối đa 10 video
    num_videos = min(10, len(all_videos))
    selected_videos = random.sample(all_videos, num_videos)
    
    print(f"Tiến hành benchmark trên {num_videos} videos...")
    
    print("Đang tải các mô hình (models)...")
    models = load_models()
    
    total_timings = defaultdict(float)
    total_frames = 0
    total_stride_frames = 0
    
    # Dummy emit function
    def emit(event):
        pass

    for i, video_path in enumerate(selected_videos, 1):
        print(f"\n[{i}/{num_videos}] Xử lý video: {os.path.basename(video_path)}")
        source = FileFrameSource(video_path)
        
        # Mở stream để lấy tổng số frames nếu có thể
        # FileFrameSource khởi tạo cv2.VideoCapture bên trong
        source.iter_frames() # để bắt đầu đọc frame nếu cần thiết, nhưng total_frames có sẵn property
        n_frames = source.total_frames
        
        if not n_frames:
            # Ước lượng hoặc bỏ qua nếu không lấy được
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
        if not n_frames:
            print("  Không lấy được số frame, bỏ qua.")
            continue
            
        n_stride_frames = n_frames // FRAME_STRIDE if FRAME_STRIDE > 0 else n_frames
        
        total_frames += n_frames
        total_stride_frames += n_stride_frames
        
        timings = {}
        
        try:
            summary = process_frames_async(
                source=source,
                emit=emit,
                models=models,
                session_id=os.path.basename(video_path),
                record_save=None,
                timings=timings
            )
            
            print(f"  Đã xử lý {summary.get('processed_frames', n_frames)} frames.")
            for stage, t in timings.items():
                total_timings[stage] += t
                
        except Exception as e:
            print(f"  Lỗi khi xử lý {video_path}: {e}")
            
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ TỐC ĐỘ (FPS) TỪNG GIAI ĐOẠN")
    print(f"Tổng số frames: {total_frames}")
    print(f"Tổng số stride frames (cho plate/OCR): {total_stride_frames}")
    print("="*50)
    
    # Tính toán FPS
    # Stage chạy trên mọi frames
    per_frame_stages = ["vehicle_detect", "vehicle_track"]
    # Stage chạy trên stride frames
    per_stride_stages = ["plate_cascade", "association", "classify", "ocr"]
    
    for stage in per_frame_stages:
        if stage in total_timings and total_timings[stage] > 0:
            fps = total_frames / total_timings[stage]
            print(f"- {stage.ljust(20)}: {fps:.2f} FPS (tổng tgian: {total_timings[stage]:.2f}s)")
            
    for stage in per_stride_stages:
        if stage in total_timings and total_timings[stage] > 0:
            fps = total_stride_frames / total_timings[stage]
            print(f"- {stage.ljust(20)}: {fps:.2f} FPS (tổng tgian: {total_timings[stage]:.2f}s)")

    if "total" in total_timings and total_timings["total"] > 0:
        overall_fps = total_frames / total_timings["total"]
        print("-" * 50)
        print(f"- OVERALL PIPELINE   : {overall_fps:.2f} FPS")
        print("="*50)

if __name__ == "__main__":
    main()
