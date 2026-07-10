import os
import random

# Configuration
SOURCE_DIR = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/realworld-videos/chunks"
BENCHMARK_DIR = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/benchmark"
VIDEOS_DIR = os.path.join(BENCHMARK_DIR, "videos")

# Target distribution
DISTRIBUTION = {
    "đoạn": 13,
    "hcm": 13,        # matches hcm_night
    "hn_night": 12,
    "hn_oto": 12
}

def create_benchmark_dataset():
    # 1. Ensure benchmark directories exist
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    # 2. Gather all videos from source
    all_videos = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')]
    
    # 3. Group videos by prefix
    grouped_videos = {prefix: [] for prefix in DISTRIBUTION.keys()}
    for video in all_videos:
        for prefix in DISTRIBUTION.keys():
            if video.startswith(prefix):
                grouped_videos[prefix].append(video)
                break
                
    # 4. Sample videos and create symlinks
    total_sampled = 0
    for prefix, count in DISTRIBUTION.items():
        available = len(grouped_videos[prefix])
        print(f"Prefix '{prefix}': target {count}, available {available}")
        
        # Randomly sample 'count' videos
        sampled = random.sample(grouped_videos[prefix], min(count, available))
        
        for video in sampled:
            src_path = os.path.join(SOURCE_DIR, video)
            dst_path = os.path.join(VIDEOS_DIR, video)
            
            # Create a symlink (or copy if preferred)
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
            total_sampled += 1
            
    print(f"Successfully sampled {total_sampled} videos into {VIDEOS_DIR}")

if __name__ == "__main__":
    create_benchmark_dataset()
