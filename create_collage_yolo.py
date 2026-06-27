import os
import cv2
import numpy as np
import random
from ultralytics import YOLO

model_path = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/runs/classify/runs/classify/legibility_finetuned_vn/weights/best.pt"
model = YOLO(model_path)

source_dir = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/datasets/ocr/train"
files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(".jpg")]

random.seed(123)
sampled_files = random.sample(files, min(200, len(files)))

# Batch inference
results = model(sampled_files, verbose=False)

poor_candidates = []
borderline_candidates = []

for i, r in enumerate(results):
    path = sampled_files[i]
    probs = r.probs
    top1_conf = probs.top1conf.item()
    top1_class = r.names[probs.top1]
    
    if top1_class.lower() == 'poor':
        poor_candidates.append((path, top1_conf))
    else:
        # If it's good but with low confidence, it's borderline
        borderline_candidates.append((path, top1_conf))

# Sort poor by confidence
poor_candidates.sort(key=lambda x: x[1], reverse=True)
borderline_candidates.sort(key=lambda x: x[1])

selected_paths = []
# Take 6 most confident poor
for p, c in poor_candidates[:6]:
    selected_paths.append(p)

# Create grid
h, w = 150, 300
padding = 10
grid_h = h*3 + padding*4
grid_w = w*2 + padding*3
canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

for i, path in enumerate(selected_paths[:6]):
    img = cv2.imread(path)
    r = i // 2
    c = i % 2
    resized = cv2.resize(img, (w, h))
    cv2.rectangle(resized, (0,0), (w-1, h-1), (0,0,0), 2)
    y = padding + r * (h + padding)
    x = padding + c * (w + padding)
    canvas[y:y+h, x:x+w] = resized

output_path = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/SOICT_DATN_Application_VIE_Template/Hinhve/challenges_grid.jpg"
cv2.imwrite(output_path, canvas)
print(f"Created collage using YOLO legibility model. Picked {len(poor_candidates)} poor and {len(borderline_candidates)} good. Saved to {output_path}")
