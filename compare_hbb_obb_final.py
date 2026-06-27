import sys
import os
sys.path.append('.')
import torch
import cv2
import numpy as np
from ultralytics import YOLO

model1_path = 'references/Character-Time-series-Matching/Vietnamese/object.pt'
model2_path = 'weights/detection/best.pt'

img1_path = 'data/datasets/lp_detection_obb/images/train/nomer28290803.jpg'
img2_path = 'data/datasets/lp_detection_obb/images/train/nomer28293909.jpg'

print("Loading models...")
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path=model1_path, force_reload=False)
model2 = YOLO(model2_path)

print("Running inference...")

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image {img_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res1 = model1(img_rgb)
    
    img_hbb = img.copy()
    for *box, conf, cls in res1.xyxy[0].cpu().numpy():
        cls_id = int(cls)
        cls_name = model1.names[cls_id].lower().replace('_', ' ')
        
        # Only draw if it is a rectangle or square license plate
        is_plate = 'rectangle' in cls_name or 'square' in cls_name or 'plate' in cls_name
        if not is_plate:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_hbb, (x1, y1), (x2, y2), (0, 0, 255), 3)

    res2 = model2(img_path)
    img_obb = img.copy()
    
    if res2[0].obb is not None:
        for pts, cls in zip(res2[0].obb.xyxyxyxy.cpu().numpy(), res2[0].obb.cls.cpu().numpy()):
            pts = pts.astype(np.int32)
            cv2.polylines(img_obb, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    
    return img_hbb, img_obb

print(f"Processing {img1_path}...")
hbb1, obb1 = process_image(img1_path)

print(f"Processing {img2_path}...")
try:
    hbb2, obb2 = process_image(img2_path)
except ValueError:
    alt_path = img2_path.replace('nomer28293909', 'normer28293909')
    hbb2, obb2 = process_image(alt_path)

target_height = 600
def resize_to_height(img, h):
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * ratio), h))

print("Resizing and stacking images...")
hbb1 = resize_to_height(hbb1, target_height)
obb1 = resize_to_height(obb1, target_height)
hbb2 = resize_to_height(hbb2, target_height)
obb2 = resize_to_height(obb2, target_height)

min_w1 = min(hbb1.shape[1], obb1.shape[1])
hbb1 = cv2.resize(hbb1, (min_w1, target_height))
obb1 = cv2.resize(obb1, (min_w1, target_height))

min_w2 = min(hbb2.shape[1], obb2.shape[1])
hbb2 = cv2.resize(hbb2, (min_w2, target_height))
obb2 = cv2.resize(obb2, (min_w2, target_height))

row1 = np.hstack((hbb1, obb1))
row2 = np.hstack((hbb2, obb2))

max_width = max(row1.shape[1], row2.shape[1])
if row1.shape[1] < max_width:
    pad = np.zeros((target_height, max_width - row1.shape[1], 3), dtype=np.uint8)
    row1 = np.hstack((row1, pad))
elif row2.shape[1] < max_width:
    pad = np.zeros((target_height, max_width - row2.shape[1], 3), dtype=np.uint8)
    row2 = np.hstack((row2, pad))

grid = np.vstack((row1, row2))

output_path = 'hbb_vs_obb_comparison_final.jpg'
cv2.imwrite(output_path, grid)
print(f"Saved comparison image to {output_path}")
