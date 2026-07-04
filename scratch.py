import os
import cv2
import random
import sys

# Add api to path
sys.path.append("/home/vietanh/Documents/DATN/ALPR_Vietnamese")
from api.core.quality_router import PlateQualityRouter

data_dir = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/datasets/ocr/train"
images = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
random.seed(42)
random.shuffle(images)

router = PlateQualityRouter()

results = []
count = 0
for img_name in images:
    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    res = router.route(img)
    if res.legibility in ["perfect", "good", "poor"] and not res.tags.occluded:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        results.append({
            "name": img_name,
            "path": img_path,
            "legibility": res.legibility,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": lap_var,
            "size": img.shape[:2]
        })
        count += 1
    if count >= 200:
        break

# We want 1 bright, 1 low light, 1 small, 1 large, 1 blurry but legible
results.sort(key=lambda x: x["brightness"])
low_light = results[0]
bright = results[-1]

results.sort(key=lambda x: x["size"][0] * x["size"][1])
small = results[0]
large = results[-1]

results.sort(key=lambda x: x["sharpness"])
blurry = results[0]
sharp = results[-1]

print("Low light:", low_light)
print("Bright:", bright)
print("Small:", small)
print("Large:", large)
print("Blurry (but legible):", blurry)
print("Sharp:", sharp)

