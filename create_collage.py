import os
import cv2
import numpy as np
import random

source_dir = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/datasets/ocr/train"
files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(".jpg")]

# Sample 1000 images to find challenges
random.seed(42)
sampled_files = random.sample(files, min(1000, len(files)))

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_brightness(image):
    return np.mean(image)

images_data = []
for f in sampled_files:
    img = cv2.imread(f)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = variance_of_laplacian(gray)
    bright = get_brightness(gray)
    images_data.append({'path': f, 'img': img, 'blur': blur, 'brightness': bright})

# Most blurry (lowest laplacian variance)
images_data.sort(key=lambda x: x['blur'])
blurry_imgs = images_data[:20]

# Most overexposed/glare (highest brightness)
images_data.sort(key=lambda x: x['brightness'], reverse=True)
glare_imgs = images_data[:20]

# Darkest / dirty
images_data.sort(key=lambda x: x['brightness'])
dark_imgs = images_data[:20]

selected_paths = set()
selected = []

def add_selection(group, count=2):
    added = 0
    # Add a bit of randomness so we don't always pick the absolute worst which might be unrecognizable
    # Skip the absolute extreme (top 2) just in case they are completely unreadable
    for data in group[2:]:
        if data['path'] not in selected_paths:
            selected.append(data)
            selected_paths.add(data['path'])
            added += 1
            if added == count:
                break

add_selection(blurry_imgs, 2)
add_selection(glare_imgs, 2)
add_selection(dark_imgs, 2)

h, w = 150, 300
padding = 10
grid_h = h*3 + padding*4
grid_w = w*2 + padding*3
canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

for i, data in enumerate(selected):
    r = i // 2
    c = i % 2
    resized = cv2.resize(data['img'], (w, h))
    
    # Add border
    cv2.rectangle(resized, (0,0), (w-1, h-1), (0,0,0), 2)
    
    y = padding + r * (h + padding)
    x = padding + c * (w + padding)
    canvas[y:y+h, x:x+w] = resized

output_path = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/SOICT_DATN_Application_VIE_Template/Hinhve/challenges_grid.jpg"
cv2.imwrite(output_path, canvas)
print("Created collage with readable challenges at", output_path)
