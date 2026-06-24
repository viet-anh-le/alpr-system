import urllib.request
import re

# Fetch from webapp
webapp_images = []
for p in range(1, 37):
    url = f"http://localhost:5000/?filter=illegible&sort=none&page={p}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            matches = re.findall(r'id="card-([^"]+)"', html)
            webapp_images.extend(matches)
    except Exception as e:
        print(f"Error fetching page {p}: {e}")
        break

print(f"Webapp collected: {len(webapp_images)}")

# Now read from the dataset we generated in v3
import os
v3_dir = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/datasets/legibility_finetune/train/illegible"
v3_images = []
if os.path.exists(v3_dir):
    v3_images = os.listdir(v3_dir)

print(f"v3 collected: {len(v3_images)}")

webapp_set = set(webapp_images)
v3_set = set(v3_images)

print(f"Intersection: {len(webapp_set.intersection(v3_set))}")
print(f"In Webapp but not in v3: {len(webapp_set - v3_set)}")
print(f"In v3 but not in Webapp: {len(v3_set - webapp_set)}")
