import os
import csv
BASE_DIR = '/home/vietanh/Documents/DATN/ALPR_Vietnamese'
OCR_IMAGES_DIR = os.path.join(BASE_DIR, 'data/raw/platesmania_vn/ocr/images')
CSV_TRAIN = os.path.join(BASE_DIR, 'runs/infer/quality_router_platesmania_vn_train/predictions.csv')
CSV_VAL = os.path.join(BASE_DIR, 'runs/infer/quality_router_platesmania_vn_val/predictions_val.csv')

pred_lookup = {}
for csv_file in [CSV_TRAIN, CSV_VAL]:
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'path' in row:
                    basename = os.path.basename(str(row['path']))
                    pred_lookup[basename] = row.get('predicted_legibility', 'Unknown')

illegible_files = []
for split in ['train', 'val']:
    split_dir = os.path.join(OCR_IMAGES_DIR, split)
    if os.path.exists(split_dir):
        for fname in os.listdir(split_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                leg = pred_lookup.get(fname, 'Unknown')
                if leg == 'illegible':
                    illegible_files.append(f"{split}/{fname}")

print(f"Total illegible found via os.listdir: {len(illegible_files)}")
print("First 5:", illegible_files[:5])
