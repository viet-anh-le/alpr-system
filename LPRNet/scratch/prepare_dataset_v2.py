import os
import shutil
import csv
import random

def format_label(label_raw):
    label_raw = label_raw.strip().upper()
    if ' ' not in label_raw:
        return label_raw

    p1, p2 = label_raw.split(' ', 1)
    p2 = p2.replace(' ', '') # remove any extra spaces
    
    if len(p1) == 4:
        p1 = f"{p1[:2]}-{p1[2:]}"
        
    if len(p2) == 4:
        return f"{p1}-{p2}"
    elif len(p2) == 5:
        if any(c.isalpha() for c in p2[-2:]):
            return f"{p1}-{p2[:3]}-{p2[3:]}"
        else:
            return f"{p1}-{p2}"
    else:
        return f"{p1}-{p2}"

def main():
    datas_dir = './datas'
    if os.path.exists(datas_dir):
        print(f"Removing existing {datas_dir}...")
        shutil.rmtree(datas_dir)
    
    train_dir = os.path.join(datas_dir, 'train')
    valid_dir = os.path.join(datas_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    dataset = []

    print("Parsing crop_labels.csv...")
    with open('VN-License-Plate-OCR/labels/crop_labels.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['Name']
            label_raw = row['Label']
            if not label_raw:
                continue
            img_path = os.path.join('VN-License-Plate-OCR', 'cropped', img_name)
            if os.path.exists(img_path):
                label = format_label(label_raw)
                dataset.append((img_path, label))

    print("Parsing gen_labels.csv...")
    with open('VN-License-Plate-OCR/labels/gen_labels.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['Name']
            label_raw = row['Label']
            if not label_raw:
                continue
            img_path = os.path.join('VN-License-Plate-OCR', 'generated', img_name)
            if os.path.exists(img_path):
                label = format_label(label_raw)
                dataset.append((img_path, label))

    random.seed(42)  # For reproducibility
    random.shuffle(dataset)

    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    valid_data = dataset[split_idx:]

    def copy_files(data, dest_dir):
        for idx, (img_path, label) in enumerate(data):
            ext = os.path.splitext(img_path)[1]
            new_name = f"{label}#{idx}{ext}"
            new_path = os.path.join(dest_dir, new_name)
            shutil.copy2(img_path, new_path)

    print(f"Copying {len(train_data)} training samples...")
    copy_files(train_data, train_dir)
    
    print(f"Copying {len(valid_data)} validation samples...")
    copy_files(valid_data, valid_dir)

    print("Preparation complete!")

if __name__ == '__main__':
    main()
