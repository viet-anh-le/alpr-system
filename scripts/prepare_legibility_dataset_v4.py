import os
import shutil
import csv
from pathlib import Path

def main():
    base_dir = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese")
    ocr_images_dir = base_dir / 'data/raw/platesmania_vn/ocr/images'
    train_csv = base_dir / "runs/infer/quality_router_platesmania_vn_train/predictions.csv"
    val_csv = base_dir / "runs/infer/quality_router_platesmania_vn_val/predictions_val.csv"
    output_dir = base_dir / "data/datasets/legibility_finetune"
    
    # Xóa thư mục cũ của lớp illegible để đảm bảo không bị sót file rác từ lần trước
    for split in ['train', 'val']:
        ill_dir = output_dir / split / 'illegible'
        if ill_dir.exists():
            print(f"Xóa thư mục cũ: {ill_dir}")
            shutil.rmtree(ill_dir)
            
    # Đọc label và conf từ CSV
    pred_lookup = {}
    print("Đang đọc dữ liệu từ các file predictions...")
    for csv_file in [train_csv, val_csv]:
        if csv_file.exists():
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'path' in row:
                        basename = os.path.basename(str(row['path']))
                        try:
                            conf = float(row.get('router_conf', 0.0))
                        except ValueError:
                            conf = 0.0
                            
                        pred_lookup[basename] = {
                            'label': row.get('predicted_legibility', 'Unknown'),
                            'conf': conf
                        }
    
    illegible_files = []
    other_clean_files = [] # good, perfect, poor
    
    print("Đang quét các file ảnh...")
    for split in ['train', 'val']:
        split_dir = ocr_images_dir / split
        if split_dir.exists():
            for fname in os.listdir(split_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pred = pred_lookup.get(fname)
                    if pred is None:
                        continue
                        
                    label = pred['label']
                    conf = pred['conf']
                    
                    if label == 'illegible':
                        # CHỈ LẤY CÁC FILE CÓ CONF >= 0.9999
                        if conf >= 0.9999:
                            illegible_files.append({
                                'split': split,
                                'fname': fname,
                                'label': label,
                                'src_path': split_dir / fname
                            })
                    elif label in ['good', 'perfect', 'poor']:
                        other_clean_files.append({
                            'split': split,
                            'fname': fname,
                            'label': label,
                            'src_path': split_dir / fname
                        })
                        
    print(f"Tổng số ảnh illegible thỏa mãn conf >= 0.9999: {len(illegible_files)}")
    print(f"Tổng số ảnh sạch khác (good/perfect/poor): {len(other_clean_files)}")
    
    final_files = other_clean_files + illegible_files
    
    print(f"\nBắt đầu copy {len(final_files)} file vào thư mục dataset...")
    count = 0
    missing = 0
    
    for i, item in enumerate(final_files):
        if i % 1000 == 0:
            print(f"Đã copy {i}/{len(final_files)} files...")
            
        src_path = item['src_path']
        split = item['split']
        label = item['label']
        
        if not src_path.exists():
            missing += 1
            continue
            
        dst_dir = output_dir / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        dst_path = dst_dir / src_path.name
        
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        count += 1
        
    print(f"\nHoàn tất! Đã copy {count} ảnh. Bỏ qua {missing} ảnh.")
    print(f"Dataset đã được làm sạch và sẵn sàng tại: {output_dir}")

if __name__ == "__main__":
    main()
