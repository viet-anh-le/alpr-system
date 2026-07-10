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
    
    # Đọc label từ CSV
    pred_lookup = {}
    print("Đang đọc nhãn từ các file predictions...")
    for csv_file in [train_csv, val_csv]:
        if csv_file.exists():
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'path' in row:
                        basename = os.path.basename(str(row['path']))
                        pred_lookup[basename] = row.get('predicted_legibility', 'Unknown')
    
    # Tái tạo lại chính xác thứ tự của os.listdir() giống như web app
    # để lấy ra 3600 file illegible đầu tiên (tương đương 36 trang trên UI)
    print("Đang duyệt thư mục ảnh theo đúng thứ tự của web app (os.listdir)...")
    
    illegible_files_ordered = []
    other_clean_files = [] # good, perfect, poor
    
    for split in ['train', 'val']:
        split_dir = ocr_images_dir / split
        if split_dir.exists():
            # os.listdir() trả về danh sách có thứ tự phụ thuộc vào filesystem (inode order)
            # Đây chính là thứ tự đã hiển thị trên giao diện web khi sort='none'
            for fname in os.listdir(split_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label = pred_lookup.get(fname, 'Unknown')
                    
                    if label == 'illegible':
                        illegible_files_ordered.append({
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
                        
    print(f"Tổng số ảnh illegible tìm thấy theo os.listdir: {len(illegible_files_ordered)}")
    print(f"Tổng số ảnh sạch khác (good/perfect/poor): {len(other_clean_files)}")
    
    # Cắt lấy 3600 ảnh illegible đầu tiên
    top_3600_illegible = illegible_files_ordered[:3600]
    print(f"Sẽ giữ lại {len(top_3600_illegible)} ảnh illegible đầu tiên (đúng 36 trang trên web).")
    
    # Gộp tất cả lại
    final_files = other_clean_files + top_3600_illegible
    
    print(f"\nBắt đầu copy {len(final_files)} file vào thư mục dataset...")
    count = 0
    missing = 0
    
    # Copy files
    for i, item in enumerate(final_files):
        if i % 500 == 0:
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
    print(f"Dataset sẵn sàng tại: {output_dir}")

if __name__ == "__main__":
    main()
