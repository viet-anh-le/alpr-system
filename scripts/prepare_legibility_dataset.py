import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def prepare_dataset(csv_path, split, base_dir, output_dir):
    """
    Đọc file predictions CSV và copy ảnh vào cấu trúc YOLO classification
    """
    df = pd.read_csv(csv_path)
    
    # Tạo thư mục cho từng nhãn trong split
    labels = df['predicted_legibility'].unique()
    for label in labels:
        if pd.isna(label):
            continue
        (output_dir / split / str(label)).mkdir(parents=True, exist_ok=True)
        
    count = 0
    missing = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        label = row['predicted_legibility']
        if pd.isna(label):
            continue
            
        img_rel_path = row['path']
        src_path = base_dir / img_rel_path
        
        if not src_path.exists():
            missing += 1
            continue
            
        dst_path = output_dir / split / str(label) / src_path.name
        
        # Chỉ copy nếu file chưa tồn tại để tiết kiệm thời gian nếu chạy lại
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        count += 1
        
    print(f"[{split}] Xử lý thành công: {count} file. Không tìm thấy: {missing} file.")

def main():
    # Thư mục gốc của project
    base_dir = Path(__file__).resolve().parent.parent
    
    # Thư mục chứa kết quả prediction
    train_csv = base_dir / "runs/infer/quality_router_platesmania_vn_train/predictions.csv"
    val_csv = base_dir / "runs/infer/quality_router_platesmania_vn_val/predictions_val.csv"
    
    # Thư mục đích cho dataset classification
    output_dir = base_dir / "data/datasets/legibility_finetune"
    
    print(f"Bắt đầu chuẩn bị dataset tại: {output_dir}")
    
    if not train_csv.exists():
        print(f"Lỗi: Không tìm thấy {train_csv}")
        return
        
    if not val_csv.exists():
        print(f"Lỗi: Không tìm thấy {val_csv}")
        return
        
    prepare_dataset(train_csv, "train", base_dir, output_dir)
    prepare_dataset(val_csv, "val", base_dir, output_dir)
    
    print("\nHoàn tất! Cấu trúc thư mục dataset:")
    for split in ["train", "val"]:
        split_dir = output_dir / split
        if split_dir.exists():
            print(f"- {split}/")
            for label_dir in sorted(split_dir.iterdir()):
                if label_dir.is_dir():
                    num_files = len(list(label_dir.glob("*.*")))
                    print(f"  - {label_dir.name}/ : {num_files} ảnh")

if __name__ == "__main__":
    main()
