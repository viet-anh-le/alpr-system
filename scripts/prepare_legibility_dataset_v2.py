import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    base_dir = Path(__file__).resolve().parent.parent
    train_csv = base_dir / "runs/infer/quality_router_platesmania_vn_train/predictions.csv"
    val_csv = base_dir / "runs/infer/quality_router_platesmania_vn_val/predictions_val.csv"
    output_dir = base_dir / "data/datasets/legibility_finetune"
    
    print("Đọc các file predictions...")
    df_train = pd.read_csv(train_csv)
    df_train['split'] = 'train'
    
    df_val = pd.read_csv(val_csv)
    df_val['split'] = 'val'
    
    # Gộp chung để xử lý lọc top illegible
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    
    # Loại bỏ các dòng không có nhãn
    df_all = df_all.dropna(subset=['predicted_legibility'])
    
    # Tách các lớp sạch (good, perfect, poor)
    df_clean = df_all[df_all['predicted_legibility'].isin(['good', 'perfect', 'poor'])]
    
    # Lấy top 3600 ảnh illegible có router_conf cao nhất
    df_illegible = df_all[df_all['predicted_legibility'] == 'illegible']
    df_illegible_sorted = df_illegible.sort_values(by='router_conf', ascending=False)
    
    print(f"Tổng số ảnh illegible ban đầu: {len(df_illegible_sorted)}")
    df_illegible_top = df_illegible_sorted.head(3600)
    print(f"Sẽ giữ lại {len(df_illegible_top)} ảnh illegible có độ tin cậy cao nhất.")
    
    # Ghép lại dataset cuối cùng
    df_final = pd.concat([df_clean, df_illegible_top], ignore_index=True)
    
    # Thống kê trước khi copy
    print("\nThống kê số lượng theo từng lớp sẽ được đưa vào finetune:")
    print(df_final.groupby(['split', 'predicted_legibility']).size())
    print("\nBắt đầu copy ảnh...")
    
    count = 0
    missing = 0
    
    for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Copying files"):
        split = row['split']
        label = row['predicted_legibility']
        img_rel_path = row['path']
        src_path = base_dir / img_rel_path
        
        if not src_path.exists():
            missing += 1
            continue
            
        dst_dir = output_dir / split / str(label)
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        dst_path = dst_dir / src_path.name
        
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        count += 1
        
    print(f"\nHoàn tất! Đã copy {count} ảnh. Bỏ qua {missing} ảnh do không tìm thấy file gốc.")
    print(f"Dữ liệu finetune đã sẵn sàng tại: {output_dir}")

if __name__ == "__main__":
    main()
