import cv2
import torch
import numpy as np
from pathlib import Path

# Thêm đường dẫn project vào sys.path để import
import sys
sys.path.append("/home/vietanh/Documents/DATN/ALPR_Vietnamese")

from api.core.models import load_models
from api.core.video_processor import draw_annotated_frame, crop_vehicle, warp_plate_crop
from api.core.config import VEHICLE_CLASSES, PLATE_DET_CONF
from api.core.cascade_plate import crop_vehicle_regions

def main():
    print("Loading models...")
    models = load_models()
    vehicle_tracker = models.create_vehicle_tracker()
    
    video_path = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/realworld-videos/chunks/đoạn_018.mp4"
    out_dir = Path("/home/vietanh/Documents/DATN/ALPR_Vietnamese/prepare")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    # Lấy 3 frame cuối cùng của giây thứ 2
    skip_frames = int(fps * 2) - 3
    for _ in range(skip_frames):
        cap.read()
        
    frames = []
    for _ in range(3):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            
    cap.release()
    
    # 1. Các khung hình input
    for i, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"01_frame_input_{i+1}.jpg"), frame)
    print("Saved 01_frame_input_*.jpg")

    frame = frames[-1].copy() # Chọn frame thứ 3 để demo
    
    # 2. Vehicle Detection
    v_pred = models.vehicle.predict(frame, classes=VEHICLE_CLASSES, verbose=False)[0]
    if v_pred.boxes is not None and len(v_pred.boxes) > 0:
        xyxy = v_pred.boxes.xyxy.cpu().numpy()
        conf = v_pred.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = v_pred.boxes.cls.cpu().numpy().reshape(-1, 1)
        dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
    else:
        dets = np.zeros((0, 6), dtype=np.float32)
        
    det_img = frame.copy()
    for d in dets:
        x1, y1, x2, y2, cnf, cl = d
        cv2.rectangle(det_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(det_img, f"{models.vehicle.names[int(cl)]} {cnf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite(str(out_dir / "02_vehicle_detection.jpg"), det_img)
    print("Saved 02_vehicle_detection.jpg")
    
    # 3. Vehicle Tracking
    # Phải cho tracker chạy qua các frame trước đó để khởi tạo
    for f in frames[:-1]:
        v_pred_tmp = models.vehicle.predict(f, classes=VEHICLE_CLASSES, verbose=False)[0]
        if v_pred_tmp.boxes is not None and len(v_pred_tmp.boxes) > 0:
            d_tmp = np.concatenate([v_pred_tmp.boxes.xyxy.cpu().numpy(), v_pred_tmp.boxes.conf.cpu().numpy().reshape(-1, 1), v_pred_tmp.boxes.cls.cpu().numpy().reshape(-1, 1)], axis=1).astype(np.float32)
        else:
            d_tmp = np.zeros((0, 6), dtype=np.float32)
        vehicle_tracker.track(d_tmp, f)
    
    boxes, ids, classes = vehicle_tracker.track(dets, frame)
    
    box_dicts = []
    tracked = []
    for box, tid, cid in zip(boxes, ids, classes):
        tid = int(tid)
        tracked.append({"id": tid, "box": box.tolist()})
        box_dicts.append({
            "id": tid,
            "box": [int(c) for c in box],
            "state": "tracked",
            "cls": models.vehicle.names[int(cid)]
        })
        
    trk_img_bytes = draw_annotated_frame(frame.copy(), box_dicts)
    with open(str(out_dir / "03_vehicle_tracking.jpg"), "wb") as f:
        f.write(trk_img_bytes)
    print("Saved 03_vehicle_tracking.jpg")
    
    # 4. Crop vùng phương tiện và Plate Detection
    plate_found = False
    for i, target_vehicle in enumerate(tracked):
        if target_vehicle["id"] != 11:
            continue
        vehicle_crops = crop_vehicle_regions(frame, [target_vehicle])
        if not vehicle_crops:
            continue
        vcrop = vehicle_crops[0]
        
        use_half = torch.cuda.is_available()
        results = models.plate.predict([vcrop.image], verbose=False, half=use_half)
        result = results[0]
        
        if result.obb is not None and result.obb.xyxyxyxy is not None:
            pts_list = result.obb.xyxyxyxy.cpu().numpy().astype(np.float32)
            confs = result.obb.conf.cpu().numpy()
            
            best_pts = None
            best_conf = 0
            for pts, cf in zip(pts_list, confs):
                if cf > PLATE_DET_CONF and cf > best_conf:
                    best_conf = cf
                    best_pts = pts
                    
            if best_pts is not None:
                cv2.imwrite(str(out_dir / "04_vehicle_crop.jpg"), vcrop.image)
                print("Saved 04_vehicle_crop.jpg")
                
                obb_img = vcrop.image.copy()
                int_pts = best_pts.astype(np.int32)
                cv2.polylines(obb_img, [int_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imwrite(str(out_dir / "05_plate_detection_obb.jpg"), obb_img)
                print("Saved 05_plate_detection_obb.jpg")
                
                rx, ry, rw, rh = cv2.boundingRect(int_pts)
                rx = max(0, rx)
                ry = max(0, ry)
                rw = min(obb_img.shape[1] - rx, rw)
                rh = min(obb_img.shape[0] - ry, rh)
                tilted_crop = vcrop.image[ry:ry+rh, rx:rx+rw]
                if tilted_crop.size > 0:
                    cv2.imwrite(str(out_dir / "06_plate_crop_tilted.jpg"), tilted_crop)
                    print("Saved 06_plate_crop_tilted.jpg")
                    
                global_pts = best_pts.copy()
                global_pts[:, 0] += vcrop.offset[0]
                global_pts[:, 1] += vcrop.offset[1]
                
                warped_plate = warp_plate_crop(frame, global_pts)
                if warped_plate.size > 0:
                    cv2.imwrite(str(out_dir / "07_plate_crop_warped.jpg"), warped_plate)
                    print("Saved 07_plate_crop_warped.jpg")
                    
                    # Create Quality Router versions
                    cv2.imwrite(str(out_dir / "08_router_direct.jpg"), warped_plate)
                    fusion_img = cv2.GaussianBlur(warped_plate, (5, 5), 0)
                    cv2.imwrite(str(out_dir / "08_router_fusion.jpg"), fusion_img)
                    wait_img = cv2.GaussianBlur(warped_plate, (15, 15), 0)
                    cv2.imwrite(str(out_dir / "08_router_wait.jpg"), wait_img)
                    print("Saved 08_router_*.jpg")
                
                plate_found = True
                break
                
    if not plate_found:
        print("No plate found in any vehicle crop.")

if __name__ == "__main__":
    main()
