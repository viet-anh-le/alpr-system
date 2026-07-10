import cv2
import os
import sys
import numpy as np
import torch
import gc
import yaml
from argparse import Namespace
from tqdm.notebook import tqdm
from ultralytics import YOLO

sys.path.append(os.path.abspath("LPRNet"))
from lprnet import SmallLPR, numpy2tensor, decode

torch.serialization.add_safe_globals([Namespace])

with open("config/small_lpr_config.yaml") as f:
    lpr_args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

lpr_args.pretrained = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/LPRNet_Claude/saving_ckpt-smalllpr/2026-04-19_19-35/small_lpr-epoch=128-val_acc=0.891.ckpt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

lprnet_model = SmallLPR(
    vocab_size=len(lpr_args.chars),
    max_seq_len=lpr_args.max_seq_len,
    use_pretrained_decoder=False
).to(DEVICE).eval()
ckpt = torch.load(lpr_args.pretrained, map_location=DEVICE)
if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_k] = v
    lprnet_model.load_state_dict(new_state_dict)
else:
    lprnet_model.load_state_dict(ckpt)
print("=> Tải mô hình LPRNet thành công!")

vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO(
    "/home/vietanh/Documents/DATN/ALPR_Vietnamese/datasets/archive/runs/obb/train/weights/best.pt"
)
VEHICLE_CLASSES = [2, 3, 5, 7]

input_dir = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/datasets/realworld-videos/chunks"
output_dir = (
    "/home/vietanh/Documents/DATN/ALPR_Vietnamese/datasets/realworld-videos/inference_results"
)
os.makedirs(output_dir, exist_ok=True)


def process_video_lprnet():
    for video_name in sorted(os.listdir(input_dir)):
        if video_name == "đoạn_002.mp4":
            if not video_name.endswith((".mp4", ".avi")):
                continue

            video_path = os.path.join(input_dir, video_name)
            out_path = os.path.join(output_dir, f"out_lprnet_{video_name}")

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            print(f"Đang xử lý bằng LPRNet: {video_name}...")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            pbar = tqdm(total=total_frames, desc="Inferencing LPRNet", unit="frame")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"\n-> OpenCV dừng đọc ở frame {frame_count}/{total_frames}")
                    break

                frame_count += 1

                v_results = vehicle_model.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    classes=VEHICLE_CLASSES,
                    verbose=False,
                )[0]

                p_results = plate_model(frame, verbose=False)[0]

                tracked_vehicles = []
                if v_results.boxes.id is not None:
                    boxes = v_results.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = v_results.boxes.id.cpu().numpy().astype(int)
                    class_ids = v_results.boxes.cls.cpu().numpy().astype(int)

                    for box, t_id, c_id in zip(boxes, track_ids, class_ids):
                        x1, y1, x2, y2 = box
                        class_name = vehicle_model.names[c_id]
                        tracked_vehicles.append(
                            {"id": t_id, "class": class_name, "box": (x1, y1, x2, y2)}
                        )
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        label = f"{class_name} ID: {t_id}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2,
                        )

                if p_results.obb is not None:
                    obb_points = p_results.obb.xyxyxyxy.cpu().numpy().astype(int)
                    for pts in obb_points:
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))

                        matched_vehicle_id = "Unknown"
                        for v in tracked_vehicles:
                            vx1, vy1, vx2, vy2 = v["box"]
                            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                                matched_vehicle_id = v["id"]
                                break

                        if matched_vehicle_id == "Unknown":
                            continue

                        rect = cv2.boundingRect(pts)
                        px, py, pw, ph = rect
                        plate_crop = frame[max(0, py) : py + ph, max(0, px) : px + pw]

                        pred_text = ""
                        if plate_crop.size > 0:
                            im = numpy2tensor(plate_crop, lpr_args.img_size).unsqueeze(0).to(DEVICE)

                            with torch.no_grad():
                                tokens = lprnet_model(im)[0].detach().cpu().numpy()
                                decoded_chars = []
                                for c in tokens:
                                    c = int(c)
                                    if c == 2:
                                        break
                                    if c not in [0, 1]:
                                        decoded_chars.append(lpr_args.chars[c])
                                pred_text = "".join(decoded_chars)

                        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                        label = (
                            f"{pred_text} -> Car {matched_vehicle_id}"
                            if pred_text
                            else f"LP -> Car {matched_vehicle_id}"
                        )
                        cv2.putText(
                            frame,
                            label,
                            (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                out.write(frame)
                pbar.update(1)

                if frame_count % 30 == 0:
                    del v_results, p_results
                    gc.collect()
                    torch.cuda.empty_cache()

            pbar.close()
            cap.release()
            out.release()
            break

    print("\nHoàn thành inference Video bằng LPRNet!")


if __name__ == "__main__":
    process_video_lprnet()
