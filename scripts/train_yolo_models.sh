#!/usr/bin/env bash
set -euo pipefail

YOLO=/home/vietanh/anaconda3/envs/myenv/bin/yolo

"$YOLO" obb train model=weights/detection/best.pt data=configs/detection/lp_detection_obb_dataset.yaml epochs=50 patience=100 batch=8 imgsz=640 device=0 workers=8 project=runs/obb/experiments/detection name=lp_detection_obb_merged lr0=0.001 lrf=0.01 warmup_bias_lr=0.1 optimizer=auto pretrained=True seed=0 deterministic=True freeze=0

"$YOLO" classify train model=runs/classify/runs/classify/plate_quality_legibility4/weights/best.pt data=/home/vietanh/Documents/DATN/data/datasets/legibility_finetune epochs=50 patience=10 batch=16 imgsz=64 workers=8 project=runs/classify/runs/classify name=legibility_finetuned_vn lr0=0.0001 lrf=0.01 optimizer=auto pretrained=True seed=0 deterministic=True
