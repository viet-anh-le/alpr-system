#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/detection/yolov8n.yaml}
python detection/train.py --config "$CONFIG"
