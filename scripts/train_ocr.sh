#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/ocr/lprnet.yaml}
python ocr/train/train_ocr.py --config "$CONFIG"
