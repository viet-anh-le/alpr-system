#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/tracking/deepsort.yaml}
python tracking/train/train_reid.py --config "$CONFIG"
