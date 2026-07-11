#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:-data/raw/sample.mp4}
python pipeline/inference_pipeline.py --source "$VIDEO"
