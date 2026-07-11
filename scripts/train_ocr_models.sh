#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/vietanh/anaconda3/envs/myenv/bin/python}"
MODEL="${1:-both}"
DATA_ROOT="${DATA_ROOT:-data/datasets/ocr}"

case "$MODEL" in
  smalllpr|small_lpr)
    "$PYTHON_BIN" scripts/train_small_lpr.py \
      --data-root "$DATA_ROOT" \
      --out-dir weights/ocr/small_lpr \
      "${@:2}"
    ;;
  parseq)
    "$PYTHON_BIN" ocr/train_parseq.py \
      --data-root "$DATA_ROOT" \
      --out-dir weights/ocr/parseq \
      --charset "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ-.[]" \
      --max-label-length 25 \
      "${@:2}"
    ;;
  smalllpr_ctc|small_lpr_ctc)
    "$PYTHON_BIN" scripts/train_small_lpr_ctc.py \
      --data-root "$DATA_ROOT" \
      --out-dir weights/ocr/small_lpr_ctc \
      "${@:2}"
    ;;
  smalllpr_nar|small_lpr_nar)
    "$PYTHON_BIN" scripts/train_small_lpr_nar.py \
      --data-root "$DATA_ROOT" \
      --out-dir weights/ocr/small_lpr_nar \
      "${@:2}"
    ;;
  both)
    "$PYTHON_BIN" scripts/train_small_lpr.py \
      --data-root "$DATA_ROOT" \
      --out-dir weights/ocr/small_lpr
    "$PYTHON_BIN" ocr/train_parseq.py \
      --data-root "$DATA_ROOT" \
      --out-dir weights/ocr/parseq \
      --charset "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ-.[]" \
      --max-label-length 25
    ;;
  *)
    echo "Usage: $0 {smalllpr|smalllpr_ctc|smalllpr_nar|parseq|both} [extra args]" >&2
    exit 2
    ;;
esac
