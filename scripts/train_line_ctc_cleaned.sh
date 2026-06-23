#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/vietanh/anaconda3/envs/myenv/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/datasets/ocr}"
OUT_DIR="${OUT_DIR:-weights/ocr/small_lpr_line_ctc}"
RUN_NAME="${RUN_NAME:-line_ctc_cleaned_$(date +%Y%m%d_%H%M%S)}"

INIT_FROM="${INIT_FROM-weights/ocr/small_lpr_line_ctc/line_ctc_oneline16_lr3e4/small_lpr_line_ctc-epoch=021-val_acc=0.9407.ckpt}"
EXCLUDE_PATHS_FILE="${EXCLUDE_PATHS_FILE-data/datasets/ocr/exclude_train_pending_raw_review.txt}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0003}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
SEED="${SEED:-42}"
ACCUMULATE_GRAD="${ACCUMULATE_GRAD:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary is not executable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/valid" ]]; then
  echo "DATA_ROOT must contain train/ and valid/: $DATA_ROOT" >&2
  exit 2
fi

if [[ -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "INIT_FROM checkpoint does not exist: $INIT_FROM" >&2
  exit 2
fi

if [[ -n "$EXCLUDE_PATHS_FILE" && ! -f "$EXCLUDE_PATHS_FILE" ]]; then
  echo "EXCLUDE_PATHS_FILE does not exist: $EXCLUDE_PATHS_FILE" >&2
  exit 2
fi

cmd=(
  "$PYTHON_BIN" scripts/train_small_lpr_line_ctc.py
  --data-root "$DATA_ROOT"
  --out-dir "$OUT_DIR"
  --run-name "$RUN_NAME"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --devices "$DEVICES"
  --precision "$PRECISION"
  --seed "$SEED"
  --accumulate-grad "$ACCUMULATE_GRAD"
)

if [[ -n "$INIT_FROM" ]]; then
  cmd+=(--init-from "$INIT_FROM")
fi

if [[ -n "$EXCLUDE_PATHS_FILE" ]]; then
  cmd+=(--exclude-paths-file "$EXCLUDE_PATHS_FILE")
fi

cmd+=("$@")

printf 'Running cleaned Line-CTC training:\n'
printf '  run_name=%s\n' "$RUN_NAME"
printf '  init_from=%s\n' "${INIT_FROM:-<scratch>}"
printf '  exclude_paths_file=%s\n' "${EXCLUDE_PATHS_FILE:-<none>}"
printf '  lr=%s batch_size=%s epochs=%s devices=%s precision=%s\n' \
  "$LR" "$BATCH_SIZE" "$EPOCHS" "$DEVICES" "$PRECISION"
printf '\nCommand:\n'
printf ' %q' "${cmd[@]}"
printf '\n\n'

exec "${cmd[@]}"
