#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=8000}"

mkdir -p "${MONITOR_UPLOAD_DIR:-/tmp/alpr_monitor_uploads}"
mkdir -p "${MPLCONFIGDIR:-/tmp/matplotlib}"

mediamtx /app/configs/mediamtx.yml &
MEDIAMTX_PID=$!

cleanup() {
  kill "${MEDIAMTX_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
