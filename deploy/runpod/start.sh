#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[runpod-start] %s\n' "$*"
}

export PORT="${PORT:-8000}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export ALPR_UPLOAD_DIR="${ALPR_UPLOAD_DIR:-/workspace/alpr/uploads}"
export ALPR_PREPROCESSED_VIDEO_DIR="${ALPR_PREPROCESSED_VIDEO_DIR:-${ALPR_UPLOAD_DIR}/preprocessed}"
export MONITOR_UPLOAD_DIR="${MONITOR_UPLOAD_DIR:-${ALPR_UPLOAD_DIR}/monitor}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export YOLOV5_CONFIG_DIR="${YOLOV5_CONFIG_DIR:-/tmp/ultralytics}"
export ALPR_API_LOAD_MODELS="${ALPR_API_LOAD_MODELS:-true}"
export WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-1}"

START_REDIS="${START_REDIS:-true}"
REDIS_DATA_DIR="${REDIS_DATA_DIR:-/workspace/alpr/redis}"
MEDIAMTX_CONFIG="${MEDIAMTX_CONFIG:-/app/configs/mediamtx.yml}"

mkdir -p \
  "${ALPR_UPLOAD_DIR}" \
  "${ALPR_PREPROCESSED_VIDEO_DIR}" \
  "${MONITOR_UPLOAD_DIR}" \
  "${MPLCONFIGDIR}" \
  "${YOLOV5_CONFIG_DIR}" \
  "${REDIS_DATA_DIR}"

is_truthy() {
  [[ "$1" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]|[Oo][Nn])$ ]]
}

PIDS=()

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if ((${#PIDS[@]} > 0)); then
    log "stopping child processes"
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi

  exit "${status}"
}
trap cleanup EXIT INT TERM

start_background() {
  local name="$1"
  shift

  log "starting ${name}: $*"
  "$@" &
  PIDS+=("$!")
}

wait_for_redis() {
  log "waiting for Redis at ${REDIS_URL}"
  python - <<'PY'
import os
import sys
import time

import redis

url = os.environ["REDIS_URL"]
deadline = time.time() + float(os.environ.get("REDIS_STARTUP_TIMEOUT_SEC", "30"))
last_error = None

while time.time() < deadline:
    try:
        redis.Redis.from_url(url).ping()
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - startup diagnostics only.
        last_error = exc
        time.sleep(0.5)

print(f"Redis is not reachable at {url}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

if is_truthy "${START_REDIS}"; then
  start_background \
    "redis" \
    redis-server \
    --bind 127.0.0.1 \
    --port 6379 \
    --protected-mode yes \
    --dir "${REDIS_DATA_DIR}" \
    --appendonly yes \
    --save 60 1
fi

wait_for_redis

if [[ -f "${MEDIAMTX_CONFIG}" ]]; then
  start_background "mediamtx" mediamtx "${MEDIAMTX_CONFIG}"
else
  log "MediaMTX config not found at ${MEDIAMTX_CONFIG}; using built-in defaults"
  start_background "mediamtx" mediamtx
fi

start_background "worker" python -m api.worker
start_background "api" uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" --workers 1

log "all processes started"
set +e
wait -n "${PIDS[@]}"
STATUS=$?
set -e

log "a child process exited with status ${STATUS}; stopping pod"
if [[ "${STATUS}" -eq 0 ]]; then
  exit 1
fi
exit "${STATUS}"
