#!/usr/bin/env bash
# ALPR stack status — one-shot operational overview of the docker-compose stack.
# Usage: ./scripts/status.sh
set -uo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || cd /home/ubuntu/Vanh_workspace/alpr-system
DC="docker compose"
hdr(){ printf "\n\033[1;36m== %s ==\033[0m\n" "$1"; }

hdr "Containers (want: all 'running (healthy)')"
$DC ps --format 'table {{.Name}}\t{{.State}}\t{{.Status}}'

hdr "Health endpoints"
printf "  /health : %s\n" "$(curl -s -m5 http://localhost:8000/health || echo UNREACHABLE)"
printf "  /ready  : %s\n" "$(curl -s -m5 http://localhost:8000/ready  || echo UNREACHABLE)"
printf "  public  : %s\n" "$(curl -s -m8 https://api.altperle.id.vn/health || echo UNREACHABLE)"

hdr "Queue & workers (Redis)"
$DC exec -T api python - <<'PY' 2>/dev/null || echo "  (api container not reachable)"
import asyncio
from api.core import jobstore
async def m():
    r = jobstore.get_redis()
    print("  queue depth (waiting + in-flight):", await jobstore.queue_depth())
    try:
        p = await r.xpending(jobstore.QUEUE_STREAM, jobstore.CONSUMER_GROUP)
        print("  in-flight (unacked):", p.get("pending") if isinstance(p, dict) else p)
        for c in await r.xinfo_consumers(jobstore.QUEUE_STREAM, jobstore.CONSUMER_GROUP):
            print(f"    worker {c['name']}: processing={c['pending']} idle={c['idle']}ms")
    except Exception:
        print("  consumer group not initialised yet")
    await jobstore.close_redis()
asyncio.run(m())
PY

hdr "GPU"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
echo "  processes:"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader

hdr "Resource usage (snapshot)"
ids=$($DC ps -q)
[ -n "$ids" ] && docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' $ids

hdr "Recent errors (last 20m)"
errs=$($DC logs --since 20m 2>&1 | grep -iE "traceback|exception|[^a-z]error[^a-z]|cuda error|out of memory|oom-kill|killed process|failed to" | grep -viE "device: cuda|models ready" | tail -8)
[ -n "$errs" ] && echo "$errs" || echo "  none"
echo
