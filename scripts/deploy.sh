#!/usr/bin/env bash
# Manual CD for the ALPR backend. Pulls the deploy branch, gates it with a real
# import check, rebuilds the image, and rolls the API + worker onto it.
#
# Usage:  ./scripts/deploy.sh [branch]      (default branch: dockerize-backend)
#
# NOTE: this does `git reset --hard origin/<branch>`, so git is the source of
# truth — any uncommitted edits made directly on the server are discarded.
set -euo pipefail

BRANCH="${1:-dockerize-backend}"
cd "$(dirname "$0")/.." 2>/dev/null || cd /home/ubuntu/Vanh_workspace/alpr-system
VENV_PY=".venv/bin/python"

say() { printf "\n\033[1;36m[deploy] %s\033[0m\n" "$1"; }

say "Fetching origin/$BRANCH …"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
echo "  now at $(git rev-parse --short HEAD) — $(git log -1 --pretty=%s)"

say "Import gate (catches missing-import / bad-import bugs before deploying) …"
if [ -x "$VENV_PY" ]; then
  if ! "$VENV_PY" -c "import api.main; import api.worker" ; then
    echo "  ❌ import failed — ABORTING, running stack left untouched."
    exit 1
  fi
  echo "  ✓ imports OK"
else
  echo "  ⚠ $VENV_PY not found — skipping host import gate (CI ruff gate still applies)."
fi

say "Building image …"
docker compose build api

say "Rolling api + worker onto the new image …"
docker compose up -d api worker

say "Health check (waits for models to load) …"
ok=""
for i in $(seq 1 24); do        # up to ~2 min
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
if [ -n "$ok" ]; then
  echo "  ✓ /health OK"
  # readiness (Redis) + worker health, best-effort
  curl -fsS http://localhost:8000/ready >/dev/null 2>&1 && echo "  ✓ /ready OK" || echo "  ⚠ /ready not ready yet"
  docker compose ps --format '{{.Name}}: {{.Status}}' | grep -E "api|worker" || true
  say "Deploy complete."
else
  echo "  ❌ /health did not come up within timeout — check: docker compose logs api --tail 40"
  exit 1
fi
