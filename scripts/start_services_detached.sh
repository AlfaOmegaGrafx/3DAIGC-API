#!/usr/bin/env bash
# Start API + scheduler in the background so jobs survive terminal disconnect.
# Logs: logs/api.log, logs/scheduler.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs run

if [[ -f run/api.pid ]] && kill -0 "$(cat run/api.pid)" 2>/dev/null; then
  echo "API already running (pid $(cat run/api.pid)). Stop it first or use run_server.sh in foreground."
  exit 1
fi

# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$ROOT"
export P3D_DRAIN_JOBS_ON_SHUTDOWN=1
export P3D_SHUTDOWN_DRAIN_SEC="${P3D_SHUTDOWN_DRAIN_SEC:-7200}"

nohup "$ROOT/venv/bin/python" scripts/scheduler_service.py \
  --redis-url "${P3D_REDIS_URL:-redis://localhost:6379}" \
  >> logs/scheduler.log 2>&1 &
echo $! > run/scheduler.pid
echo "Scheduler started (pid $(cat run/scheduler.pid))"

sleep 3
nohup "$ROOT/venv/bin/uvicorn" api.main_multiworker:app \
  --host "${P3D_HOST:-0.0.0.0}" \
  --port "${P3D_PORT:-7842}" \
  --workers "${P3D_WORKERS:-4}" \
  >> logs/api.log 2>&1 &
echo $! > run/api.pid
echo "API started (pid $(cat run/api.pid)) on port ${P3D_PORT:-7842}"
echo "Tail logs: tail -f logs/scheduler.log logs/api.log"
