#!/usr/bin/env bash
# Stop 3DAIGC-API scheduler + uvicorn workers cleanly (avoids duplicate schedulers).
#
# Usage:
#   bash scripts/stop_services.sh          # graceful (drain in-flight jobs first)
#   bash scripts/stop_services.sh --force  # immediate SIGKILL, no drain
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

PID_DIR="$ROOT/run"
SCHEDULER_PID_FILE="$PID_DIR/scheduler.pid"
API_PID_FILE="$PID_DIR/api.pid"
REDIS_URL="${P3D_REDIS_URL:-redis://localhost:6379}"

wait_for_jobs_to_finish() {
  if [[ "$FORCE" -eq 1 ]]; then
    return 0
  fi
  if [[ "${P3D_DRAIN_JOBS_ON_SHUTDOWN:-1}" != "1" ]]; then
    return 0
  fi

  local redis_cli=()
  if command -v redis-cli &>/dev/null; then
    redis_cli=(redis-cli -u "$REDIS_URL")
  elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx '3daigc-redis'; then
    redis_cli=(docker exec 3daigc-redis redis-cli)
  else
    echo "   (redis unavailable; skipping job drain)"
    return 0
  fi

  local processing
  processing="$("${redis_cli[@]}" SCARD 3daigc:queue:processing 2>/dev/null || echo 0)"
  if [[ "${processing:-0}" -eq 0 ]]; then
    return 0
  fi

  local max_wait="${P3D_SHUTDOWN_DRAIN_SEC:-300}"
  local poll="${P3D_SHUTDOWN_POLL_SEC:-5}"
  echo "   Waiting for ${processing} in-flight GPU job(s) (max ${max_wait}s)..."
  local elapsed=0
  while [[ "$elapsed" -lt "$max_wait" ]]; do
    processing="$("${redis_cli[@]}" SCARD 3daigc:queue:processing 2>/dev/null || echo 0)"
    if [[ "${processing:-0}" -eq 0 ]]; then
      echo "   All in-flight jobs finished."
      return 0
    fi
    sleep "$poll"
    elapsed=$((elapsed + poll))
  done
  echo "   Drain timeout — stopping services anyway."
}

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  local grace="${3:-5}"

  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    return 0
  fi

  echo "   Stopping ${label} (pid ${pid})..."
  if [[ "$FORCE" -eq 1 ]]; then
    kill -9 "$pid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [[ "$waited" -lt "$grace" ]]; do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
}

echo "Stopping 3DAIGC-API services..."
wait_for_jobs_to_finish

# Scheduler first so it tears down GPU workers gracefully.
stop_pid_file "scheduler" "$SCHEDULER_PID_FILE" 15
stop_pid_file "API" "$API_PID_FILE" 5

# Remove duplicate/stale processes left from manual restarts.
if [[ "$FORCE" -eq 1 ]]; then
  pkill -9 -f "$ROOT/scripts/scheduler_service.py" 2>/dev/null || true
  pkill -9 -f "uvicorn api.main_multiworker:app" 2>/dev/null || true
else
  pkill -TERM -f "$ROOT/scripts/scheduler_service.py" 2>/dev/null || true
  sleep 2
  pkill -9 -f "$ROOT/scripts/scheduler_service.py" 2>/dev/null || true
  pkill -TERM -f "uvicorn api.main_multiworker:app" 2>/dev/null || true
  sleep 2
  pkill -9 -f "uvicorn api.main_multiworker:app" 2>/dev/null || true
fi

# Orphaned model worker subprocesses (scheduler gone but CUDA workers remain).
sleep 1
pkill -9 -f "$ROOT/venv/bin/python -c from multiprocessing.spawn import spawn_main" 2>/dev/null || true
pkill -9 -f "$ROOT/venv/bin/python3 -c from multiprocessing.spawn import spawn_main" 2>/dev/null || true

remaining_sched=0
remaining_api=0
remaining_sched="$(pgrep -fc "$ROOT/scripts/scheduler_service.py" 2>/dev/null)" || remaining_sched=0
remaining_api="$(pgrep -fc 'uvicorn api.main_multiworker:app' 2>/dev/null)" || remaining_api=0
echo "Done. Remaining schedulers: ${remaining_sched}, API: ${remaining_api}"
