#!/usr/bin/env bash
# Verify Redis + scheduler source continuity + API/scheduler processes + health endpoint.
#
# Usage:
#   bash scripts/verify_api_stack.sh
#   bash scripts/verify_api_stack.sh --smoke-kimodo   # submit short text-to-motion job
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SMOKE_KIMODO=0
for arg in "$@"; do
  case "$arg" in
    --smoke-kimodo) SMOKE_KIMODO=1 ;;
  esac
done

API_PORT="${P3D_PORT:-7842}"
API_BASE="http://127.0.0.1:${API_PORT}"

echo "=== ensure_redis ==="
bash "$ROOT/scripts/ensure_redis.sh"

echo "=== check_scheduler_continuity ==="
bash "$ROOT/scripts/check_scheduler_continuity.sh"

echo "=== process check ==="
sched_ok=0
api_ok=0
if [[ -f run/scheduler.pid ]] && kill -0 "$(cat run/scheduler.pid)" 2>/dev/null; then
  sched_ok=1
  echo "scheduler pid $(cat run/scheduler.pid) running"
elif pgrep -f "$ROOT/scripts/scheduler_service.py" >/dev/null 2>&1; then
  sched_ok=1
  echo "scheduler process running (untracked pid)"
else
  echo "WARN: scheduler not running"
fi

if [[ -f run/api.pid ]] && kill -0 "$(cat run/api.pid)" 2>/dev/null; then
  api_ok=1
  echo "API pid $(cat run/api.pid) running"
elif pgrep -f "uvicorn api.main_multiworker:app" >/dev/null 2>&1; then
  api_ok=1
  echo "API process running (untracked pid)"
else
  echo "WARN: API not running"
fi

echo "=== health ==="
health_ok=0
for _ in $(seq 1 30); do
  if curl -sf "${API_BASE}/health" | python3 -m json.tool >/dev/null 2>&1; then
    curl -sf "${API_BASE}/health" | python3 -m json.tool
    health_ok=1
    break
  fi
  sleep 1
done
if [[ "$health_ok" -ne 1 ]]; then
  echo "ERROR: ${API_BASE}/health failed after wait"
  exit 1
fi

if [[ "$sched_ok" -eq 0 ]]; then
  echo "ERROR: scheduler not running — run: bash scripts/restart_services.sh"
  exit 1
fi

if grep -q "Scheduler Service is READY" logs/scheduler.log 2>/dev/null; then
  echo "scheduler log: READY seen"
else
  echo "WARN: no READY line in logs/scheduler.log yet"
fi

if [[ "$SMOKE_KIMODO" -eq 1 ]]; then
  echo "=== kimodo smoke job ==="
  job_json="$(curl -sf -X POST "${API_BASE}/api/v1/motion-generation/text-to-motion" \
    -H 'Content-Type: application/json' \
    -d '{"text_prompt":"walking forward slowly","duration":3,"output_format":"studio_motion"}')"
  job_id="$(echo "$job_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")"
  echo "job_id=$job_id"
  for _ in $(seq 1 120); do
    sleep 5
    status="$(curl -sf "${API_BASE}/api/v1/system/jobs/${job_id}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")"
    echo "  status=$status"
    case "$status" in
      completed) echo "Kimodo smoke OK"; exit 0 ;;
      failed|cancelled)
        curl -sf "${API_BASE}/api/v1/system/jobs/${job_id}" | python3 -m json.tool
        exit 1
        ;;
    esac
  done
  echo "ERROR: Kimodo smoke timed out"
  exit 1
fi

echo "verify_api_stack OK"
