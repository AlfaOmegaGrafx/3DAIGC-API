#!/usr/bin/env bash
# Fail fast before starting workers if scheduler modules are missing on disk.
# Parent scheduler can keep running after accidental deletions; worker subprocesses crash.
#
# Usage:
#   bash scripts/check_scheduler_continuity.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REQUIRED=(
  core/scheduler/job_queue.py
  core/scheduler/multiprocess_scheduler.py
  core/scheduler/redis_job_queue.py
  core/scheduler/database_manager.py
  core/scheduler/database_models.py
  core/utils/job_time.py
)

missing=0
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "ERROR: scheduler continuity check failed."
  echo "Restore: git checkout -- core/scheduler/ core/utils/job_time.py"
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$ROOT"
"$ROOT/venv/bin/python" -c "
from core.scheduler.job_queue import JobQueue, JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler
print('scheduler imports OK')
"
