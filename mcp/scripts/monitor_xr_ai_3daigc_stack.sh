#!/usr/bin/env bash
# Tail the latest 3daigc-vlm-example per-run log folder (orchestrator + all services).
set -euo pipefail

LOG_ROOT="${1:-}"
if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="$(ls -td /tmp/log_3daigc-vlm-example_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "${LOG_ROOT}" || ! -d "${LOG_ROOT}" ]]; then
  echo "No /tmp/log_3daigc-vlm-example_* directory yet. Start the stack first:" >&2
  echo "  bash /home/sifr/3DAIGC-API/mcp/scripts/run_xr_ai_3daigc_stack.sh" >&2
  exit 1
fi

echo "Monitoring: ${LOG_ROOT}"
echo "Press Ctrl-C to stop tail (stack keeps running)."
exec tail -F "${LOG_ROOT}"/*.log
