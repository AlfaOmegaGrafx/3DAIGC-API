#!/usr/bin/env bash
# Start 3daigc-mcp HTTP server for XR AI worker integration (background).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${DAIGC_MCP_HTTP_PORT:-8260}"
PIDFILE="${ROOT}/.daigc-mcp-http.pid"
LOG="${ROOT}/logs/daigc-mcp-http.log"

mkdir -p "${ROOT}/logs"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "3daigc-mcp-http already running (pid $(cat "$PIDFILE"), port ${PORT})"
  exit 0
fi

export DAIGC_API_BASE_URL="${DAIGC_API_BASE_URL:-http://localhost:7842}"

nohup uv run 3daigc-mcp-http --host 0.0.0.0 --port "$PORT" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started 3daigc-mcp-http pid $(cat "$PIDFILE") on :${PORT} (log: ${LOG})"
