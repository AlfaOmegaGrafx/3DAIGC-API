#!/usr/bin/env bash
# Start XR voice end-to-end: DGX hub (:8088) + Surface proxy (:8443).
# Galaxy XR URL: https://10.0.0.32:8443  (NOT bare 10.0.0.32, NOT http)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MCP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/logs/xr-ai-stack.log"
PID_FILE="${ROOT}/.xr-ai-stack.pid"
DGX_IP="${DGX_IP:-10.0.0.158}"

mkdir -p "${ROOT}/logs"

hub_code() {
  curl -sk --connect-timeout 5 -o /dev/null -w '%{http_code}' "https://127.0.0.1:8088/" 2>/dev/null || echo "000"
}

echo "==> [1/3] 3DAIGC prerequisites (API :7842, MCP :8260)"
bash "${MCP_ROOT}/scripts/start_prerequisites.sh"

echo ""
echo "==> [2/3] DGX xr-ai voice stack (:8088)"
if pgrep -f 'daigc_vlm_example' >/dev/null 2>&1 && [[ "$(hub_code)" == "200" ]]; then
  echo "    OK  xr-ai orchestrator already running (hub HTTP 200)"
else
  if pgrep -f 'daigc_vlm_example' >/dev/null 2>&1; then
    echo "    Stale orchestrator — restarting"
    pkill -f 'daigc_vlm_example' 2>/dev/null || true
    sleep 2
  fi
  nohup bash "${MCP_ROOT}/scripts/run_xr_ai_3daigc_stack.sh" >>"$LOG" 2>&1 &
  echo $! >"$PID_FILE"
  echo "    Started background stack (pid $(cat "$PID_FILE"), log ${LOG})"
  for _ in $(seq 1 60); do
    if [[ "$(hub_code)" == "200" ]]; then
      echo "    OK  DGX hub https://${DGX_IP}:8088 (HTTP 200)"
      break
    fi
    sleep 5
  done
  if [[ "$(hub_code)" != "200" ]]; then
    echo "FAIL: DGX hub not ready — tail ${LOG}" >&2
    exit 1
  fi
fi

echo ""
echo "==> [3/3] Surface xr-hub-proxy (:8443 → DGX :8088)"
bash "${MCP_ROOT}/scripts/ensure_surface_xr_hub_proxy.sh"

echo ""
echo "XR_VOICE_READY"
echo "  Galaxy XR / headset: https://10.0.0.32:8443"
echo "  DGX direct (LAN only): https://${DGX_IP}:8088"
echo "  Logs: tail -f ${LOG}  or  bash ${MCP_ROOT}/scripts/monitor_xr_ai_3daigc_stack.sh"
