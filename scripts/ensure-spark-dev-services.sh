#!/usr/bin/env bash
# Start DGX dev services after reboot (API + LingBot check + optional MSF + optional XR voice).
#
# Usage:
#   bash scripts/ensure-spark-dev-services.sh
#   bash scripts/ensure-spark-dev-services.sh --with-xr
#   bash scripts/ensure-spark-dev-services.sh --with-routing
#   bash scripts/ensure-spark-dev-services.sh --with-routing funnel
#   bash scripts/ensure-spark-dev-services.sh --api-only
#   bash scripts/ensure-spark-dev-services.sh --force
#   bash scripts/ensure-spark-dev-services.sh --skip-lingbot
#
# After start, run: bash scripts/verify-spark-dev-stack.sh [--with-xr]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MSF_ROOT="${MSF_MAP_SVC_ROOT:-/home/sifr/MSF_Map_Svc}"
XR_AI_ROOT="${XR_AI_ROOT:-/home/sifr/xr-ai}"
RP1_ENV="${RP1_ENV_FILE:-/home/sifr/.config/rp1-spatial-fabric/rp1.env}"

API_ONLY=0
WITH_XR=0
WITH_ROUTING=0
ROUTING_MODE="serve"
FORCE=0
SKIP_LINGBOT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-only)     API_ONLY=1 ;;
    --with-xr)      WITH_XR=1 ;;
    --with-routing) WITH_ROUTING=1 ;;
    --force)        FORCE=1 ;;
    --skip-lingbot) SKIP_LINGBOT=1 ;;
    serve|funnel)   ROUTING_MODE="$1" ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
  esac
  shift
done

echo "=============================================="
echo " DGX post-reboot: LingBot-Map (in-process)"
echo "=============================================="
cd "$ROOT"
if [[ "$SKIP_LINGBOT" -eq 1 ]]; then
  echo "Skipping LingBot check (--skip-lingbot)."
else
  # Install if missing; LingBot rides on :7842 workers (not a separate port).
  bash "$ROOT/scripts/ensure_lingbot_map.sh" --install
fi

echo ""
echo "=============================================="
echo " DGX post-reboot: 3DAIGC-API"
echo "=============================================="
restart_args=()
if [[ "$FORCE" -eq 1 ]]; then
  restart_args+=(--force)
fi
bash "$ROOT/scripts/restart_services.sh" "${restart_args[@]}"

if [[ -f "$RP1_ENV" ]]; then
  echo ""
  echo "==> Sync spatial-fabric env → 3DAIGC-API .env"
  bash "$ROOT/scripts/sync-spatial-fabric-env.sh" || true
fi

if [[ "$API_ONLY" -eq 1 ]]; then
  echo ""
  echo "API-only mode — skipping MSF and XR."
  echo "Verify: bash $ROOT/scripts/verify-spark-dev-stack.sh"
  exit 0
fi

if [[ -d "$MSF_ROOT/dist" && -f "$RP1_ENV" ]]; then
  echo ""
  echo "=============================================="
  echo " DGX post-reboot: MSF Map Service (:8443)"
  echo "=============================================="
  bash "$MSF_ROOT/scripts/run-msf-map-svc.sh"
else
  echo ""
  echo "Skipping MSF (missing $MSF_ROOT/dist or $RP1_ENV)."
fi

if [[ "$WITH_XR" -eq 1 ]]; then
  echo ""
  echo "=============================================="
  echo " DGX post-reboot: XR voice hub (:8088)"
  echo "=============================================="
  if [[ -d "$XR_AI_ROOT/agent-samples/3daigc-vlm-example" ]]; then
    bash "$ROOT/mcp/scripts/start_prerequisites.sh"
    LOG="$ROOT/logs/xr-ai-stack.log"
    PID_FILE="$ROOT/.xr-ai-stack.pid"
    mkdir -p "$ROOT/logs"
    hub_code() {
      curl -sk --connect-timeout 5 -o /dev/null -w '%{http_code}' "https://127.0.0.1:8088/" 2>/dev/null || echo "000"
    }
    if pgrep -f 'daigc_vlm_example' >/dev/null 2>&1 && [[ "$(hub_code)" == "200" ]]; then
      echo "XR voice stack already running (hub HTTP 200)"
    else
      if pgrep -f 'daigc_vlm_example' >/dev/null 2>&1; then
        pkill -f 'daigc_vlm_example' 2>/dev/null || true
        sleep 2
      fi
      nohup bash "$ROOT/mcp/scripts/run_xr_ai_3daigc_stack.sh" >>"$LOG" 2>&1 &
      echo $! >"$PID_FILE"
      echo "Started XR stack (pid $(cat "$PID_FILE"), log $LOG)"
      for _ in $(seq 1 36); do
        if [[ "$(hub_code)" == "200" ]]; then
          echo "XR hub ready: https://127.0.0.1:8088/"
          break
        fi
        sleep 5
      done
    fi
    echo "Surface proxy (run on PC): cd OpenNexus3DStudio && npm run dev:spark-proxies"
  else
    echo "WARN: xr-ai sample missing at $XR_AI_ROOT — skip --with-xr"
  fi
fi

if [[ "$WITH_ROUTING" -eq 1 ]]; then
  echo ""
  echo "=============================================="
  echo " DGX post-reboot: Tailscale routing ($ROUTING_MODE)"
  echo "=============================================="
  if [[ -x "$MSF_ROOT/scripts/setup-dgx-public-routing.sh" ]]; then
    bash "$MSF_ROOT/scripts/setup-dgx-public-routing.sh" "$ROUTING_MODE"
  else
    echo "WARN: $MSF_ROOT/scripts/setup-dgx-public-routing.sh not found"
  fi
fi

echo ""
echo "ensure-spark-dev-services done."
echo "Verify: bash $ROOT/scripts/verify-spark-dev-stack.sh$([[ $WITH_XR -eq 1 ]] && echo ' --with-xr')"
