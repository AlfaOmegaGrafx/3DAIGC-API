#!/usr/bin/env bash
# Verify DGX dev stack after reboot (API + MSF + optional XR).
#
# Usage:
#   bash scripts/verify-spark-dev-stack.sh
#   bash scripts/verify-spark-dev-stack.sh --with-xr
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MSF_ROOT="${MSF_MAP_SVC_ROOT:-/home/sifr/MSF_Map_Svc}"
RP1_ENV="${RP1_ENV_FILE:-/home/sifr/.config/rp1-spatial-fabric/rp1.env}"
WITH_XR=0

for arg in "$@"; do
  case "$arg" in
    --with-xr) WITH_XR=1 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
  esac
done

echo "=== 3DAIGC-API ==="
bash "$ROOT/scripts/verify_api_stack.sh"

echo ""
echo "=== LingBot-Map (environment-scan) ==="
bash "$ROOT/scripts/ensure_lingbot_map.sh"
# Confirm feature is registered on the live API when models endpoint works.
if curl -sf "http://127.0.0.1:${P3D_PORT:-7842}/api/v1/system/features" >/dev/null 2>&1; then
  if curl -sf "http://127.0.0.1:${P3D_PORT:-7842}/api/v1/system/features" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); feats=d.get('features') or d; names=feats if isinstance(feats,list) else list(feats.keys()) if isinstance(feats,dict) else []; assert any('environment_scan' in str(x) for x in names), names"; then
    echo "API feature environment_scan registered"
  else
    echo "WARN: environment_scan not listed in /system/features — check config/models.yaml"
  fi
fi

if [[ -d "$MSF_ROOT/dist" && -f "$RP1_ENV" ]]; then
  echo ""
  echo "=== MSF Map Service ==="
  bash "$MSF_ROOT/scripts/verify-fabric-url.sh"
fi

if [[ "$WITH_XR" -eq 1 ]]; then
  echo ""
  echo "=== XR voice hub (:8088) ==="
  code="$(curl -sk --connect-timeout 8 -o /dev/null -w '%{http_code}' "https://127.0.0.1:8088/" 2>/dev/null || echo "000")"
  if [[ "$code" == "200" ]]; then
    echo "XR hub OK (HTTP 200)"
  else
    echo "ERROR: XR hub not ready (HTTP $code) — run: bash $ROOT/scripts/ensure-spark-dev-services.sh --with-xr"
    exit 1
  fi
  MCP_PORT="${DAIGC_MCP_HTTP_PORT:-8260}"
  if curl -sf "http://127.0.0.1:${MCP_PORT}/health" >/dev/null 2>&1; then
    echo "3daigc-mcp-http OK (:${MCP_PORT})"
  else
    echo "WARN: 3daigc-mcp-http not on :${MCP_PORT}"
  fi
fi

echo ""
echo "verify-spark-dev-stack OK"
