#!/usr/bin/env bash
# Verify DGX hub + Surface proxy for XR voice.
set -euo pipefail

DGX_IP="${DGX_IP:-10.0.0.158}"
SURFACE_IP="${SURFACE_IP:-10.0.0.32}"
fail=0

check() {
  local label="$1" url="$2"
  local code
  code="$(curl -sk --connect-timeout 8 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)"
  if [[ "$code" == "200" ]]; then
    echo "  OK  ${label} — ${url} (HTTP ${code})"
  else
    echo "  FAIL ${label} — ${url} (HTTP ${code})"
    fail=$((fail + 1))
  fi
}

echo "=== XR voice stack verify ==="
check "DGX hub" "https://${DGX_IP}:8088/"
check "Surface proxy (Galaxy XR)" "https://${SURFACE_IP}:8443/"
curl -sf "http://127.0.0.1:7842/api/v1/system/health" >/dev/null && echo "  OK  3DAIGC-API :7842" || { echo "  FAIL 3DAIGC-API :7842"; fail=$((fail + 1)); }
ss -tln 2>/dev/null | grep -q ':8260' && echo "  OK  MCP :8260" || { echo "  FAIL MCP :8260"; fail=$((fail + 1)); }

if [[ "$fail" -eq 0 ]]; then
  echo "XR_VOICE_VERIFY_OK"
  exit 0
fi
echo "XR_VOICE_VERIFY_FAIL"
exit 1
