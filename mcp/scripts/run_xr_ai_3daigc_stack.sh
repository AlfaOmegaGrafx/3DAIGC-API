#!/usr/bin/env bash
# Start 3DAIGC prerequisites + xr-ai 3daigc-vlm-example stack.
set -euo pipefail

MCP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XR_AI_ROOT="${XR_AI_ROOT:-/home/sifr/xr-ai}"
SAMPLE="${XR_AI_ROOT}/agent-samples/3daigc-vlm-example"

bash "${MCP_ROOT}/scripts/start_prerequisites.sh"

# Tailscale `serve --tcp=8088` reserves the port even when xr_media_hub is down.
if ss -tln 2>/dev/null | grep -q ':8088'; then
  if ! curl -sk --connect-timeout 2 -o /dev/null https://127.0.0.1:8088/ 2>/dev/null; then
    echo "==> Port 8088 held but hub not responding — resetting tailscale serve"
    tailscale serve reset 2>/dev/null || true
    sleep 1
  fi
fi

echo "==> Syncing 3daigc-vlm-example (orchestrator + worker)"
cd "${SAMPLE}"
uv sync
(cd worker && uv sync)

echo "==> Starting 3daigc-vlm-example (xr-ai)"
exec uv run daigc_vlm_example
