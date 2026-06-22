#!/usr/bin/env bash
# Start 3DAIGC prerequisites + xr-ai 3daigc-vlm-example stack.
set -euo pipefail

MCP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XR_AI_ROOT="${XR_AI_ROOT:-/home/sifr/xr-ai}"
SAMPLE="${XR_AI_ROOT}/agent-samples/3daigc-vlm-example"

bash "${MCP_ROOT}/scripts/start_prerequisites.sh"

echo "==> Syncing 3daigc-vlm-example (orchestrator + worker)"
cd "${SAMPLE}"
uv sync
(cd worker && uv sync)

echo "==> Starting 3daigc-vlm-example (xr-ai)"
exec uv run daigc_vlm_example
