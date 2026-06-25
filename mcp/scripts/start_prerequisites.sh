#!/usr/bin/env bash
# Verify 3DAIGC-API and 3daigc-mcp-http are up before starting XR AI sample.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_URL="${DAIGC_API_BASE_URL:-http://localhost:7842}"
MCP_PORT="${DAIGC_MCP_HTTP_PORT:-8260}"

echo "==> Checking 3DAIGC-API at ${API_URL}"
if ! curl -sf "${API_URL}/api/v1/system/health" >/dev/null; then
  echo "ERROR: 3DAIGC-API not reachable at ${API_URL}" >&2
  echo "Start it with: bash ${ROOT}/../scripts/run_server.sh" >&2
  exit 1
fi
echo "    OK"

echo "==> Ensuring 3daigc-mcp-http on :${MCP_PORT}"
bash "${ROOT}/scripts/run_http.sh"

echo "==> Probing MCP endpoint"
cd "${ROOT}"
uv run python - <<PY
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:${MCP_PORT}/mcp") as c:
        tools = await c.list_tools()
        names = [t.name for t in tools]
        assert "health_check" in names, names

asyncio.run(main())
print("    OK (health_check tool listed)")
PY

echo "Prerequisites ready."
