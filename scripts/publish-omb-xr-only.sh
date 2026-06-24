#!/usr/bin/env bash
# Stage only OMB/RP1 spatial fabric, XR MCP, and new model adapters for a public push.
# Usage: bash scripts/publish-omb-xr-only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

git add \
  .env.example \
  adapters/colmap_3dgs_adapter.py \
  adapters/hunyuan3d_adapter_v21.py \
  adapters/pixal3d_adapter.py \
  adapters/worldmirror2_adapter.py \
  api/main_multiworker.py \
  api/main_singleworker.py \
  api/routers/image_tools.py \
  api/routers/multi_image_resolve.py \
  api/routers/spatial_fabric.py \
  config/models.yaml \
  core/scheduler/model_factory.py \
  core/spatial_fabric/ \
  core/utils/multi_image_input.py \
  docs/api_documentation.md \
  mcp/daigc_mcp/client.py \
  mcp/docs/XR_VOICE_COMMANDS.md \
  mcp/scripts/xr-spark-media-relay.mjs \
  mcp/yaml/xr_ai_3daigc_overlay.yaml \
  scripts/publish-omb-xr-only.sh \
  scripts/run_server.sh \
  scripts/start_services_detached.sh \
  scripts/sync-spatial-fabric-env.sh

echo "Staged OMB/XR + models slice. Review with: git diff --cached --stat"
