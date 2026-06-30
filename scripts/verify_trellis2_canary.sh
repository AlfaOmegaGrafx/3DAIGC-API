#!/usr/bin/env bash
# Fast canary for TRELLIS.2 after pip/env changes. Uses fox_test.png + seed 42.
# Full verify: scripts/verify_all_enabled_models.sh (trellis2_image).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"

IMG="${ROOT}/assets/example_image/fox_test.png"
if [[ ! -f "$IMG" ]]; then
  IMG="${ROOT}/assets/example_image/203.png"
fi

export SPARSE_BACKEND=spconv SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=sdpa \
       XFORMERS_DISABLED=1 SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
       TORCH_CUDA_ARCH_LIST="9.0+PTX" PYOPENGL_PLATFORM=egl TQDM_DISABLE=1

echo "=== TRELLIS.2 canary (fox, seed=42, texture 1024) ==="
timeout 3600 ./venv/bin/python scripts/verify_model.py \
  trellis2_canary adapters.trellis2_adapter Trellis2ImageToTexturedMeshAdapter \
  "{\"image_path\": \"$IMG\", \"output_format\": \"glb\", \"seed\": 42, \"texture_size\": 1024, \"decimation_target\": 100000}"
