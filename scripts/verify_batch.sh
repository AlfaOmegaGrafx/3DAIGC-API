#!/usr/bin/env bash
# Sequentially verify multiple model adapters (one process each so GPU memory frees
# between models). Concise output; tqdm disabled. Logs per-model under logs/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"
export SPARSE_BACKEND=spconv SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=sdpa \
       XFORMERS_DISABLED=1 SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
       TORCH_CUDA_ARCH_LIST="9.0+PTX" PYOPENGL_PLATFORM=egl TQDM_DISABLE=1

run() {  # name module class json
  local name="$1" mod="$2" cls="$3" js="$4"
  echo "================ $name ================"
  if timeout 1800 python scripts/verify_model.py "$mod" "$cls" "$js" > "logs/vb_${name}.log" 2>&1; then
    grep -E "VERIFY_OK|output_mesh_path|output exists|done in" "logs/vb_${name}.log" | tail -4
    echo "[$name] PASS"
  else
    echo "[$name] FAIL (last error):"
    grep -aE "Error|Exception|Traceback|No module|RuntimeError|assert" "logs/vb_${name}.log" | tail -6
  fi
}

run fastmesh adapters.fastmesh_adapter FastMeshRetopologyAdapter \
    '{"mesh_path": "assets/example_mesh/typical_humanoid_goblin.obj"}'
run trellis2_image adapters.trellis2_adapter Trellis2ImageToTexturedMeshAdapter \
    '{"image_path": "assets/example_image/203.png"}'
run hunyuan_textured adapters.hunyuan3d_adapter_v21 Hunyuan3DV21ImageToTexturedMeshAdapter \
    '{"image_path": "assets/example_image/203.png"}'
run trellis_text adapters.trellis_adapter TrellisTextToTexturedMeshAdapter \
    '{"text_prompt": "a cute cartoon robot"}'
echo "VERIFY_BATCH_DONE"
