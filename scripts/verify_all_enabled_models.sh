#!/usr/bin/env bash
# Verify every enabled model in config/models.yaml: load + one inference each.
# One subprocess per model so VRAM is freed between runs. Logs: logs/verify_all/
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs/verify_all"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"
: >"$SUMMARY"

# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"
export SPARSE_BACKEND=spconv SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=sdpa \
       XFORMERS_DISABLED=1 SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
       TORCH_CUDA_ARCH_LIST="9.0+PTX" PYOPENGL_PLATFORM=egl TQDM_DISABLE=1

IMG="assets/example_image/203.png"
MESH="assets/example_mesh/typical_humanoid_goblin.obj"
MESH_GLB="assets/example_mesh/typical_creature_dragon.obj"
VOX_SRC="assets/example_meshedit/images/2d_render.png"
VOX_TGT="assets/example_meshedit/images/2d_edit.png"
VOX_MASK="assets/example_meshedit/images/2d_mask.png"

echo "=== Preflight ===" | tee "$LOG_DIR/preflight.log"
./venv/bin/python scripts/verify_env_compat.py 2>&1 | tee -a "$LOG_DIR/preflight.log"
if ! grep -q PREFLIGHT_OK "$LOG_DIR/preflight.log"; then
  echo "Aborting: fix preflight errors first." | tee -a "$SUMMARY"
  exit 1
fi

run() {
  local name="$1" mod="$2" cls="$3" js="$4" timeout_s="${5:-3600}"
  echo "" | tee -a "$SUMMARY"
  echo "================ $name ================" | tee -a "$SUMMARY"
  if timeout "$timeout_s" ./venv/bin/python scripts/verify_model.py "$mod" "$cls" "$js" \
      > "$LOG_DIR/${name}.log" 2>&1; then
    grep -E "VERIFY_OK|output_mesh_path|output exists|done in|loaded in" "$LOG_DIR/${name}.log" | tail -5 | tee -a "$SUMMARY"
    echo "[$name] PASS" | tee -a "$SUMMARY"
    return 0
  fi
  echo "[$name] FAIL" | tee -a "$SUMMARY"
  grep -aE "Error|Exception|Traceback|Failed|fatal|RuntimeError|No module|VERIFY" "$LOG_DIR/${name}.log" | tail -8 | tee -a "$SUMMARY"
  return 1
}

PASS=0
FAIL=0
record() { if "$@"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi; }

# Light / CPU-first
record run xatlas_uv adapters.xatlas_adapter XatlasUVUnwrappingAdapter \
  "{\"mesh_path\": \"$MESH\", \"output_format\": \"obj\"}" 600
record run instant_meshes adapters.instant_meshes_adapter InstantMeshesRetopologyAdapter \
  "{\"mesh_path\": \"$MESH\", \"output_format\": \"obj\", \"target_vertex_count\": 5000}" 900

# Auto-rig
record run unirig adapters.unirig_adapter UniRigAdapter \
  "{\"mesh_path\": \"$MESH\", \"rig_mode\": \"full\", \"output_format\": \"glb\"}" 1800

# Image -> mesh (moderate)
record run hunyuan_raw adapters.hunyuan3d_adapter_v21 Hunyuan3DV21ImageToRawMeshAdapter \
  "{\"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 1800
record run ultrashape adapters.ultrashape_adapter UltraShapeImageToRawMeshAdapter \
  "{\"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 2400

# TRELLIS family (spconv-sensitive)
record run trellis_text adapters.trellis_adapter TrellisTextToTexturedMeshAdapter \
  '{"text_prompt": "a small bird", "output_format": "glb"}' 2400
record run trellis_text_paint adapters.trellis_adapter TrellisTextMeshPaintingAdapter \
  "{\"mesh_path\": \"$MESH\", \"text_prompt\": \"rusty metal goblin\", \"output_format\": \"glb\"}" 2400
record run trellis_image adapters.trellis_adapter TrellisImageToTexturedMeshAdapter \
  "{\"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 2400
record run trellis_image_paint adapters.trellis_adapter TrellisImageMeshPaintingAdapter \
  "{\"mesh_path\": \"$MESH\", \"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 2400

# TRELLIS.2 + Hunyuan textured
record run trellis2_image adapters.trellis2_adapter Trellis2ImageToTexturedMeshAdapter \
  "{\"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 3600
record run trellis2_image_paint adapters.trellis2_adapter Trellis2ImageMeshPaintingAdapter \
  "{\"mesh_path\": \"$MESH\", \"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 3600
record run hunyuan_textured adapters.hunyuan3d_adapter_v21 Hunyuan3DV21ImageToTexturedMeshAdapter \
  "{\"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 2400
record run hunyuan_image_paint adapters.hunyuan3d_adapter_v21 Hunyuan3DV21ImageMeshPaintingAdapter \
  "{\"mesh_path\": \"$MESH\", \"image_path\": \"$IMG\", \"output_format\": \"glb\"}" 2400

# Segmentation (heavy VRAM)
record run p3sam adapters.p3sam_adapter P3SAMSegmentationAdapter \
  "{\"mesh_path\": \"$MESH\", \"output_format\": \"glb\"}" 3600

# VoxHammer (TRELLIS + editing, 40GB)
record run voxhammer_text adapters.voxhammer_adapter VoxHammerTextMeshEditingAdapter \
  "{\"mesh_path\": \"$MESH_GLB\", \"mask_type\": \"bbox\", \"mask_center\": [0,0,0], \"mask_params\": {\"dimensions\": [0.5,0.5,0.5]}, \"source_prompt\": \"dragon\", \"target_prompt\": \"stone dragon\", \"num_views\": 50, \"resolution\": 256}" 5400
record run voxhammer_image adapters.voxhammer_adapter VoxHammerImageMeshEditingAdapter \
  "{\"mesh_path\": \"$MESH_GLB\", \"mask_type\": \"bbox\", \"mask_center\": [0,0,0], \"mask_params\": {\"dimensions\": [0.5,0.5,0.5]}, \"source_image_path\": \"$VOX_SRC\", \"target_image_path\": \"$VOX_TGT\", \"mask_image_path\": \"$VOX_MASK\", \"num_views\": 50, \"resolution\": 256}" 5400

echo "" | tee -a "$SUMMARY"
echo "VERIFY_ALL_DONE pass=$PASS fail=$FAIL" | tee -a "$SUMMARY"
[[ "$FAIL" -eq 0 ]]
