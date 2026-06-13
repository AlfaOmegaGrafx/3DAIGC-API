#!/usr/bin/env bash
# Bare-metal venv install: bootstrap + GPU stack + most pip-only thirdparty deps from
# scripts/install.sh, with aarch64-safe skips (no conda; no x86-only spconv-cu120 wheel;
# no PyPI open3d==0.18.0 on many aarch64 indexes).
#
# Usage (from repo root):
#   ./scripts/install_local_venv_extras.sh
#
# Optional: export CUDA_HOME first. For interactive shells afterward:
#   source ./scripts/env_local_gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="$ROOT/venv"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

if [[ ! -x "$PIP" ]]; then
  echo "Expected venv at $VENV" >&2
  exit 1
fi

# Match install_next_steps / build_spconv toolchain defaults.
# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"

echo "=== [1/6] bootstrap_local_models.sh ==="
"$ROOT/scripts/bootstrap_local_models.sh"

echo "=== [2/6] install_next_steps.sh (spconv/cumm + PyG) ==="
"$ROOT/scripts/install_next_steps.sh"

echo "=== [3a/6] Core model runtime deps (TRELLIS/Hunyuan/PartPacker/VoxHammer) ==="
# diffusers/omegaconf/addict are imported at model-load time by several adapters and were
# missing from earlier runs (server started but every diffusion/TRELLIS job failed).
"$PIP" install -q diffusers==0.32.2 omegaconf addict einops opencv-python-headless || true

echo "=== [3/6] PartField-style deps (from install.sh) ==="
# pytz is required by thirdparty/PartField/partfield/config; PartField PVCNN needs PyG (below).
"$PIP" install -q lightning==2.2 h5py yacs trimesh scikit-image loguru boto3 pytz \
  mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh psutil \
  pymeshfix igraph || true

echo "=== [4/6] UniRig / PartPacker / PartUV / P3SAM / FastMesh / UltraShape / VoxHammer (pip) ==="
# UniRig: skip spconv-cu120 (x86 wheel); use editable spconv from install_next_steps.
# addict is required by UniRig at import time.
"$PIP" install -q addict pyrender fast-simplification python-box timm || true

"$PIP" install -q pybind11==3.0.1 meshiki kiui fpsample pymcubes einops || true

"$PIP" install -q seaborn partuv || true
# blenderproc is heavy / optional; skip by default
# "$PIP" install -q blenderproc || true

if [[ -d "$ROOT/thirdparty/Hunyuan3DPart/P3SAM" ]]; then
  "$PIP" install -q numba scikit-learn fpsample || true
fi

if [[ -f "$ROOT/thirdparty/FastMesh/requirement_extra.txt" ]]; then
  "$PIP" install -q -r "$ROOT/thirdparty/FastMesh/requirement_extra.txt" || true
fi

"$PIP" install -q "git+https://github.com/ashawkey/cubvh" --no-build-isolation || true
"$PIP" install -q "git+https://github.com/huanngzh/bpy-renderer.git" || true
"$PIP" install -q pysdf sentencepiece || true

echo "=== [4b/6] nvdiffrast (TRELLIS texture baking / GLB export; CUDA backend, JIT-compiled) ==="
# Required by thirdparty/TRELLIS/trellis/utils/postprocessing_utils.py (import nvdiffrast.torch).
"$PIP" install -q "git+https://github.com/NVlabs/nvdiffrast.git" || \
  echo "[install_local_venv_extras] nvdiffrast install failed; TRELLIS textured-mesh export will fail." >&2

echo "=== [5/6] Hunyuan3D-2.1 native pieces (optional) ==="
if [[ -d "$ROOT/thirdparty/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer" ]]; then
  ( cd "$ROOT/thirdparty/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer" && "$PIP" install -q -e . --no-build-isolation ) || \
    echo "[install_local_venv_extras] Hunyuan custom_rasterizer build skipped/failed (see log)." >&2
fi
if [[ -f "$ROOT/thirdparty/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/compile_mesh_painter.sh" ]]; then
  ( cd "$ROOT/thirdparty/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh ) || \
    echo "[install_local_venv_extras] Hunyuan compile_mesh_painter.sh skipped/failed." >&2
fi
# Full requirements-inference.txt pins old numpy/torch; do not install wholesale on cu128 venv.

echo "=== [6/6] Project requirements.txt (aarch64: drop open3d + strict numpy pin if needed) ==="
_req_filtered="$(mktemp)"
if [[ "$(uname -m)" == "aarch64" ]]; then
  grep -v '^open3d==' "$ROOT/requirements.txt" | grep -v '^numpy==' > "$_req_filtered" || true
  echo "[install_local_venv_extras] aarch64: installing requirements without open3d== / numpy== pins (install open3d separately if a wheel exists for your platform)." >&2
else
  cp "$ROOT/requirements.txt" "$_req_filtered"
fi
"$PIP" install -q -r "$_req_filtered" || {
  echo "[install_local_venv_extras] requirements.txt install had errors; retry after resolving conflicts." >&2
}
rm -f "$_req_filtered"

"$PIP" install -q huggingface_hub || true

echo ""
echo "Done. Next:"
echo "  - Models:  ./scripts/download_models.sh -m trellis,partfield   # or -m all"
echo "  - API:     ./scripts/run_local_venv.sh"
echo "  - Kaolin:  no official torch-2.11+cu128 wheel index on NVIDIA S3 yet; build from source or use Docker for TRELLIS.2 mesh paths that need it."
echo "  - Shell:   source ./scripts/env_local_gpu.sh"
