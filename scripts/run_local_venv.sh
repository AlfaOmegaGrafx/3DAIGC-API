#!/usr/bin/env bash
# Single-worker API using the repo venv (aarch64-friendly; no Docker/conda).
# Expects ./venv from `python3 -m venv venv` and pip-installed deps.
#
# TRELLIS sparse backend: default is spconv (see thirdparty/TRELLIS/trellis/modules/sparse).
# If you built/installed torchsparse instead of spconv:
#   export SPARSE_BACKEND=torchsparse
# For spconv on aarch64 there are no PyPI wheels; use scripts/build_spconv_local.sh
# with CUDA_HOME pointing at a CUDA 12.x toolkit matching PyTorch cu12x.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Toolchain env (PATH with venv+nvcc so spconv/cumm JIT finds ninja, CUDA_HOME,
# CUMM_CUDA_ARCH_LIST, Python headers, PYTHONPATH=repo root). Required for the
# spconv editable build to import on aarch64.
# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$ROOT"
PY="$ROOT/venv/bin/python"

# --- Attention backends for NVIDIA Blackwell / GB10 (compute capability 12.1) ---
# xformers' memory_efficient_attention has no kernel for sm_121 (its backends
# require cap <= 9.0), so:
#   * Dense attention (TRELLIS DiT, etc.) -> PyTorch SDPA, which supports sm_121.
#   * Sparse/SLAT attention only supports xformers|flash_attn; xformers' fp16
#     flash backend does work on sm_121, so keep that for the sparse path.
#   * DINOv2 (image conditioner) uses xformers directly in fp32 (fails on sm_121);
#     XFORMERS_DISABLED=1 makes it fall back to SDPA.
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
if [[ -z "${SPARSE_ATTN_BACKEND:-}" ]]; then
  if "$PY" -c "import flash_attn" 2>/dev/null; then
    export SPARSE_ATTN_BACKEND=flash_attn
  else
    export SPARSE_ATTN_BACKEND=xformers
  fi
fi
export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
# spconv: 'native' implicit-gemm avoids autotune issues on new GPUs.
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
# Offscreen rendering for pyrender thumbnails (no X server).
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Prefer spconv; if missing and torchsparse is available, switch backend.
export SPARSE_BACKEND="${SPARSE_BACKEND:-spconv}"
if [[ "$SPARSE_BACKEND" == "spconv" ]] && ! "$PY" -c "import spconv.pytorch" 2>/dev/null; then
  if "$PY" -c "import torchsparse" 2>/dev/null; then
    export SPARSE_BACKEND=torchsparse
    echo "[run_local_venv] spconv not found; using SPARSE_BACKEND=torchsparse" >&2
  else
    echo "[run_local_venv] WARNING: neither spconv nor torchsparse is importable. TRELLIS jobs will fail until" >&2
    echo "  you install one of them (CUDA 12.x nvcc: ./scripts/build_spconv_local.sh or torchsparse build)." >&2
  fi
fi

cd "$ROOT"
HOST="${P3D_HOST:-0.0.0.0}"
PORT="${P3D_PORT:-7842}"
exec "$ROOT/venv/bin/uvicorn" api.main_singleworker:app --host "$HOST" --port "$PORT" "$@"
