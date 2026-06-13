#!/usr/bin/env bash
# Source from bash before Python jobs that JIT-build cumm/spconv:
#   source /path/to/3DAIGC-API/scripts/env_local_gpu.sh
#
# Sets CUDA_HOME (CUDA 12.x if found), PATH (venv + nvcc), CPATH for Python headers,
# optional GCC 12 + NVCC_PREPEND_FLAGS, CUMM_CUDA_ARCH_LIST, and PYTHONPATH=repo root.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "usage:  source ${0}" >&2
  exit 2
fi

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_VENV="$_ROOT/venv"

detect_cuda12_home() {
  local d ver maj cand
  for d in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-12; do
    if [[ -x "$d/bin/nvcc" ]]; then
      ver="$("$d/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
      maj="${ver%%.*}"
      if [[ "$maj" == "12" ]]; then echo "$d"; return 0; fi
    fi
  done
  cand="/usr/lib/nvidia-cuda-toolkit"
  if [[ -x "$cand/bin/nvcc" ]]; then
    ver="$("$cand/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
    maj="${ver%%.*}"
    if [[ "$maj" == "12" ]]; then echo "$cand"; return 0; fi
  fi
  return 1
}

if [[ -z "${CUDA_HOME:-}" ]] || [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if _d="$(detect_cuda12_home)"; then
    export CUDA_HOME="$_d"
  fi
fi
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$_VENV/bin:$PATH"
else
  export PATH="$_VENV/bin:$PATH"
  echo "[env_local_gpu] WARNING: CUDA 12.x toolkit not found; set CUDA_HOME manually." >&2
fi

export CUMM_CUDA_ARCH_LIST="${CUMM_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0+PTX}"

if [[ -x "$_VENV/bin/python" ]]; then
  _pyver="$("$_VENV/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
  for _cand in "/usr/include/python${_pyver}" "$HOME/.local/py${_pyver//./}-dev/usr/include/python${_pyver}" "${PYTHON_INCLUDE_DIR:-}"; do
    if [[ -n "$_cand" && -f "$_cand/Python.h" ]]; then
      _cand_root="$(dirname "$_cand")"
      export CPATH="${_cand_root}:${_cand}${CPATH:+:$CPATH}"
      break
    fi
  done
fi

if command -v g++-12 >/dev/null 2>&1 && [[ -z "${CXX:-}" ]]; then
  export CC="${CC:-gcc-12}" CXX="${CXX:-g++-12}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v "$CXX")}"
  _ccbin="$(command -v "$CXX")"
  if [[ -z "${NVCC_PREPEND_FLAGS:-}" ]]; then
    export NVCC_PREPEND_FLAGS="-ccbin=$_ccbin --std=c++17 -Xcompiler=-std=c++17"
  fi
elif command -v g++-12 >/dev/null 2>&1 && [[ -z "${NVCC_PREPEND_FLAGS:-}" ]]; then
  export NVCC_PREPEND_FLAGS="-ccbin=$(command -v g++-12) --std=c++17 -Xcompiler=-std=c++17"
fi
if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]] && [[ "${NVCC_PREPEND_FLAGS}" != *"deprecated-gpu-targets"* ]]; then
  export NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS -Wno-deprecated-gpu-targets"
fi

export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$_ROOT"

# PyTorch extension modules (custom_rasterizer, flash_attn, etc.) need libc10/c10 at runtime.
if [[ -x "$_VENV/bin/python" ]]; then
  _torch_lib="$("$_VENV/bin/python" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || true)"
  if [[ -n "$_torch_lib" && -d "$_torch_lib" ]]; then
    export LD_LIBRARY_PATH="${_torch_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

if command -v blender >/dev/null 2>&1; then
  export BLENDER_BIN="${BLENDER_BIN:-$(command -v blender)}"
fi
export DAIGC_ROOT="${DAIGC_ROOT:-$_ROOT}"

# Blackwell / GB10 (sm_121): route dense attention through PyTorch SDPA, not xformers fp32.
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-xformers}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "[env_local_gpu] ROOT=$_ROOT CUDA_HOME=${CUDA_HOME:-"(unset)"} CUMM_CUDA_ARCH_LIST=$CUMM_CUDA_ARCH_LIST BLENDER_BIN=${BLENDER_BIN:-"(unset)"} ATTN_BACKEND=$ATTN_BACKEND" >&2

unset _ROOT _VENV _d _pyver _cand _cand_root _ccbin _torch_lib
