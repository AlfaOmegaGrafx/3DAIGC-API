#!/usr/bin/env bash
# Echo CUDA 12.x detection and run common local GPU dependency steps (3DAIGC-API venv).
# Does not install system packages (python*-dev, cuda-toolkit); see messages below.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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

echo "=== CUDA / nvcc (default PATH) ==="
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "(nvcc not on PATH)"
echo ""
if detected="$(detect_cuda12_home)"; then
  echo "Detected CUDA 12.x toolkit: $detected"
  echo "Suggested session exports:"
  echo "  export CUDA_HOME=$detected"
  echo "  export PATH=\"\$CUDA_HOME/bin:\$ROOT/venv/bin:\$PATH\""
  echo "  # If cumm/spconv fails to compile: sudo apt install gcc-12 g++-12 (this script uses them when present)"
  export CUDA_HOME="$detected"
  export PATH="$CUDA_HOME/bin:$ROOT/venv/bin:$PATH"
  # Avoid cumm nvidia-smi fallback (unknown GPU embeds sm_52 in ninja); aarch64 CUDA 12.8 nvcc rejects compute_52.
  export CUMM_CUDA_ARCH_LIST="${CUMM_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0+PTX}"
  nvrel="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
  # Same rationale as build_spconv_local.sh: cumm + nvcc + g++-13 is fragile; default to gcc-12 when available.
  if command -v g++-12 >/dev/null 2>&1 && [[ -z "${CXX:-}" ]]; then
    export CC="${CC:-gcc-12}" CXX="${CXX:-g++-12}"
    export CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v "$CXX")}"
    echo "[install_next_steps] Using CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX for native extension builds." >&2
  fi
else
  echo "No CUDA 12.x nvcc found under /usr/local/cuda-12* or /usr/lib/nvidia-cuda-toolkit."
  echo "Install NVIDIA CUDA 12.8 toolkit or distro cuda-toolkit-12 packages, then re-run."
  exit 1
fi

pyver="$("$ROOT/venv/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
for cand in "/usr/include/python${pyver}" "$HOME/.local/py${pyver//./}-dev/usr/include/python${pyver}" "${PYTHON_INCLUDE_DIR:-}"; do
  if [[ -n "$cand" && -f "$cand/Python.h" ]]; then
    cand_root="$(dirname "$cand")"
    export CPATH="$cand_root:$cand${CPATH:+:$CPATH}"
    echo "[install_next_steps] Using Python headers: $cand (+ multiarch root $cand_root)" >&2
    break
  fi
done
if [[ -n "${CXX:-}" ]]; then
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

echo ""
echo "=== ./scripts/build_spconv_local.sh ==="
if [[ -x "$ROOT/scripts/build_spconv_local.sh" ]]; then
  "$ROOT/scripts/build_spconv_local.sh" || {
    echo "[install_next_steps] build_spconv_local.sh failed (see above)." >&2
    exit 1
  }
else
  echo "Missing scripts/build_spconv_local.sh" >&2
  exit 1
fi

echo ""
echo "=== PyG: torch-scatter torch-cluster (aarch64 often builds from source) ==="
"$ROOT/venv/bin/pip" install --no-build-isolation torch-scatter torch-cluster \
  -f "https://data.pyg.org/whl/torch-2.11.0+cu128.html" \
  || echo "[install_next_steps] torch-scatter/torch-cluster install failed — install python*-dev, use GCC 12 with CUDA 12.0, or install CUDA 12.8 toolkit." >&2

echo ""
echo "Done. Kaolin: see NVIDIA Kaolin docs for wheel index matching your torch+cuda (aarch64 wheels may be unavailable)."
