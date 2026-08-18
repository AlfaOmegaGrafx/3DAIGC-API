#!/usr/bin/env bash
# Rebuild JeffreyXiang/CuMesh for DGX Spark GB10 (PyTorch cu128).
# CUDA 12.8 nvcc cannot target sm_121; use 12.0+PTX (+ optional 9.0).
set -euo pipefail
ROOT="${1:-/tmp/CuMesh-gb10}"
VENV="${VENV:-/home/sifr/3DAIGC-API/venv}"
if [[ ! -d "$ROOT/.git" ]]; then
  git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$ROOT"
fi
cd "$ROOT"
git fetch origin
git checkout 12289e1062f0603f2f0d0771b02e1395d247f26f
git submodule update --init --recursive
# Apply M==0 guard if marker missing
if ! grep -q 'M==0 => no boundary adjacency' src/connectivity.cu; then
  echo "ERROR: apply scripts/patches/cumesh-connectivity-m0-guard (or re-copy patched connectivity.cu)" >&2
  exit 1
fi
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;12.0+PTX}"
export NVCC_APPEND_FLAGS="--extended-lambda"
rm -rf build *.egg-info
"$VENV/bin/pip" install . --no-build-isolation
echo "Installed. Restart API workers to load the new .so."
