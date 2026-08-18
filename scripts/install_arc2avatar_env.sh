#!/usr/bin/env bash
# GB10 / CUDA 12.8 Arc2Avatar env (upstream README pins CUDA 11.8 — do not use on Spark).
# Creates thirdparty/Arc2Avatar/.venv and appends ARC2AVATAR_* to .env
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARC_ROOT="${ARC2AVATAR_ROOT:-$ROOT/thirdparty/Arc2Avatar}"
VENV="$ARC_ROOT/.venv"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:${PATH:-}"
# GB10 (Blackwell) — PTX fallback if arch list wrong
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

if [[ ! -d "$ARC_ROOT" ]]; then
  echo "ERROR: missing $ARC_ROOT — clone Arc2Avatar first" >&2
  exit 1
fi

echo "==> Arc2Avatar env at $VENV (CUDA_HOME=$CUDA_HOME)"
if [[ ! -x "$VENV/bin/python" ]]; then
  /usr/bin/python3.12 -m venv "$VENV"
fi
"$VENV/bin/pip" install -U pip setuptools wheel

echo "==> torch cu128"
"$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda not available"
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

echo "==> Python deps (no cu118 pins; skip open3d — no aarch64 wheel)"
# Install requirements without torch/torchvision/torchaudio/xformers/open3d
REQ_FILTERED="$(mktemp)"
grep -vE '^(--|torch|torchvision|torchaudio|xformers|pip==|triton$|open3d)' \
  "$ARC_ROOT/requirements.txt" >"$REQ_FILTERED" || true
# Drop onnxruntime-gpu on aarch64 (often missing); use CPU onnxruntime
grep -v 'onnxruntime-gpu' "$REQ_FILTERED" >"${REQ_FILTERED}.2"
echo "onnxruntime" >>"${REQ_FILTERED}.2"
"$VENV/bin/pip" install -r "${REQ_FILTERED}.2"
rm -f "$REQ_FILTERED" "${REQ_FILTERED}.2"

echo "==> CUDA extensions (diff-gaussian; simple_knn optional — Arc2Avatar uses scipy KDTree)"
# Prefer copy from API venv when same torch/cu128 (avoids GB10 header rebuild pain).
API_SITE="$ROOT/venv/lib/python3.12/site-packages"
ARC_SITE="$VENV/lib/python3.12/site-packages"
if [[ -d "$API_SITE/diff_gaussian_rasterization" ]]; then
  echo "Copying diff_gaussian_rasterization from API venv"
  cp -a "$API_SITE/diff_gaussian_rasterization" "$ARC_SITE/"
  cp -a "$API_SITE"/diff_gaussian_rasterization*.dist-info "$ARC_SITE/" 2>/dev/null || true
else
  # Fix common CUDA12 + cstdint issue in upstream submodule
  HDR="$ARC_ROOT/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
  if [[ -f "$HDR" ]] && ! grep -q cstdint "$HDR"; then
    sed -i '14a#include <cstdint>' "$HDR"
  fi
  "$VENV/bin/pip" install --no-build-isolation "$ARC_ROOT/submodules/diff-gaussian-rasterization/"
fi
# simple_knn is unused by current Arc2Avatar gaussian_model (ndistCUDA2 → scipy KDTree)
if ! "$VENV/bin/python" -c "import simple_knn" 2>/dev/null; then
  echo "WARN: simple_knn not installed (ok — scipy KDTree path used)"
fi

echo "==> Import smoke"
"$VENV/bin/python" - <<PY
import sys
sys.path.insert(0, "$ARC_ROOT")
import torch
import diff_gaussian_rasterization  # noqa: F401
assert torch.cuda.is_available()
print("OK imports", torch.__version__)
PY

ENV_FILE="$ROOT/.env"
touch "$ENV_FILE"
PY="$VENV/bin/python"
if grep -q '^ARC2AVATAR_PYTHON=' "$ENV_FILE" 2>/dev/null; then
  sed -i "s|^ARC2AVATAR_PYTHON=.*|ARC2AVATAR_PYTHON=$PY|" "$ENV_FILE"
else
  echo "ARC2AVATAR_PYTHON=$PY" >>"$ENV_FILE"
fi
if grep -q '^ARC2AVATAR_ROOT=' "$ENV_FILE" 2>/dev/null; then
  sed -i "s|^ARC2AVATAR_ROOT=.*|ARC2AVATAR_ROOT=$ARC_ROOT|" "$ENV_FILE"
else
  echo "ARC2AVATAR_ROOT=$ARC_ROOT" >>"$ENV_FILE"
fi

echo "==> Done. Restart API: P3D_SKIP_PREFLIGHT=1 bash $ROOT/scripts/restart_services.sh"
echo "    curl -s http://127.0.0.1:7842/api/v1/arc2avatar/status | jq ."
