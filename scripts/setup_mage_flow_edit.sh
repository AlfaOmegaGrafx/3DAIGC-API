#!/usr/bin/env bash
# Install Mage-Flow-Edit-Turbo in an isolated venv (does not touch main API transformers pin).
# Run from 3DAIGC-API repo root on DGX / CUDA host.
#
# Usage:
#   bash scripts/setup_mage_flow_edit.sh              # deps + download weights
#   bash scripts/setup_mage_flow_edit.sh --deps-only  # skip HF weight download
#
# Weights mirror (official microsoft/Mage-Flow-Edit-Turbo host withdrawn):
#   mage-flow-community/Mage-Flow-Edit-Turbo @ pinned revision
# Code: thirdparty/Mage (microsoft/Mage)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAGE_ROOT="${MAGE_ROOT:-${ROOT}/thirdparty/Mage}"
VENV="${MAGE_FLOW_VENV:-${ROOT}/.venv-mage-flow}"
WEIGHTS_DIR="${ROOT}/pretrained/mage-flow/Mage-Flow-Edit-Turbo"
HF_ID="${MAGE_FLOW_EDIT_HF_ID:-mage-flow-community/Mage-Flow-Edit-Turbo}"
HF_REVISION="${MAGE_FLOW_EDIT_HF_REVISION:-66df6fa1aba5b40cd4120739134292eab9779da3}"
DEPS_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --deps-only) DEPS_ONLY=1 ;;
    -h|--help)
      sed -n '1,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

echo "=== Mage-Flow-Edit setup ==="
echo "Code: ${MAGE_ROOT}"
echo "Venv: ${VENV}"
echo "Weights: ${WEIGHTS_DIR}"
echo "HF: ${HF_ID}@${HF_REVISION}"

if [[ ! -d "${MAGE_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/microsoft/Mage.git "${MAGE_ROOT}"
else
  echo "Mage repo present — fetch (non-fatal)"
  git -C "${MAGE_ROOT}" fetch --depth 1 origin main || true
fi

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel

# Reuse main venv torch if present (GB10 / cu128) — Mage docs want 2.13 but
# aarch64 Spark currently runs 2.11+cu128 in the main stack.
MAIN_TORCH_VER="$("${ROOT}/venv/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [[ -n "${MAIN_TORCH_VER}" ]]; then
  echo "Syncing torch from main venv: ${MAIN_TORCH_VER}"
  pip install "torch==${MAIN_TORCH_VER}" "torchvision" --index-url https://download.pytorch.org/whl/cu128 \
    || pip install "torch==${MAIN_TORCH_VER}" torchvision
else
  echo "Installing torch from PyPI (fallback)"
  pip install "torch>=2.11" "torchvision"
fi

echo "Installing Mage-Flow runtime deps (isolated — transformers 5.x OK here)…"
pip install \
  "transformers>=5.3.0,<5.6" \
  "diffusers>=0.37.0" \
  "accelerate>=1.0.0" \
  "einops>=0.8.0" \
  "safetensors>=0.4.0" \
  "huggingface_hub>=0.20" \
  "pillow>=10.0" \
  "numpy>=1.26" \
  "pydantic>=2.0" \
  "loguru>=0.7.0"

echo "Installing mage_flow package (editable, no deps)…"
pip install -e "${MAGE_ROOT}/mage_flow" --no-deps

# flash-attn optional — default runner uses SDPA backend on Spark.
if [[ "${MAGE_FLOW_INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  echo "Installing flash-attn (requested via MAGE_FLOW_INSTALL_FLASH_ATTN=1)…"
  pip install setuptools wheel ninja
  pip install --no-build-isolation "flash-attn==2.8.3" || \
    echo "WARN: flash-attn install failed — SDPA backend will be used"
fi

export PYTHONPATH="${MAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PY'
from mage_flow import MageFlowPipeline
from mage_flow.models.modules._attn_backend import set_attn_backend
set_attn_backend("sdpa")
print("MageFlowPipeline OK:", MageFlowPipeline)
PY

if [[ "${DEPS_ONLY}" == "1" ]]; then
  echo "Deps only — skipping weight download."
  exit 0
fi

mkdir -p "$(dirname "${WEIGHTS_DIR}")"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "${HF_ID}",
    revision="${HF_REVISION}",
    local_dir="${WEIGHTS_DIR}",
)
print("Weights OK:", "${WEIGHTS_DIR}")
PY

test -f "${WEIGHTS_DIR}/model_index.json"

echo "=== Mage-Flow-Edit setup complete ==="
echo "Enable route: mage_flow_edit_turbo in config/models.yaml"
echo "Default attn backend: SDPA (set MAGE_FLOW_ATTN_BACKEND=flash2 if flash-attn installed)"
echo "Do NOT pip-install Mage into main venv/ — keeps transformers==4.57.3 for Krea"
