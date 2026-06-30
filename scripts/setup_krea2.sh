#!/usr/bin/env bash
# Install Krea 2 text-to-image deps (local open weights via diffusers — no Krea API).
# Run from 3DAIGC-API repo root on DGX / CUDA host.
#
# Usage:
#   bash scripts/setup_krea2.sh              # install deps + download Turbo weights
#   bash scripts/setup_krea2.sh --raw        # also download Raw (LoRA training base)
#   bash scripts/setup_krea2.sh --deps-only  # skip HF weight download
#
# Weights: https://huggingface.co/krea/Krea-2-Turbo
# Code ref: https://github.com/krea-ai/krea-2 (optional; diffusers pipeline is primary)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-${ROOT}/venv}"
DOWNLOAD_RAW=0
DEPS_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --raw) DOWNLOAD_RAW=1 ;;
    --deps-only) DEPS_ONLY=1 ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${VENV}" ]]; then
  echo "Creating venv at ${VENV}"
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel

CONSTRAINTS_HF="${ROOT}/scripts/constraints-hf.txt"
CONSTRAINTS_RT="${ROOT}/scripts/constraints-models-runtime.txt"
# Pin diffusers main until PyPI ships Krea2Pipeline (see huggingface/diffusers pipelines/krea2).
DIFFUSERS_KREA_GIT_REF="${DIFFUSERS_KREA_GIT_REF:-ea802951f5fb235b6af8fe9247f56187d49748b2}"

echo "=== Installing Krea 2 deps (pinned diffusers git ref for Krea2Pipeline) ==="
pip install -c "$CONSTRAINTS_HF" -c "$CONSTRAINTS_RT" \
  "accelerate>=1.0.0" "einops>=0.7.0" "safetensors>=0.4.0"
pip install -c "$CONSTRAINTS_HF" -c "$CONSTRAINTS_RT" peft==0.17.1
pip install -c "$CONSTRAINTS_HF" -c "$CONSTRAINTS_RT" \
  "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_KREA_GIT_REF}"

python3 - <<'PY'
from diffusers import Krea2Pipeline
print("Krea2Pipeline OK:", Krea2Pipeline)
PY

if [[ "${DEPS_ONLY}" == "1" ]]; then
  echo "Deps only — skipping weight download."
  exit 0
fi

if ! command -v hf >/dev/null 2>&1 && ! ./venv/bin/python3 -c "import huggingface_hub" 2>/dev/null; then
  pip install -U "huggingface_hub[cli]"
fi

TURBO_DIR="${ROOT}/pretrained/krea/Krea-2-Turbo"
echo "=== Downloading Krea 2 Turbo weights to ${TURBO_DIR} ==="
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("krea/Krea-2-Turbo", local_dir="${TURBO_DIR}")
print("Turbo weights OK:", "${TURBO_DIR}")
PY

if [[ "${DOWNLOAD_RAW}" == "1" ]]; then
  RAW_DIR="${ROOT}/pretrained/krea/Krea-2-Raw"
  echo "=== Downloading Krea 2 Raw weights to ${RAW_DIR} ==="
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("krea/Krea-2-Raw", local_dir="${RAW_DIR}")
print("Raw weights OK:", "${RAW_DIR}")
PY
fi

echo "=== Krea 2 setup complete ==="
echo "Run post-pip guard: bash scripts/post_pip_guard.sh"
echo "Enable route: krea2_turbo_text_to_image in config/models.yaml"
echo "Commercial: Krea 2 Community License — see docs/MODEL_LICENSES.md"
