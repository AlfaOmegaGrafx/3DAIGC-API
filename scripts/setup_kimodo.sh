#!/usr/bin/env bash
# Install NVIDIA Kimodo under thirdparty/kimodo for text-to-motion (3DAIGC-API).
# Run ON DGX from 3DAIGC-API repo root.
#
# Usage:
#   bash scripts/setup_kimodo.sh
#
# Docs: https://github.com/nv-tlabs/kimodo
# Models: https://huggingface.co/collections/nvidia/kimodo-v1
#
# Note: Kimodo lists scenepic in pyproject but CLI inference does not import it.
# pip sdist for scenepic fails on aarch64 (missing scenepic.min.js) — we install
# without scenepic and skip demo-only deps.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${ROOT}/thirdparty/kimodo"
VENV="${ROOT}/.venv-kimodo"

echo "=== Kimodo setup ==="
echo "Target: ${TARGET}"

if [[ ! -d "${TARGET}/.git" ]]; then
  git clone --depth 1 https://github.com/nv-tlabs/kimodo.git "${TARGET}"
else
  echo "Kimodo repo already present — git pull"
  git -C "${TARGET}" pull --ff-only || true
fi

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel

echo "Installing Kimodo package (skip MotionCorrection C++ ext — x86 AVX flags fail on aarch64)…"
export SKIP_MOTION_CORRECTION_IN_SETUP=1
pip install -e "${TARGET}" --no-deps

echo "Installing runtime dependencies…"
pip install \
  "hydra-core>=1.3" \
  "omegaconf>=2.3" \
  "numpy>=1.23" \
  "scipy>=1.10" \
  "transformers==5.1.0" \
  "urllib3>=2.6.3" \
  boto3 \
  "peft>=0.18" \
  "einops>=0.7" \
  "tqdm>=4.0" \
  "packaging>=21.0" \
  "pydantic>=2.0" \
  "filelock>=3.20.3" \
  "gradio>=6.8.0" \
  "gradio_client>=1.0" \
  "trimesh>=3.21.7" \
  "pillow>=9.0" \
  "av>=16.1.0" \
  bvhio

# SOMA-X is optional (demo/viz skinning). Requires usd-core — not published for Linux aarch64.
if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
  echo "Skipping SOMA-X on aarch64 (usd-core unavailable; text-to-motion inference does not need it)."
else
  echo "Installing SOMA-X (optional viz)…"
  pip install "py-soma-x @ git+https://github.com/NVlabs/SOMA-X.git" || \
    echo "WARN: SOMA-X install failed — Kimodo inference still works without it."
fi

python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("kimodo")
if spec is None:
    raise SystemExit("Kimodo import failed after install")
print("Kimodo import OK:", spec.origin)
PY

echo ""
echo "Done. Kimodo installed in ${VENV}"
echo "For 3DAIGC-API workers, ensure PYTHONPATH includes ${TARGET} or use the same venv."
echo "Tip: export TEXT_ENCODER_DEVICE=cpu to reduce VRAM during generation."
echo "Default HF model: Kimodo-SOMA-RP-v1.1 (see docs/MODEL_LICENSES.md — SMPL-X variant is BLOCKED)."
echo "On aarch64, motion post-processing is disabled at runtime (MotionCorrection is x86-only)."
