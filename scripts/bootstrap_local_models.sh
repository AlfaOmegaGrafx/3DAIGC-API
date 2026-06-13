#!/usr/bin/env bash
# One-shot: thirdparty submodules + pip extras for local venv (aarch64-friendly where wheels exist).
# Does NOT replace Docker/conda for Kaolin / spconv / torchsparse — see comments at end.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/venv/bin/pip" ]]; then
  echo "Expected venv at $ROOT/venv (python3 -m venv venv)" >&2
  exit 1
fi

echo "[bootstrap] git submodule update --init --recursive"
git submodule update --init --recursive

echo "[bootstrap] pip install -r requirements-models-extra.txt"
"$ROOT/venv/bin/pip" install -r "$ROOT/requirements-models-extra.txt"

echo "[bootstrap] utils3d (TRELLIS pin, --no-deps to avoid PyPI open3d / heavy GL stacks)"
"$ROOT/venv/bin/pip" install --no-deps \
  "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

echo ""
echo "Done. Remaining native / platform-specific steps (pick what you use):"
echo "  - One-shot (venv + CUDA 12.x on disk):  ./scripts/install_local_venv_extras.sh"
echo "  - Shell session (PATH, CUDA_HOME, CUMM_CUDA_ARCH_LIST, CPATH):  source ./scripts/env_local_gpu.sh"
echo "  1) TRELLIS sparse conv: install CUDA 12.x toolkit matching PyTorch cu12x, then either:"
echo "       export CUDA_HOME=/usr/local/cuda-12.8   # example"
echo "       ./scripts/build_spconv_local.sh"
echo "     OR build torchsparse with: pip install --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git"
echo "     Then either leave SPARSE_BACKEND=spconv (default) or export SPARSE_BACKEND=torchsparse."
echo "  2) TRELLIS FlexiCubes / mesh decode needs NVIDIA Kaolin for your exact torch+CUDA+arch — use the"
echo "     official Kaolin install instructions or the Docker image in docker-compose when wheels are missing."
echo "  3) PartField / UniRig / Hunyuan: follow sections in scripts/install.sh for heavy deps (torch-scatter, etc.)."
