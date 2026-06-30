#!/usr/bin/env bash
# Guarded pip for main 3DAIGC-API venv — always applies HF + runtime constraints.
# Usage: bash scripts/pip_main_venv.sh install <packages…>
#        bash scripts/pip_main_venv.sh install -r requirements.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIP="${ROOT}/venv/bin/pip"
HF="${ROOT}/scripts/constraints-hf.txt"
RT="${ROOT}/scripts/constraints-models-runtime.txt"

if [[ ! -x "$PIP" ]]; then
  echo "FAIL: venv not found at ${ROOT}/venv" >&2
  exit 1
fi

"$PIP" install -c "$HF" -c "$RT" "$@"
echo ""
echo "=== post-pip guard ==="
bash "${ROOT}/scripts/post_pip_guard.sh"
