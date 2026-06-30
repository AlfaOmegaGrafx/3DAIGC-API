#!/usr/bin/env bash
# Run after any pip install into main venv. Fails fast on HF/runtime drift.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== post_pip_guard: venv drift ==="
bash scripts/check_venv_drift.sh

echo ""
echo "=== post_pip_guard: HF conditioning (quick) ==="
./venv/bin/python scripts/verify_hf_conditioning.py

echo ""
echo "=== post_pip_guard: enabled adapter imports ==="
./venv/bin/python scripts/verify_registry.py --tier quick --all-enabled

echo ""
echo "POST_PIP_GUARD_OK"
