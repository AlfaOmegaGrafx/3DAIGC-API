#!/usr/bin/env bash
# Weekly / post-release model health (DGX). Quick tier skips heavy GPU inference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${ROOT}/logs/weekly_model_health.log"
mkdir -p "${ROOT}/logs"

{
  echo "=== weekly_model_health $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  bash scripts/check_venv_drift.sh
  bash scripts/check_kimodo_venv_drift.sh
  ./venv/bin/python scripts/verify_registry.py --validate
  ./venv/bin/python scripts/verify_env_compat.py
  ./venv/bin/python scripts/verify_registry.py --tier quick --all-enabled
  echo ""
  echo "Optional full GPU HF check:"
  echo "  P3D_HF_VERIFY_GPU=1 ./venv/bin/python scripts/verify_hf_conditioning.py --gpu"
  echo "Optional light infer matrix (skips heavy models):"
  echo "  ./venv/bin/python scripts/verify_registry.py --tier infer --all-enabled --skip-heavy"
  echo "Optional full infer matrix:"
  echo "  bash scripts/verify_all_enabled_models.sh"
  echo "WEEKLY_MODEL_HEALTH_OK"
} 2>&1 | tee "$LOG"
