#!/usr/bin/env bash
# Contract: LingBot gravity + 3DGS refine unit tests (orientation lock-in).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/venv/bin/python"
[[ -x "$PY" ]] || PY=python3
"$PY" -m pytest \
  tests/test_lingbot_map_pipeline_budget.py \
  tests/test_lingbot_3dgs_refine.py \
  tests/test_lingbot_3dgs_train.py \
  tests/test_metric_scale.py \
  -q --tb=line
echo "LINGBOT_ENV_SCAN_ORIENTATION_OK"
