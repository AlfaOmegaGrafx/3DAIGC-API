#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env_local_gpu.sh" 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$ROOT"
exec "$ROOT/venv/bin/python" -m pytest tests/test_humanoid_template.py -q "$@"
