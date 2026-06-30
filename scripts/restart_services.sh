#!/usr/bin/env bash
# Clean restart: stop (with drain) then start detached API + scheduler.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash "$ROOT/scripts/stop_services.sh" "$@"
sleep 2

if [[ "${P3D_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "Preflight (HF / DINOv3 conditioning)..."
  ./venv/bin/python scripts/verify_hf_conditioning.py
fi

bash "$ROOT/scripts/start_services_detached.sh"
