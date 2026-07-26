#!/usr/bin/env bash
# Ensure LingBot-Map is installed and weights are present for environment-scan jobs.
# LingBot is NOT a separate daemon — it loads inside 3DAIGC-API scheduler workers.
#
# Usage:
#   bash scripts/ensure_lingbot_map.sh           # verify only (exit 1 if missing)
#   bash scripts/ensure_lingbot_map.sh --install # install/update if missing
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
  esac
done

status_json() {
  ./venv/bin/python - <<'PY'
from core.utils.lingbot_map_pipeline import lingbot_map_status
import json
print(json.dumps(lingbot_map_status()))
PY
}

echo "=== ensure_lingbot_map ==="
json="$(status_json)"
echo "$json" | python3 -m json.tool

available="$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('available'))")"
weights="$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('weights_present'))")"

if [[ "$available" == "True" && "$weights" == "True" ]]; then
  echo "LingBot-Map OK (package + weights)"
  exit 0
fi

if [[ "$INSTALL" -eq 1 ]]; then
  echo "LingBot incomplete — running install_lingbot_map.sh …"
  bash "$ROOT/scripts/install_lingbot_map.sh"
  json="$(status_json)"
  echo "$json" | python3 -m json.tool
  available="$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('available'))")"
  weights="$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('weights_present'))")"
fi

if [[ "$available" != "True" || "$weights" != "True" ]]; then
  echo "ERROR: LingBot-Map not ready (available=$available weights_present=$weights)"
  echo "Run: bash scripts/install_lingbot_map.sh"
  exit 1
fi

echo "LingBot-Map OK after install"
