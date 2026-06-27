#!/usr/bin/env bash
# Clean restart: stop (with drain) then start detached API + scheduler.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash "$ROOT/scripts/stop_services.sh" "$@"
sleep 2
bash "$ROOT/scripts/start_services_detached.sh"
