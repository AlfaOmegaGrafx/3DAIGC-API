#!/usr/bin/env bash
# Start DGX dev services after reboot (API + LingBot check + optional MSF + optional XR voice).
#
# Default: ensure LingBot-Map weights/package, then 3DAIGC-API (Redis + scheduler + :7842)
# and MSF Map Service when rp1.env exists. LingBot is in-process (not a separate port).
#
# Usage:
#   bash scripts/start-dgx-after-reboot.sh
#   bash scripts/start-dgx-after-reboot.sh --with-xr
#   bash scripts/start-dgx-after-reboot.sh --with-routing
#   bash scripts/start-dgx-after-reboot.sh --with-xr --with-routing funnel
#   bash scripts/start-dgx-after-reboot.sh --api-only
#   bash scripts/start-dgx-after-reboot.sh --force
#
# Surface (not DGX): npm run dev && npm run dev:spark-proxies
set -euo pipefail
exec bash "$(cd "$(dirname "$0")" && pwd)/ensure-spark-dev-services.sh" "$@"
