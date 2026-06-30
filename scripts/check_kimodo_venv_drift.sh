#!/usr/bin/env bash
# Drift check for Kimodo isolated venv (.venv-kimodo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIP="${ROOT}/.venv-kimodo/bin/pip"
CONSTRAINTS="${ROOT}/scripts/constraints-kimodo.txt"

if [[ ! -x "$PIP" ]]; then
  echo "WARN: .venv-kimodo not found — run: bash scripts/setup_kimodo.sh"
  exit 0
fi

echo "=== Kimodo venv drift check ==="
reqs=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"
  [[ -z "$line" ]] && continue
  reqs+=("$line")
done < "$CONSTRAINTS"

if ! "$PIP" install -c "$CONSTRAINTS" --dry-run "${reqs[@]}" >/tmp/kimodo_drift_dryrun.log 2>&1; then
  echo "  FAIL  Kimodo pip constraints not satisfied:"
  tail -15 /tmp/kimodo_drift_dryrun.log
  exit 1
fi
if grep -qE '^Would install ' /tmp/kimodo_drift_dryrun.log; then
  echo "  FAIL  Kimodo venv drift:"
  grep -E '^Would install ' /tmp/kimodo_drift_dryrun.log || true
  exit 1
fi
for spec in "${reqs[@]}"; do
  pkg="${spec%%[<>=!]*}"
  pkg="$(echo "$pkg" | xargs)"
  ver="$("$PIP" show "$pkg" 2>/dev/null | awk -F': ' '/^Version:/{print $2}')"
  echo "  OK    ${pkg}==${ver}"
done
echo "KIMODO_VENV_DRIFT_OK"
