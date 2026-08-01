#!/usr/bin/env bash
# Compare installed pip versions against scripts/constraints-hf.txt.
# Exit 1 if pip would need to change HF stack packages to satisfy constraints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIP="${ROOT}/venv/bin/pip"
CONSTRAINTS_HF="${ROOT}/scripts/constraints-hf.txt"
CONSTRAINTS_RT="${ROOT}/scripts/constraints-models-runtime.txt"
LOCK="${ROOT}/constraints/venv-lock-hf.txt"

if [[ ! -x "$PIP" ]]; then
  echo "FAIL: venv not found at ${ROOT}/venv" >&2
  exit 1
fi

echo "=== HF constraint drift check ==="

# Dry-run reinstall of pinned specs; fails if installed versions violate constraints.
reqs=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"
  [[ -z "$line" ]] && continue
  reqs+=("$line")
done < "$CONSTRAINTS_HF"
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"
  [[ -z "$line" ]] && continue
  reqs+=("$line")
done < "$CONSTRAINTS_RT"

if ! "$PIP" install -c "$CONSTRAINTS_HF" -c "$CONSTRAINTS_RT" --dry-run "${reqs[@]}" >/tmp/venv_drift_dryrun.log 2>&1; then
  echo "  FAIL  pip constraints not satisfied:"
  tail -20 /tmp/venv_drift_dryrun.log
  echo ""
  echo "VENV_DRIFT_FAIL"
  echo "Fix: bash scripts/pip_main_venv.sh install -r requirements.txt"
  exit 1
fi

# Dry-run can succeed while still planning upgrades/downgrades — treat that as drift.
if grep -qE '^Would install ' /tmp/venv_drift_dryrun.log; then
  echo "  FAIL  installed HF stack does not match constraints:"
  grep -E '^Would install ' /tmp/venv_drift_dryrun.log || true
  echo ""
  echo "VENV_DRIFT_FAIL"
  echo "Fix: bash scripts/pip_main_venv.sh install -r requirements.txt"
  exit 1
fi

for spec in "${reqs[@]}"; do
  pkg="${spec%%[<>=!]*}"
  pkg="$(echo "$pkg" | xargs)"
  ver="$("$PIP" show "$pkg" 2>/dev/null | awk -F': ' '/^Version:/{print $2}')"
  echo "  OK    ${pkg}==${ver}"
done

if [[ -f "$LOCK" ]]; then
  echo ""
  echo "=== Lock snapshot (informational) ==="
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^# ]] && continue
    line="$(echo "$line" | xargs)"
    [[ -z "$line" ]] && continue
    pkg="${line%%==*}"
    want="${line#*==}"
    got="$("$PIP" show "$pkg" 2>/dev/null | awk -F': ' '/^Version:/{print $2}')"
    if [[ -n "$want" && -n "$got" && "$got" != "$want" ]]; then
      echo "  WARN  $pkg lock=$want installed=$got"
    fi
  done < "$LOCK"
fi

echo ""
echo "VENV_DRIFT_OK"
