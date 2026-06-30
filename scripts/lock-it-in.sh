#!/usr/bin/env bash
# Lock-it-in for 3DAIGC-API: HF verify + stage protected manifest + commit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MANIFEST="$ROOT/scripts/protected-paths.manifest"

msg="${1:-}"
if [[ "${1:-}" == "--message" ]]; then
  msg="${2:-}"
fi
[[ -z "$msg" ]] && msg="chore(lock-in): commit protected API state and verify tooling"

echo "=== Lock it in (3DAIGC-API) ==="

if [[ -x "$ROOT/venv/bin/python" ]]; then
  "$ROOT/venv/bin/python" "$ROOT/scripts/verify_hf_conditioning.py" | tail -3
fi

# shellcheck source=scripts/protected-paths-lib.sh
source "$ROOT/scripts/protected-paths-lib.sh"
protected_load_manifest "$MANIFEST"

to_stage=()
for rel in "${PROTECTED_PATHS[@]}"; do
  [[ -e "$ROOT/$rel" ]] || continue
  if ! git diff --quiet HEAD -- "$rel" 2>/dev/null || ! git ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
    to_stage+=("$rel")
  fi
done

for rel in \
  scripts/lock-it-in.sh \
  scripts/reconcile-api-to-surface.sh \
  scripts/protected-paths-lib.sh \
  scripts/protected-paths.manifest; do
  [[ -e "$ROOT/$rel" ]] && to_stage+=("$rel")
done

mapfile -t to_stage < <(printf '%s\n' "${to_stage[@]}" | sort -u)

if [[ ${#to_stage[@]} -eq 0 ]]; then
  echo "Nothing to commit — protected paths already match HEAD."
  exit 0
fi

echo "Staging ${#to_stage[@]} path(s)..."
for rel in "${to_stage[@]}"; do
  if git check-ignore -q "$rel" 2>/dev/null; then
    echo "  skip gitignored: $rel"
    continue
  fi
  git add -- "$rel"
  echo "  + $rel"
done

git commit -m "$(cat <<EOF
$msg

Protected manifest paths, rules, and HF verify tooling — lock-in requires git commit.
EOF
)"

git status --short
echo "LOCK_IT_IN_OK"
