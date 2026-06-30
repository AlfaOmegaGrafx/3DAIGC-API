#!/usr/bin/env bash
# Push 3DAIGC-API (DGX) -> Surface 3DAIGC clone. Protected manifest paths are DGX-canonical.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SURFACE_SSH="${SURFACE_SSH:-Surface-PC-Tailscale}"
SURFACE_ROOT="${SURFACE_ROOT:-C:/Users/alfao/Documents/GitHub/3DAIGC}"
MANIFEST="$ROOT/scripts/protected-paths.manifest"

# shellcheck source=scripts/protected-paths-lib.sh
source "$ROOT/scripts/protected-paths-lib.sh"
protected_load_manifest "$MANIFEST"

CRUFT_RE='^(\.claude/|thirdparty/)'

echo "=== Reconcile 3DAIGC-API: DGX -> Surface ==="
echo "DGX:     $ROOT"
echo "Surface: ${SURFACE_SSH}:${SURFACE_ROOT}"
echo "Protected paths in manifest: ${#PROTECTED_PATHS[@]}"
echo ""

if [[ -x "$ROOT/venv/bin/python" ]]; then
  echo "Preflight HF/DINOv3..."
  "$ROOT/venv/bin/python" "$ROOT/scripts/verify_hf_conditioning.py" | tail -3
fi

echo ""
echo "Protected manifest paths..."
for rel in "${PROTECTED_PATHS[@]}"; do
  protected_push_to_surface "$ROOT" "$rel" "$SURFACE_SSH" "$SURFACE_ROOT"
done

echo ""
echo "Other git-changed + untracked WIP (non-protected, non-cruft)..."
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  protected_is_path "$rel" && continue
  [[ "$rel" =~ $CRUFT_RE ]] && continue
  [[ -f "$ROOT/$rel" ]] || continue
  protected_push_to_surface "$ROOT" "$rel" "$SURFACE_SSH" "$SURFACE_ROOT"
done < <(
  git status --porcelain -u | while IFS= read -r line; do protected_trim_git_path "$line" || true; done | sort -u
)

echo ""
echo "Protected checksum verify..."
mismatch=0
protected_verify_surface_checksums "$ROOT" "$SURFACE_SSH" "$SURFACE_ROOT" || mismatch=$?

if [[ "$mismatch" -gt 0 ]]; then
  echo "WARN: fixed $mismatch protected checksum mismatch(es)." >&2
fi

echo ""
echo "RECONCILE_3DAIGC_OK"
