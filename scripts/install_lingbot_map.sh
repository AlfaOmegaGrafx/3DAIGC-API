#!/usr/bin/env bash
# Install LingBot-Map (optional) for Galaxy XR / walk-video environment scan.
# Does not change default image-to-world (TripoSplat).
#
# Usage (DGX):
#   cd /home/sifr/3DAIGC-API && bash scripts/install_lingbot_map.sh
#   bash scripts/install_lingbot_map.sh --weights-only   # after clone exists
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${LINGBOT_MAP_ROOT:-$ROOT/thirdparty/lingbot-map}"
WEIGHTS_DIR="${DEST}/checkpoints"
REPO_URL="${LINGBOT_MAP_REPO:-https://github.com/Robbyant/lingbot-map.git}"
HF_REPO="${LINGBOT_MAP_HF:-robbyant/lingbot-map}"

WEIGHTS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --weights-only) WEIGHTS_ONLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p "$ROOT/thirdparty"

if [[ "$WEIGHTS_ONLY" -eq 0 ]]; then
  if [[ -d "$DEST/.git" ]]; then
    echo "Updating existing clone at $DEST"
    git -C "$DEST" fetch --depth 1 origin main || true
    git -C "$DEST" pull --ff-only || true
  else
    echo "Cloning $REPO_URL → $DEST"
    git clone --depth 1 "$REPO_URL" "$DEST"
  fi

  if [[ -x "$ROOT/venv/bin/pip" ]]; then
    echo "Installing Python package (editable, io extras if present)…"
    "$ROOT/venv/bin/pip" install -e "${DEST}[io]" 2>/dev/null \
      || "$ROOT/venv/bin/pip" install -e "$DEST" \
      || echo "WARN: pip install failed — demo.py may still run with PYTHONPATH=$DEST"
  fi
fi

mkdir -p "$WEIGHTS_DIR"
echo "Downloading weights from Hugging Face ($HF_REPO) if available…"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$HF_REPO" --local-dir "$WEIGHTS_DIR" --local-dir-use-symlinks False || true
elif [[ -x "$ROOT/venv/bin/huggingface-cli" ]]; then
  "$ROOT/venv/bin/huggingface-cli" download "$HF_REPO" --local-dir "$WEIGHTS_DIR" --local-dir-use-symlinks False || true
else
  echo "Install huggingface_hub / huggingface-cli to fetch weights, or place checkpoint under:"
  echo "  $WEIGHTS_DIR"
fi

echo ""
echo "Done. Status check:"
cd "$ROOT"
./venv/bin/python - <<'PY'
from core.utils.lingbot_map_pipeline import lingbot_map_status
import json
print(json.dumps(lingbot_map_status(), indent=2))
PY
echo ""
echo "API: POST /api/v1/world-generation/environment-scan"
echo "Upload video: POST /api/v1/file-upload/video"
echo "1:1 scale: pass metric_calibration { mode, true_meters, recon_length|point_a/point_b }"
