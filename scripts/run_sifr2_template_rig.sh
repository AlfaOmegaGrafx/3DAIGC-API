#!/usr/bin/env bash
# Rig a textured humanoid GLB using the template VRM (template.vrm).
# Usage: ./scripts/run_sifr2_template_rig.sh /path/to/your_humanoid.glb
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env_local_gpu.sh" 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <textured_humanoid.glb> [output.glb]"
  exit 1
fi

INPUT="$(readlink -f "$1")"
if [[ ! -f "$INPUT" ]]; then
  echo "Input not found: $INPUT"
  exit 1
fi

BASENAME="$(basename "${INPUT%.*}")"
OUTPUT="${2:-$ROOT/outputs/rigged/template_rig_${BASENAME}.glb}"

"$ROOT/venv/bin/python" << PY
from pathlib import Path
from core.utils.format_utils import apply_humanoid_template_rig
from core.utils.humanoid_template import get_template

spec = get_template("template")
out = apply_humanoid_template_rig(str(spec.vrm_path), "$INPUT", "$OUTPUT")
print("RIGGED_GLB", out)
PY

echo "Done: $OUTPUT"
