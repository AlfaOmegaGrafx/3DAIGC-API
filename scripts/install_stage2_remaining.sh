#!/usr/bin/env bash
# Stage-2 remaining deps for the aarch64 / sm_121 / cu128 venv.
# Groups are independent (|| true) so one failure does not abort the rest.
# Run after core pure-python deps; from repo root with env sourced.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"
PIP="$ROOT/venv/bin/pip"

echo "###### A: pure-python per-model deps ######"
$PIP install -q pytz timm meshiki fpsample pymcubes numba scikit-learn seaborn pysdf sentencepiece kiui pyrender mesh2sdf tetgen fast-simplification || echo "!! group A had failures"

echo "###### B: open3d + pymeshlab (aarch64 wheels) ######"
$PIP install -q open3d || echo "!! open3d failed (aarch64 wheel may be unavailable)"
$PIP install -q pymeshlab || echo "!! pymeshlab failed (aarch64 wheel may be unavailable)"

echo "###### C: PyG torch-scatter / torch-cluster (source build vs cu128) ######"
$PIP install -q --no-build-isolation torch-scatter torch-cluster \
  -f "https://data.pyg.org/whl/torch-2.11.0+cu128.html" || echo "!! PyG build failed"

echo "###### D: nvdiffrast (TRELLIS texture baking) ######"
$PIP install -q "git+https://github.com/NVlabs/nvdiffrast.git" || echo "!! nvdiffrast failed"

echo "###### E: cubvh (UltraShape / VoxHammer) ######"
$PIP install -q "git+https://github.com/ashawkey/cubvh" --no-build-isolation || echo "!! cubvh failed"

echo "STAGE2_REMAINING_DONE"
