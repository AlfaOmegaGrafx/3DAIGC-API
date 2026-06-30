#!/usr/bin/env bash
# Verify every enabled model in config/models.yaml (config-driven via verify_profiles.yaml).
# One subprocess per infer job so VRAM is freed between runs. Logs: logs/verify_all/
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs/verify_all"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"
: >"$SUMMARY"

# shellcheck source=scripts/env_local_gpu.sh
source "$ROOT/scripts/env_local_gpu.sh"
export SPARSE_BACKEND=spconv SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=sdpa \
       XFORMERS_DISABLED=1 SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=0 \
       TORCH_CUDA_ARCH_LIST="9.0+PTX" PYOPENGL_PLATFORM=egl TQDM_DISABLE=1

echo "=== Preflight ===" | tee "$LOG_DIR/preflight.log"
./venv/bin/python scripts/verify_registry.py --validate 2>&1 | tee -a "$LOG_DIR/preflight.log"
./venv/bin/python scripts/verify_env_compat.py 2>&1 | tee -a "$LOG_DIR/preflight.log"
if ! grep -q PREFLIGHT_OK "$LOG_DIR/preflight.log"; then
  echo "Aborting: fix preflight errors first." | tee -a "$SUMMARY"
  exit 1
fi
if ! grep -q REGISTRY_VALIDATE_OK "$LOG_DIR/preflight.log"; then
  echo "Aborting: fix verify registry (enabled model missing profile)." | tee -a "$SUMMARY"
  exit 1
fi

models_cfg="$(./venv/bin/python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(open(Path("config/models.yaml")))
ids = []
for feat, models in cfg.items():
    if not isinstance(models, dict):
        continue
    for mid, spec in models.items():
        if isinstance(spec, dict) and spec.get("enabled", True):
            ids.append(mid)
print(" ".join(sorted(ids)))
PY
)"

PASS=0
FAIL=0
for model_id in $models_cfg; do
  echo "" | tee -a "$SUMMARY"
  echo "================ $model_id ================" | tee -a "$SUMMARY"
  if ./venv/bin/python scripts/verify_registry.py --tier infer --model "$model_id" \
      > "$LOG_DIR/${model_id}.log" 2>&1; then
    grep -E "VERIFY_REGISTRY_OK|infer OK" "$LOG_DIR/${model_id}.log" | tail -3 | tee -a "$SUMMARY"
    echo "[$model_id] PASS" | tee -a "$SUMMARY"
    PASS=$((PASS+1))
  else
    echo "[$model_id] FAIL" | tee -a "$SUMMARY"
    grep -aE "FAIL|Error|Exception|Traceback|VERIFY" "$LOG_DIR/${model_id}.log" | tail -8 | tee -a "$SUMMARY"
    FAIL=$((FAIL+1))
  fi
done

echo "" | tee -a "$SUMMARY"
echo "VERIFY_ALL_DONE pass=$PASS fail=$FAIL" | tee -a "$SUMMARY"
[[ "$FAIL" -eq 0 ]]
