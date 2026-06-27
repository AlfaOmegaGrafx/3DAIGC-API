#!/usr/bin/env bash
# Pre-download Kimodo dependencies (Llama-3-8B + Kimodo weights) into HF cache.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
source .env 2>/dev/null || true
set +a
exec venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
repos = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "nvidia/Kimodo-SOMA-RP-v1.1",
]
for repo in repos:
    print(f"Downloading {repo} ...", flush=True)
    path = snapshot_download(repo, token=token)
    print(f"  -> {path}", flush=True)
print("Kimodo prefetch complete.", flush=True)
PY
