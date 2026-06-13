#!/usr/bin/env python3
"""Preflight: CUDA toolkit, spconv, torch, and enabled-model prerequisites."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.utils.gpu_env import apply_local_gpu_env

import yaml


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL {msg}")


def warn(msg: str) -> None:
    print(f"  WARN {msg}")


def main() -> int:
    print("=== GPU / CUDA environment ===")
    applied = apply_local_gpu_env(ROOT)
    if applied:
        print(f"  Applied env: {applied}")

    cuda_home = os.environ.get("CUDA_HOME", "")
    default_cuda = "/usr/local/cuda"
    if Path(default_cuda).is_symlink():
        target = os.path.realpath(default_cuda)
        if target != cuda_home:
            warn(f"/usr/local/cuda -> {target} (workers use CUDA_HOME={cuda_home or 'unset'})")

    nvcc = Path(cuda_home) / "bin" / "nvcc" if cuda_home else None
    if nvcc and nvcc.is_file():
        ver = subprocess.check_output([str(nvcc), "--version"], text=True)
        rel = next((l for l in ver.splitlines() if "release" in l), ver.splitlines()[0])
        ok(f"nvcc ({cuda_home}): {rel.strip()}")
        if "release 13" in rel or "release 13." in rel:
            fail("CUDA 13 nvcc breaks spconv/TRELLIS; need CUDA 12.x via CUDA_HOME")
    else:
        fail("CUDA 12.x nvcc not found; run scripts/build_spconv_local.sh after installing cuda-12.8")

    try:
        import torch

        ok(f"torch {torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    except Exception as e:
        fail(f"torch: {e}")

    try:
        import spconv.pytorch as sp

        ok(f"spconv import ({sp.__file__})")
    except Exception as e:
        fail(f"spconv: {e}")

    print("\n=== Enabled models: paths & imports ===")
    with open(ROOT / "config" / "models.yaml") as f:
        cfg = yaml.safe_load(f)

    errors = 0
    for feature, models in cfg.items():
        if not isinstance(models, dict):
            continue
        for model_id, m in models.items():
            if not isinstance(m, dict):
                continue
            enabled = m.get("enabled", True)
            mp = m.get("model_path", "")
            tag = "enabled" if enabled else "disabled"
            path_ok = True
            if mp.startswith("pip:"):
                pkg = mp.split(":", 1)[1]
                try:
                    importlib.import_module(pkg)
                    ok(f"[{tag}] {model_id}: pip:{pkg}")
                except Exception as e:
                    fail(f"[{tag}] {model_id}: pip:{pkg} — {e}")
                    if enabled:
                        errors += 1
                continue
            p = ROOT / mp
            if not p.exists():
                fail(f"[{tag}] {model_id}: missing {mp}")
                if enabled:
                    errors += 1
                continue
            ok(f"[{tag}] {model_id}: {mp}")

    print("\n=== Summary ===")
    if errors:
        print(f"PREFLIGHT_FAIL ({errors} enabled-model prerequisite errors)")
        return 1
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
