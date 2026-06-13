"""
Ensure CUDA 12.x + spconv/cumm build env in worker and verify subprocesses.

Mirrors scripts/env_local_gpu.sh so scheduler workers inherit correct nvcc even when
/usr/local/cuda symlinks to CUDA 13.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _detect_cuda12_home() -> Optional[str]:
    candidates = [
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-12",
        "/usr/lib/nvidia-cuda-toolkit",
    ]
    for d in candidates:
        nvcc = Path(d) / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        try:
            out = subprocess.check_output([str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            if "release" in line:
                parts = line.split("release", 1)[-1].strip().split(",", 1)[0].strip()
                if parts.startswith("12."):
                    return d
    return None


def apply_local_gpu_env(repo_root: Optional[Path] = None) -> dict[str, str]:
    """Set process env for PyTorch CUDA extension / spconv JIT builds. Idempotent."""
    applied: dict[str, str] = {}
    if repo_root is None:
        env_root = os.environ.get("DAIGC_ROOT")
        if env_root:
            repo_root = Path(env_root)
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent
    root = repo_root.resolve()
    os.environ.setdefault("DAIGC_ROOT", str(root))
    try:
        os.chdir(root)
    except OSError:
        pass

    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home or not Path(cuda_home, "bin", "nvcc").is_file():
        detected = _detect_cuda12_home()
        if detected:
            cuda_home = detected
            os.environ["CUDA_HOME"] = cuda_home
            applied["CUDA_HOME"] = cuda_home

    if cuda_home and Path(cuda_home, "bin", "nvcc").is_file():
        bin_dir = str(Path(cuda_home) / "bin")
        path = os.environ.get("PATH", "")
        if not path.startswith(bin_dir):
            os.environ["PATH"] = f"{bin_dir}:{path}"
            applied["PATH"] = os.environ["PATH"]

    arch = os.environ.get("CUMM_CUDA_ARCH_LIST")
    if not arch:
        arch = "7.5;8.0;8.6;8.9;9.0+PTX"
        os.environ["CUMM_CUDA_ARCH_LIST"] = arch
        applied["CUMM_CUDA_ARCH_LIST"] = arch

    venv_bin = root / "venv" / "bin"
    if venv_bin.is_dir():
        path = os.environ.get("PATH", "")
        vb = str(venv_bin)
        if vb not in path.split(":"):
            os.environ["PATH"] = f"{vb}:{path}"
            applied["PATH"] = os.environ["PATH"]

    py_root = str(root)
    pypath = os.environ.get("PYTHONPATH", "")
    if py_root not in pypath.split(":"):
        os.environ["PYTHONPATH"] = f"{py_root}{':' + pypath if pypath else ''}"

    # Blackwell / sm_121: TRELLIS dense attention -> PyTorch SDPA; DINOv2 avoids xformers fp32.
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    applied["ATTN_BACKEND"] = os.environ["ATTN_BACKEND"]
    applied["XFORMERS_DISABLED"] = os.environ["XFORMERS_DISABLED"]

    try:
        from utils.open3d_shim import install_open3d_shim

        install_open3d_shim()
    except Exception:
        pass

    return applied
