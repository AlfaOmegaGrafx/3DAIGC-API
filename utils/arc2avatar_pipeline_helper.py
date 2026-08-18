"""
Helpers for Arc2Avatar SDS head training on DGX.

Upstream expects its own CUDA 11.8 / Python 3.9 env (RTX 4090 reference).
On GB200-class Spark, set ARC2AVATAR_PYTHON to a working interpreter after
you validate torch+CUDA there.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_ARC2AVATAR_ROOT = Path(
    os.environ.get(
        "ARC2AVATAR_ROOT",
        Path(__file__).resolve().parents[1] / "thirdparty" / "Arc2Avatar",
    )
)


def resolve_arc2avatar_python() -> Optional[str]:
    env = (os.environ.get("ARC2AVATAR_PYTHON") or "").strip()
    if env and Path(env).exists():
        return env
    venv_py = DEFAULT_ARC2AVATAR_ROOT / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return None


def _python_runtime_ok(python_bin: str) -> Tuple[bool, str]:
    """Require torch+CUDA and the local gaussian rasterization extension."""
    probe = (
        "import importlib, torch;"
        "assert torch.cuda.is_available(), 'torch.cuda unavailable';"
        "importlib.import_module('diff_gaussian_rasterization');"
        "print(torch.__version__)"
    )
    try:
        proc = subprocess.run(
            [python_bin, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return False, f"runtime probe failed: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "import failed").strip().splitlines()
        return False, err[-1] if err else "torch/CUDA/rasterizer import failed"
    return True, (proc.stdout or "").strip()


def probe_arc2avatar_install(root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(root or DEFAULT_ARC2AVATAR_ROOT)
    train_py = root / "train.py"
    config = root / "configs" / "config.yaml"
    weights_marker = root / "models" / "arc2face" / "diffusion_pytorch_model.safetensors"
    cuda_ext = root / "submodules" / "diff-gaussian-rasterization"
    python_bin = resolve_arc2avatar_python()
    reasons: List[str] = []
    runtime_note = ""
    if not train_py.is_file():
        reasons.append(f"missing train.py under {root}")
    if not config.is_file():
        reasons.append("missing configs/config.yaml")
    if not cuda_ext.is_dir():
        reasons.append("missing submodules/diff-gaussian-rasterization")
    if not weights_marker.is_file():
        reasons.append("Arc2Face weights missing — run: python download_models.py")
    if not python_bin:
        reasons.append(
            "set ARC2AVATAR_PYTHON or create thirdparty/Arc2Avatar/.venv "
            "(run scripts/install_arc2avatar_env.sh on GB10/CUDA 12.8)"
        )
    else:
        ok, runtime_note = _python_runtime_ok(python_bin)
        if not ok:
            reasons.append(
                "Arc2Avatar Python env incomplete — "
                f"{runtime_note}. Run: bash scripts/install_arc2avatar_env.sh"
            )
    return {
        "root": str(root),
        "python": python_bin,
        "runtime": runtime_note or None,
        "weights_ready": weights_marker.is_file(),
        "train_script": str(train_py) if train_py.is_file() else None,
        "integrated": len(reasons) == 0,
        "blocking_reasons": reasons,
        "upstream": "https://github.com/dimgerogiannis/Arc2Avatar",
        "license_note": "FLAME + Arc2Face terms — see docs/MODEL_LICENSES.md and docs/ARC2AVATAR_TRACK.md",
    }


def _latest_point_cloud_ply(splat_dir: Path) -> Optional[Path]:
    cloud_root = splat_dir / "point_cloud"
    if not cloud_root.is_dir():
        return None
    candidates = sorted(
        cloud_root.glob("iteration_*/point_cloud.ply"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def prepare_subject_dir(
    image_path: Path,
    work_root: Path,
    subject_name: str = "subject",
) -> Path:
    """Arc2Avatar expects a folder containing one face image; creates splat/ inside."""
    subject = work_root / subject_name
    if subject.exists():
        shutil.rmtree(subject)
    subject.mkdir(parents=True, exist_ok=True)
    ext = image_path.suffix.lower() or ".png"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".png"
    dest = subject / f"input{ext}"
    shutil.copy2(image_path, dest)
    return subject


def run_arc2avatar_train(
    image_path: Path,
    output_dir: Path,
    *,
    root: Optional[Path] = None,
    iterations: Optional[int] = None,
    batch_size: int = 4,
    timeout_s: int = 6 * 60 * 60,
) -> Dict[str, Any]:
    root = Path(root or DEFAULT_ARC2AVATAR_ROOT)
    status = probe_arc2avatar_install(root)
    if not status["integrated"]:
        raise RuntimeError(
            "Arc2Avatar not ready: " + "; ".join(status["blocking_reasons"])
        )

    python_bin = status["python"]
    assert python_bin
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject = prepare_subject_dir(Path(image_path), output_dir, "subject")

    cmd = [
        python_bin,
        "train.py",
        "--opt",
        "./configs/config.yaml",
        "--subject",
        str(subject.resolve()),
        "--batch_size",
        str(max(1, int(batch_size))),
    ]
    # Upstream OptimizationParams.iterations lives in yaml; optional override via env.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if iterations is not None:
        env["ARC2AVATAR_ITERATIONS"] = str(int(iterations))

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.time() - t0
    log_path = output_dir / "arc2avatar_train.log"
    log_path.write_text(
        (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""),
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Arc2Avatar train failed (exit {proc.returncode}). See {log_path}"
        )

    splat_dir = subject / "splat"
    ply = _latest_point_cloud_ply(splat_dir)
    if ply is None:
        raise FileNotFoundError(
            f"No point_cloud.ply under {splat_dir}/point_cloud/iteration_* after train"
        )

    exported = output_dir / "head_point_cloud.ply"
    shutil.copy2(ply, exported)
    return {
        "success": True,
        "subject_dir": str(subject),
        "splat_dir": str(splat_dir),
        "output_splat_path": str(exported),
        "source_ply": str(ply),
        "train_log": str(log_path),
        "elapsed_s": elapsed,
        "iterations": iterations,
    }
