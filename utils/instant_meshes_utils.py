"""
Instant Meshes CLI wrapper (BSD-3-Clause).

Batch mode: ``instant-meshes -o out.obj -v <count> input.obj``
See https://github.com/wjakob/instant-meshes
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

_REPO_REL = Path("thirdparty/instant-meshes")
_BUILD_CANDIDATES = (
    _REPO_REL / "build" / "Instant Meshes",
    _REPO_REL / "build" / "instant-meshes",
    _REPO_REL / "instant-meshes",
)


def find_instant_meshes_binary() -> Optional[Path]:
    """Resolve Instant Meshes executable (env, thirdparty build, or PATH)."""
    env_bin = os.environ.get("INSTANT_MESHES_BIN")
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.is_file():
            return p.resolve()

    root = Path(__file__).resolve().parent.parent
    for candidate in _BUILD_CANDIDATES:
        p = (root / candidate).resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return p

    which = shutil.which("instant-meshes")
    if which:
        return Path(which).resolve()
    return None


def run_instant_meshes(
    mesh_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    target_vertex_count: Optional[int] = None,
    target_face_count: Optional[int] = None,
    threads: Optional[int] = None,
    smooth_iterations: int = 2,
    deterministic: bool = False,
    timeout_sec: int = 3600,
) -> Path:
    """
    Run Instant Meshes in batch mode and write ``output_path``.

    At least one of ``target_vertex_count`` or ``target_face_count`` is required.
    """
    binary = find_instant_meshes_binary()
    if binary is None:
        raise FileNotFoundError(
            "Instant Meshes binary not found. Build with "
            "./scripts/install_instant_meshes.sh or set INSTANT_MESHES_BIN."
        )

    mesh_path = Path(mesh_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if target_vertex_count is None and target_face_count is None:
        raise ValueError(
            "Instant Meshes requires target_vertex_count or target_face_count"
        )

    with tempfile.TemporaryDirectory(prefix="im_") as tmp:
        work_in = Path(tmp) / f"in{mesh_path.suffix}"
        work_out = Path(tmp) / f"out{output_path.suffix}"
        shutil.copy2(mesh_path, work_in)

        cmd = [str(binary), "-o", str(work_out), "-S", str(smooth_iterations)]
        if deterministic:
            cmd.append("-d")
        if threads is not None and threads > 0:
            cmd.extend(["-t", str(threads)])
        if target_vertex_count is not None:
            cmd.extend(["-v", str(int(target_vertex_count))])
        if target_face_count is not None:
            cmd.extend(["-f", str(int(target_face_count))])
        cmd.append(str(work_in))

        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")

        logger.info("Running Instant Meshes: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(binary.parent),
            env=env,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            raise RuntimeError(
                f"Instant Meshes failed (exit {result.returncode}): "
                f"{stderr or stdout or 'no output'}"
            )

        if not work_out.is_file():
            raise RuntimeError(
                f"Instant Meshes did not produce output at {work_out}"
            )

        shutil.copy2(work_out, output_path)

    logger.info("Instant Meshes wrote %s", output_path)
    return output_path


def get_instant_meshes_info() -> Dict[str, Any]:
    binary = find_instant_meshes_binary()
    return {
        "name": "instant_meshes",
        "license": "BSD-3-Clause",
        "binary": str(binary) if binary else None,
        "available": binary is not None,
    }
