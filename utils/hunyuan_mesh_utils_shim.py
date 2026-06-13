"""
Hunyuan paint pipeline mesh_utils without in-venv bpy.

Injects a bpy-free ``hy3dpaint.DifferentiableRenderer.mesh_utils`` module before the
real thirdparty module (which imports bpy at import time) is loaded.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

from utils.blender_runtime import find_blender_binary

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BLENDER_SCRIPT = _REPO_ROOT / "scripts" / "blender" / "hunyuan_obj_to_glb.py"
_SHIM_INSTALLED = False


def _run_blender_obj_to_glb(job: dict[str, Any]) -> None:
    binary = find_blender_binary()
    if binary is None:
        raise RuntimeError(
            "Hunyuan paint requires Blender for OBJ→GLB conversion. "
            "Install: sudo apt install -y blender"
        )
    if not _BLENDER_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing Blender helper: {_BLENDER_SCRIPT}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="hunyuan_job_"
    ) as f:
        json.dump(job, f)
        job_path = f.name

    env = os.environ.copy()
    env["HUNYUAN_JOB_JSON"] = job_path
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    cmd = [str(binary), "--background", "--python", str(_BLENDER_SCRIPT)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender OBJ→GLB failed (exit {result.returncode}): "
                f"{result.stderr[-2000:]}"
            )
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass


def convert_obj_to_glb(
    obj_path: str,
    glb_path: str,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
    try:
        _run_blender_obj_to_glb(
            {
                "obj_path": os.path.abspath(obj_path),
                "glb_path": os.path.abspath(glb_path),
                "shade_type": shade_type,
                "auto_smooth_angle": auto_smooth_angle,
                "merge_vertices": merge_vertices,
            }
        )
        return True
    except Exception as e:
        logger.error("convert_obj_to_glb failed: %s", e)
        return False


def _load_bpy_free_mesh_utils() -> types.ModuleType:
    """Load mesh_utils helpers that do not require bpy."""
    path = (
        _REPO_ROOT
        / "thirdparty"
        / "Hunyuan3D-2.1"
        / "hy3dpaint"
        / "DifferentiableRenderer"
        / "mesh_utils.py"
    )
    source = path.read_text(encoding="utf-8")
    source = "\n".join(
        line for line in source.splitlines() if line.strip() != "import bpy"
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, str(path), "exec"), namespace)
    mod = types.ModuleType("hy3dpaint.DifferentiableRenderer.mesh_utils")
    for name in (
        "load_mesh",
        "save_mesh",
        "save_obj_mesh",
        "_save_texture_map",
        "_create_mtl_file",
    ):
        if name in namespace:
            setattr(mod, name, namespace[name])
    mod.convert_obj_to_glb = convert_obj_to_glb
    return mod


def _ensure_hy3dpaint_packages() -> None:
    """Register hy3dpaint namespace packages so sibling modules remain importable."""
    hy3d_root = _REPO_ROOT / "thirdparty" / "Hunyuan3D-2.1" / "hy3dpaint"
    dr_root = hy3d_root / "DifferentiableRenderer"

    if "hy3dpaint" not in sys.modules:
        hy3d_pkg = types.ModuleType("hy3dpaint")
        hy3d_pkg.__path__ = [str(hy3d_root)]
        sys.modules["hy3dpaint"] = hy3d_pkg

    if "hy3dpaint.DifferentiableRenderer" not in sys.modules:
        dr_pkg = types.ModuleType("hy3dpaint.DifferentiableRenderer")
        dr_pkg.__path__ = [str(dr_root)]
        sys.modules["hy3dpaint.DifferentiableRenderer"] = dr_pkg


def install_hunyuan_mesh_utils_shim() -> None:
    """Pre-register bpy-free mesh_utils for Hunyuan paint imports."""
    global _SHIM_INSTALLED
    if _SHIM_INSTALLED:
        return
    try:
        import bpy  # noqa: F401

        _SHIM_INSTALLED = True
        return
    except ImportError:
        pass

    _ensure_hy3dpaint_packages()
    mod = _load_bpy_free_mesh_utils()
    sys.modules["hy3dpaint.DifferentiableRenderer.mesh_utils"] = mod
    _SHIM_INSTALLED = True
    logger.info("Hunyuan mesh_utils using Blender subprocess (no in-venv bpy)")
