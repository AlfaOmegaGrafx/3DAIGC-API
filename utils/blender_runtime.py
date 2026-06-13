"""
Run UniRig Blender steps via the system ``blender`` binary when ``bpy`` is not
importable in the API venv (typical on Linux aarch64 / DGX Spark).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTRACT_SCRIPT = _REPO_ROOT / "scripts" / "blender" / "unirig_extract.py"
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "blender" / "unirig_export_fbx.py"


def find_blender_binary() -> Optional[Path]:
    """Resolve Blender executable (BLENDER_BIN, PATH, common locations)."""
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin:
        p = Path(env_bin).expanduser()
        if p.is_file():
            return p.resolve()

    for candidate in (
        shutil.which("blender"),
        "/usr/bin/blender",
        "/snap/bin/blender",
    ):
        if candidate:
            p = Path(candidate)
            if p.is_file() and os.access(p, os.X_OK):
                return p.resolve()
    return None


def bpy_importable() -> bool:
    try:
        import bpy  # noqa: F401

        return True
    except ImportError:
        return False


def require_blender_or_bpy() -> None:
    if bpy_importable():
        return
    if find_blender_binary() is None:
        raise RuntimeError(
            "UniRig requires Blender. On aarch64 install with: "
            "sudo apt install -y blender  "
            "Then set BLENDER_BIN if needed, or ensure `blender` is on PATH."
        )


def _run_blender_script(
    script: Path,
    job: Dict[str, Any],
    *,
    timeout_sec: int = 3600,
) -> None:
    binary = find_blender_binary()
    if binary is None:
        raise FileNotFoundError(
            "Blender binary not found. Install: sudo apt install -y blender"
        )
    if not script.is_file():
        raise FileNotFoundError(f"Blender helper script missing: {script}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="unirig_job_"
    ) as f:
        json.dump(job, f)
        job_path = f.name

    env = os.environ.copy()
    env["UNIRIG_JOB_JSON"] = job_path
    env["DAIGC_ROOT"] = str(_REPO_ROOT)
    env.setdefault("BLENDER_USER_SCRIPTS", "")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    cmd = [
        str(binary),
        "--background",
        "--python",
        str(script),
    ]
    logger.info("Running Blender subprocess: %s (job %s)", binary, script.name)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=str(_REPO_ROOT),
        )
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(
            f"Blender subprocess failed (exit {result.returncode}): "
            f"{stderr or stdout or 'no output'}"
        )
    if result.stdout:
        logger.debug("Blender stdout: %s", result.stdout[-2000:])


def run_extract_builtin(
    *,
    output_folder: str,
    target_count: int,
    num_runs: int,
    job_id: int,
    time: str,
    files: List[Tuple[str, str]],
) -> None:
    """Run UniRig mesh preprocessing (``extract_builtin``)."""
    if bpy_importable():
        from thirdparty.UniRig.src.data.extract import extract_builtin

        extract_builtin(
            output_folder=output_folder,
            target_count=target_count,
            num_runs=num_runs,
            id=job_id,
            time=time,
            files=files,
        )
        return

    require_blender_or_bpy()
    _run_blender_script(
        _EXTRACT_SCRIPT,
        {
            "op": "extract_builtin",
            "output_folder": output_folder,
            "target_count": target_count,
            "num_runs": num_runs,
            "id": job_id,
            "time": time,
            "files": [[a, b] for a, b in files],
        },
    )


def run_export_fbx(
    *,
    path: str,
    vertices: Any,
    joints: Any,
    skin: Any,
    parents: Any,
    names: Any,
    faces: Any,
    tails: Any = None,
    extrude_size: float = 0.03,
    group_per_vertex: int = -1,
    add_root: bool = False,
    do_not_normalize: bool = False,
    use_extrude_bone: bool = True,
    use_connect_unique_child: bool = True,
    extrude_from_parent: bool = True,
) -> None:
    """Run UniRig ``Exporter._export_fbx`` (FBX write with armature)."""
    if bpy_importable():
        from thirdparty.UniRig.src.data.exporter import Exporter

        Exporter()._export_fbx(
            path=path,
            vertices=vertices,
            joints=joints,
            skin=skin,
            parents=parents,
            names=names,
            faces=faces,
            tails=tails,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
        )
        return

    require_blender_or_bpy()

    def _to_list(x):
        if x is None:
            return None
        if hasattr(x, "tolist"):
            return x.tolist()
        return x

    payload_dir = Path(path).parent
    payload_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", delete=False, prefix="unirig_export_"
    ) as tmp:
        import numpy as np

        save_kw: Dict[str, Any] = {
            "vertices": vertices,
            "joints": joints,
            "skin": skin,
            "parents": np.array(parents, dtype=object) if parents is not None else None,
            "names": np.array(names, dtype=object) if names is not None else None,
            "faces": faces,
        }
        if tails is not None:
            save_kw["tails"] = tails
        np.savez(tmp.name, **{k: v for k, v in save_kw.items() if v is not None})
        payload_npz = tmp.name

    try:
        _run_blender_script(
            _EXPORT_SCRIPT,
            {
                "op": "export_fbx",
                "payload_npz": payload_npz,
                "path": os.path.abspath(path),
                "extrude_size": extrude_size,
                "group_per_vertex": group_per_vertex,
                "add_root": add_root,
                "do_not_normalize": do_not_normalize,
                "use_extrude_bone": use_extrude_bone,
                "use_connect_unique_child": use_connect_unique_child,
                "extrude_from_parent": extrude_from_parent,
            },
        )
    finally:
        try:
            os.unlink(payload_npz)
        except OSError:
            pass


def install_exporter_bpy_shim() -> None:
    """
    Patch UniRig ``Exporter._export_fbx`` to use Blender subprocess when bpy
    is missing in the API interpreter. Safe to call multiple times.
    """
    if bpy_importable():
        return

    from thirdparty.UniRig.src.data import exporter as exporter_mod

    if getattr(exporter_mod.Exporter._export_fbx, "_daigc_shim", False):
        return

    def _export_fbx_shim(
        self,
        path: str,
        vertices,
        joints,
        skin,
        parents,
        names,
        faces,
        tails=None,
        extrude_size: float = 0.03,
        group_per_vertex: int = -1,
        add_root: bool = False,
        do_not_normalize: bool = False,
        use_extrude_bone: bool = True,
        use_connect_unique_child: bool = True,
        extrude_from_parent: bool = True,
    ):
        run_export_fbx(
            path=path,
            vertices=vertices,
            joints=joints,
            skin=skin,
            parents=parents,
            names=names,
            faces=faces,
            tails=tails,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
        )

    _export_fbx_shim._daigc_shim = True  # type: ignore[attr-defined]
    exporter_mod.Exporter._export_fbx = _export_fbx_shim
    logger.info("UniRig Exporter._export_fbx using Blender subprocess (no in-venv bpy)")
