"""
Run VoxHammer bpy-dependent steps via Blender when ``bpy`` is not in the API venv.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from utils.blender_runtime import bpy_importable, find_blender_binary

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RENDER_SCRIPT = _REPO_ROOT / "scripts" / "blender" / "voxhammer_render.py"
_MASK_SCRIPT = _REPO_ROOT / "scripts" / "blender" / "voxhammer_voxel_mask.py"
_PRESET_VOXEL = _REPO_ROOT / "thirdparty" / "VoxHammer" / "assets" / "preset" / "preset_grid64.ply"


def _run_blender_step(script: Path, params: Dict[str, Any], *, timeout_sec: int = 7200) -> None:
    if bpy_importable():
        raise RuntimeError("_run_blender_step should not be called when bpy is importable")

    binary = find_blender_binary()
    if binary is None:
        raise RuntimeError(
            "VoxHammer requires Blender for 3D rendering. Install: sudo apt install -y blender"
        )
    if not script.is_file():
        raise FileNotFoundError(f"Missing Blender helper: {script}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="voxhammer_job_"
    ) as f:
        json.dump({"params": params}, f)
        job_path = f.name

    env = os.environ.copy()
    env["VOXHAMMER_JOB_JSON"] = job_path
    env["DAIGC_ROOT"] = str(_REPO_ROOT)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    cmd = [str(binary), "--background", "--python", str(script)]
    logger.info("Running VoxHammer Blender step: %s", script.name)
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
        detail = (result.stderr or result.stdout or "").strip()[-4000:]
        raise RuntimeError(
            f"VoxHammer Blender step failed (exit {result.returncode}): {detail or 'no output'}"
        )


def run_3d_rendering(input_model_path: str, render_dir: str, **render_kwargs) -> dict:
    if os.path.exists(os.path.join(render_dir, "transforms.json")) and os.path.exists(
        os.path.join(render_dir, "mesh.ply")
    ):
        return {
            "rendered": True,
            "num_views": render_kwargs.get("num_views", 150),
            "output_dir": render_dir,
            "transforms_file": os.path.join(render_dir, "transforms.json"),
            "mesh_file": os.path.join(render_dir, "mesh.ply"),
        }

    default_params = {
        "num_views": 150,
        "scale": 1.0,
        "offset": None,
        "resolution": 512,
        "engine": "CYCLES",
        "geo_mode": False,
        "split_normal": False,
        "save_mesh": True,
    }
    default_params.update(render_kwargs)
    params = {
        "file_path": os.path.abspath(input_model_path),
        "output_dir": os.path.abspath(render_dir),
        **default_params,
    }

    if bpy_importable():
        from voxhammer.bpy_render import render_3d_model

        return render_3d_model(**params)

    os.makedirs(render_dir, exist_ok=True)
    _run_blender_step(_RENDER_SCRIPT, params)
    mesh_file = os.path.join(render_dir, "mesh.ply")
    return {
        "rendered": True,
        "num_views": params["num_views"],
        "output_dir": render_dir,
        "transforms_file": os.path.join(render_dir, "transforms.json"),
        "mesh_file": mesh_file if os.path.isfile(mesh_file) else None,
    }


def run_feature_extraction(render_dir: str, **feature_kwargs) -> dict:
    from voxhammer.extract_feature import extract_features

    default_params = {"model": "dinov2_vitl14_reg", "batch_size": 10}
    default_params.update(feature_kwargs)
    extract_features(render_dir, **default_params)
    features_path = os.path.join(render_dir, "features.npz")
    return {"features_path": features_path}


def run_voxel_masking(mask_glb_path: str, render_dir: str, **mask_kwargs) -> dict:
    default_params = {"filter_method": "volume", "voxel_size": 1 / 64}
    default_params.update(mask_kwargs)
    params = {
        "mask_glb_path": os.path.abspath(mask_glb_path),
        "render_dir": os.path.abspath(render_dir),
        **default_params,
    }

    if bpy_importable():
        from voxhammer.delete_region_voxel import process_delete_ply

        process_delete_ply(
            params["mask_glb_path"],
            params["render_dir"],
            filter_method=params["filter_method"],
            voxel_size=params["voxel_size"],
        )
    else:
        if not _PRESET_VOXEL.is_file():
            raise FileNotFoundError(f"Missing VoxHammer preset voxels: {_PRESET_VOXEL}")
        _run_blender_step(_MASK_SCRIPT, params)

    voxels_delete_path = os.path.join(render_dir, "voxels_delete.ply")
    return {"mask_path": voxels_delete_path}


def run_3d_editing(
    pipeline,
    render_dir: str,
    output_path: str,
    image_dir: str,
    is_text: bool,
    source_prompt: str,
    target_prompt: str,
    **edit_kwargs,
) -> dict:
    from voxhammer.edit_pipeline import run_edit

    default_params = {"skip_step": 0, "re_init": False, "cfg": [5.0, 6.0, 0.0, 0.0]}
    default_params.update(edit_kwargs)

    required_files = [
        os.path.join(render_dir, "voxels.ply"),
        os.path.join(render_dir, "features.npz"),
        os.path.join(render_dir, "voxels_delete.ply"),
    ]
    if not is_text:
        required_files.extend(
            [
                os.path.join(image_dir, "2d_render.png"),
                os.path.join(image_dir, "2d_edit.png"),
                os.path.join(image_dir, "2d_mask.png"),
            ]
        )
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    run_edit(
        pipeline,
        render_dir,
        output_path,
        image_dir,
        is_text,
        source_prompt,
        target_prompt,
        **default_params,
    )
    return {"output_path": output_path}
