"""
Format conversion utilities.

This module provides utilities for converting between different 3D file formats
using Blender as the conversion engine.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_TEXTURE_SOURCE_SUFFIXES = {".glb", ".gltf", ".obj"}


def source_mesh_has_textures(mesh_path: str) -> bool:
    """Return True if the input mesh likely carries materials/textures."""
    path = Path(mesh_path)
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix not in _TEXTURE_SOURCE_SUFFIXES:
        return False
    if suffix == ".obj":
        mtl = path.with_suffix(".mtl")
        return mtl.is_file()
    if suffix in (".glb", ".gltf"):
        try:
            import struct

            data = path.read_bytes()
            if len(data) < 20:
                return False
            json_len = struct.unpack_from("<I", data, 12)[0]
            gltf = json.loads(data[20 : 20 + json_len])
            return bool(gltf.get("images") or gltf.get("textures"))
        except Exception:
            return True
    return False


def _read_gltf_json(glb_path: Path) -> dict:
    import struct

    data = glb_path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"Invalid GLB: {glb_path}")
    json_len = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + json_len])


def _write_gltf_json(glb_path: Path, gltf: dict) -> None:
    import struct

    data = glb_path.read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    bin_start = 20 + json_len
    bin_chunk = data[bin_start:]
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # 4-byte align JSON chunk per glTF spec
    pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad
    total_len = 12 + 8 + len(json_bytes) + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    json_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    glb_path.write_bytes(header + json_header + json_bytes + bin_chunk)


def _preserve_pbr_materials_from_source(source_glb: Path, output_glb: Path) -> None:
    """Copy glTF PBR material fields (metallic/roughness/texCoord) from source."""
    try:
        src = _read_gltf_json(source_glb)
        out = _read_gltf_json(output_glb)
    except Exception as exc:
        logger.warning("Could not patch PBR materials: %s", exc)
        return

    src_mats = src.get("materials") or []
    out_mats = out.get("materials") or []
    if not src_mats or not out_mats:
        return

    for i, src_mat in enumerate(src_mats):
        if i >= len(out_mats):
            break
        merged = dict(out_mats[i])
        src_pbr = src_mat.get("pbrMetallicRoughness") or {}
        out_pbr = dict(merged.get("pbrMetallicRoughness") or {})
        # Keep exported baseColorTexture (baked into output buffer) but restore PBR factors.
        for key in ("metallicFactor", "roughnessFactor", "baseColorFactor"):
            if key in src_pbr:
                out_pbr[key] = src_pbr[key]
        bct = out_pbr.get("baseColorTexture") or src_pbr.get("baseColorTexture")
        if bct:
            bct = dict(bct)
            if "texCoord" in src_pbr.get("baseColorTexture", {}):
                bct["texCoord"] = src_pbr["baseColorTexture"]["texCoord"]
            out_pbr["baseColorTexture"] = bct
        merged["pbrMetallicRoughness"] = out_pbr
        for key in ("normalTexture", "occlusionTexture", "emissiveFactor", "emissiveTexture", "alphaMode", "doubleSided"):
            if key in src_mat:
                merged[key] = src_mat[key]
        out_mats[i] = merged

    out["materials"] = out_mats
    _write_gltf_json(output_glb, out)


def merge_rigged_fbx_with_source_mesh(
    source_mesh_path: str,
    rig_fbx_path: str,
    output_glb_path: str,
    *,
    apply_skinning: bool = False,
) -> str:
    """
    Export rigged GLB with UniRig bones on the proxy mesh plus projected textures.

    UniRig skinning targets its remeshed ``character`` mesh, not the upload.
    This step keeps that mesh/armature intact and projects source UVs/materials
    onto it via world-space surface lookup (bones stay aligned with fbx_to_glb).
    """
    import subprocess

    from utils.blender_runtime import find_blender_binary

    source_mesh_path = str(Path(source_mesh_path).resolve())
    rig_fbx_path = str(Path(rig_fbx_path).resolve())
    output_glb_path = str(Path(output_glb_path).resolve())

    if not Path(source_mesh_path).is_file():
        raise FileNotFoundError(f"Source mesh not found: {source_mesh_path}")
    if not Path(rig_fbx_path).is_file():
        raise FileNotFoundError(f"Rig FBX not found: {rig_fbx_path}")

    output_path = Path(output_glb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blender_bin = find_blender_binary()
    if blender_bin is None:
        raise FileNotFoundError(
            "Blender executable not found for rig+texture merge. "
            "Install: sudo apt install -y blender"
        )

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "blender" / "merge_rig_textures.py"
    if not script.is_file():
        raise FileNotFoundError(f"Merge script missing: {script}")

    logger.info(
        "Merging rig FBX with textured source: %s + %s -> %s",
        source_mesh_path,
        rig_fbx_path,
        output_glb_path,
    )

    job = {
        "source_mesh": source_mesh_path,
        "rig_fbx": rig_fbx_path,
        "output_glb": output_glb_path,
        "apply_skinning": apply_skinning,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="merge_rig_job_"
    ) as job_file:
        json.dump(job, job_file)
        job_path = job_file.name

    env = os.environ.copy()
    env["MERGE_JOB_JSON"] = job_path
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        result = subprocess.run(
            [str(blender_bin), "--background", "--python", str(script)],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=str(repo_root),
        )
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass

    detail = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender rig+texture merge failed: {detail[-800:] if detail else 'no output'}"
        )
    if not output_path.exists():
        raise RuntimeError(
            f"Merged GLB was not created at: {output_glb_path}"
            + (f" — Blender: {detail[-500:]}" if detail else "")
        )

    logger.info("Merged rig FBX with textured source mesh: %s", output_glb_path)
    if source_mesh_path.lower().endswith((".glb", ".gltf")):
        _preserve_pbr_materials_from_source(Path(source_mesh_path), output_path)

    from core.utils.unirig_glb_checks import validate_unirig_merged_glb

    regressions = validate_unirig_merged_glb(source_mesh_path, output_path)
    if regressions:
        detail = "\n  - ".join(regressions)
        raise RuntimeError(
            "UniRig rig+texture merge produced a regressed GLB:\n  - " + detail
        )

    return output_glb_path


def _run_blender_script(
    script: Path,
    job: dict,
    *,
    job_env_key: str,
    ok_token: str,
    error_label: str,
) -> None:
    import subprocess

    from utils.blender_runtime import find_blender_binary

    blender_bin = find_blender_binary()
    if blender_bin is None:
        raise FileNotFoundError(
            f"Blender executable not found for {error_label}. "
            "Install: sudo apt install -y blender"
        )

    repo_root = Path(__file__).resolve().parent.parent.parent
    if not script.is_file():
        raise FileNotFoundError(f"Blender script missing: {script}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="blender_job_"
    ) as job_file:
        json.dump(job, job_file)
        job_path = job_file.name

    env = os.environ.copy()
    env[job_env_key] = job_path
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    venv_site = repo_root / "venv" / "lib" / "python3.12" / "site-packages"
    if venv_site.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(venv_site) + (f":{existing}" if existing else "")
        )

    try:
        result = subprocess.run(
            [str(blender_bin), "--background", "--python", str(script)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(repo_root),
        )
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass

    combined = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
    if result.returncode != 0 or ok_token not in combined:
        raise RuntimeError(
            f"{error_label} failed: {combined[-1000:] if combined else 'no output'}"
        )


def extract_vrm_skeleton_fbx(vrm_path: str, output_fbx_path: str) -> str:
    """Export VRM armature (+ skinned meshes) to FBX for skeleton reference."""
    vrm_path = str(Path(vrm_path).resolve())
    output_fbx_path = str(Path(output_fbx_path).resolve())
    if not Path(vrm_path).is_file():
        raise FileNotFoundError(f"VRM not found: {vrm_path}")
    Path(output_fbx_path).parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "blender" / "vrm_extract_skeleton.py"
    addon = repo_root / "thirdparty" / "UniRig" / "blender" / "add-on-vrm-v2.20.77_modified.zip"
    _run_blender_script(
        script,
        {"vrm_path": vrm_path, "output_fbx": output_fbx_path, "vrm_addon_zip": str(addon)},
        job_env_key="VRM_JOB_JSON",
        ok_token="VRM_EXTRACT_SKELETON_OK",
        error_label="VRM skeleton extract",
    )
    if not Path(output_fbx_path).is_file():
        raise RuntimeError(f"VRM skeleton FBX was not created: {output_fbx_path}")
    logger.info("Extracted VRM skeleton FBX: %s", output_fbx_path)
    return output_fbx_path


def apply_humanoid_template_rig(
    template_vrm_path: str,
    target_mesh_path: str,
    output_glb_path: str,
) -> Tuple[str, dict]:
    """
    Rig target mesh using a humanoid VRM template armature (bones-only path).

    Preserves target materials; does not transfer template blend shapes.
    Returns ``(output_glb_path, rig_info.validation dict)``.
    """
    template_vrm_path = str(Path(template_vrm_path).resolve())
    target_mesh_path = str(Path(target_mesh_path).resolve())
    output_glb_path = str(Path(output_glb_path).resolve())

    if not Path(template_vrm_path).is_file():
        raise FileNotFoundError(f"Template VRM not found: {template_vrm_path}")
    if not Path(target_mesh_path).is_file():
        raise FileNotFoundError(f"Target mesh not found: {target_mesh_path}")

    output_path = Path(output_glb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "blender" / "apply_humanoid_template_rig.py"
    addon = repo_root / "thirdparty" / "UniRig" / "blender" / "add-on-vrm-v2.20.77_modified.zip"
    from core.utils.template_rig_alignment import DEFAULT_ARMATURE_YAW_RAD

    _run_blender_script(
        script,
        {
            "template_vrm": template_vrm_path,
            "target_mesh": target_mesh_path,
            "output_glb": output_glb_path,
            "vrm_addon_zip": str(addon),
            "armature_yaw_rad": DEFAULT_ARMATURE_YAW_RAD,
        },
        job_env_key="TEMPLATE_RIG_JOB_JSON",
        ok_token="APPLY_HUMANOID_TEMPLATE_RIG_OK",
        error_label="Humanoid template rig",
    )
    if not output_path.exists():
        raise RuntimeError(f"Template-rigged GLB was not created: {output_glb_path}")

    if target_mesh_path.lower().endswith((".glb", ".gltf")):
        _preserve_pbr_materials_from_source(Path(target_mesh_path), output_path)

    from core.utils.humanoid_template_checks import validate_template_rigged_glb
    from core.utils.aigc_rig_contract import (
        format_contract_log,
        validate_aigc_rigged_glb,
    )

    regressions = validate_template_rigged_glb(target_mesh_path, output_glb_path)
    if regressions:
        detail = "\n  - ".join(regressions)
        raise RuntimeError(
            "Humanoid template rig produced a regressed GLB:\n  - " + detail
        )

    contract = validate_aigc_rigged_glb(output_glb_path)
    logger.info(format_contract_log(contract))
    if not contract.passed:
        raise RuntimeError(
            "Humanoid template rig failed API avatar contract: "
            + ", ".join(contract.codes)
            + f" metrics={contract.metrics}"
        )

    logger.info("Applied humanoid template rig: %s", output_glb_path)
    return output_glb_path, contract.to_dict()


def fbx_to_glb(fbx_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert FBX file to GLB format using Blender.

    Args:
        fbx_path: Path to the input FBX file
        output_path: Optional path for the output GLB file. If not provided,
                    will use the same directory and filename as input with .glb extension

    Returns:
        str: Path to the converted GLB file

    Raises:
        ImportError: If bpy (Blender Python module) is not available
        FileNotFoundError: If the input FBX file doesn't exist
        RuntimeError: If conversion fails
    """
    try:
        import bpy  # noqa: F401
    except ImportError:
        logger.info("bpy not in API venv; using Blender headless for FBX→GLB")
        return fbx_to_glb_headless(str(fbx_path), str(output_path) if output_path else None)

    # Validate input file
    fbx_path = Path(fbx_path)
    if not fbx_path.exists():
        raise FileNotFoundError(f"Input FBX file not found: {fbx_path}")

    # Determine output path
    if output_path is None:
        output_path = fbx_path.with_suffix(".glb")
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting FBX to GLB: {fbx_path} -> {output_path}")

    try:
        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import FBX file
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))

        # Export as GLB (minimal kwargs for Blender 4.0+ apt package compatibility)
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            export_apply=True,
            export_animations=True,
            export_skins=True,
        )

        # Verify the output file was created
        if not output_path.exists():
            raise RuntimeError(f"GLB file was not created at: {output_path}")

        logger.info(f"Successfully converted FBX to GLB: {output_path}")
        return str(output_path)

    except Exception as e:
        error_msg = f"Failed to convert FBX to GLB: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def fbx_to_glb_headless(fbx_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert FBX file to GLB format using Blender in headless mode.

    This function runs Blender as a subprocess in headless mode, which is useful
    when running in environments where GUI is not available or when you need
    better isolation.

    Args:
        fbx_path: Path to the input FBX file
        output_path: Optional path for the output GLB file. If not provided,
                    will use the same directory and filename as input with .glb extension

    Returns:
        str: Path to the converted GLB file

    Raises:
        FileNotFoundError: If the input FBX file doesn't exist or Blender is not found
        RuntimeError: If conversion fails
    """
    import subprocess

    from utils.blender_runtime import find_blender_binary

    blender_bin = find_blender_binary()
    if blender_bin is None:
        raise FileNotFoundError(
            "Blender executable not found for FBX→GLB. Install: sudo apt install -y blender"
        )

    # Validate input file
    fbx_path = Path(fbx_path).resolve()
    if not fbx_path.exists():
        raise FileNotFoundError(f"Input FBX file not found: {fbx_path}")

    # Determine output path
    if output_path is None:
        output_path = fbx_path.with_suffix(".glb")
    else:
        output_path = Path(output_path)
    output_path = output_path.resolve()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting FBX to GLB (headless): {fbx_path} -> {output_path}")

    fbx_str = str(fbx_path).replace("\\", "\\\\")
    out_str = str(output_path).replace("\\", "\\\\")

    # Minimal glTF export kwargs (Ubuntu Blender 4.0 rejects newer keyword args)
    script_content = f'''
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath="{fbx_str}")
bpy.ops.export_scene.gltf(
    filepath="{out_str}",
    export_format="GLB",
    export_apply=True,
    export_animations=True,
    export_skins=True,
)
print("Conversion completed successfully")
'''

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(script_content)
        script_path = script_file.name

    try:
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")

        cmd = [str(blender_bin), "--background", "--python", script_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
        )

        if result.returncode != 0:
            error_msg = f"Blender conversion failed: {result.stderr or result.stdout}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not output_path.exists():
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"GLB file was not created at: {output_path}"
                + (f" — Blender output: {detail[-500:]}" if detail else "")
            )

        logger.info(f"Successfully converted FBX to GLB (headless): {output_path}")
        return str(output_path)

    except subprocess.TimeoutExpired:
        error_msg = "Blender conversion timed out"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        error_msg = (
            "Blender executable not found. Make sure Blender is installed and in PATH."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    finally:
        # Clean up temporary script file
        try:
            os.unlink(script_path)
        except OSError:
            pass
