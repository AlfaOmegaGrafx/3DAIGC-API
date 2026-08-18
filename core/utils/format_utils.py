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


def _vrm_addon_zip(repo_root: Path) -> Path:
    from utils.blender_runtime import resolve_vrm_addon_zip

    return resolve_vrm_addon_zip(repo_root)

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
    addon = _vrm_addon_zip(repo_root)
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
    *,
    output_vrm_path: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Rig target mesh using a humanoid VRM template armature (bones-only path).

    Preserves target materials; does not transfer template blend shapes.
    Returns ``(output_glb_path, rig_info.validation dict)``. Primary download
    should prefer the sibling ``.vrm`` when present.
    """
    template_vrm_path = str(Path(template_vrm_path).resolve())
    target_mesh_path = str(Path(target_mesh_path).resolve())
    output_glb_path = str(Path(output_glb_path).resolve())
    vrm_out = str(
        Path(output_vrm_path).resolve()
        if output_vrm_path
        else Path(output_glb_path).with_suffix(".vrm")
    )

    if not Path(template_vrm_path).is_file():
        raise FileNotFoundError(f"Template VRM not found: {template_vrm_path}")
    if not Path(target_mesh_path).is_file():
        raise FileNotFoundError(f"Target mesh not found: {target_mesh_path}")

    output_path = Path(output_glb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(vrm_out).parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "blender" / "apply_humanoid_template_rig.py"
    addon = _vrm_addon_zip(repo_root)
    from core.utils.template_rig_alignment import DEFAULT_ARMATURE_YAW_RAD

    _run_blender_script(
        script,
        {
            "template_vrm": template_vrm_path,
            "target_mesh": target_mesh_path,
            "output_glb": output_glb_path,
            "output_vrm": vrm_out,
            "vrm_addon_zip": str(addon),
            "armature_yaw_rad": DEFAULT_ARMATURE_YAW_RAD,
        },
        job_env_key="TEMPLATE_RIG_JOB_JSON",
        ok_token="APPLY_HUMANOID_TEMPLATE_RIG_OK",
        error_label="Humanoid template rig",
    )
    if not output_path.exists():
        raise RuntimeError(f"Template-rigged GLB was not created: {output_glb_path}")
    if not Path(vrm_out).is_file():
        raise RuntimeError(f"Template-rigged VRM was not created: {vrm_out}")

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

    validation = contract.to_dict()
    validation.update(
        {
            "output_glb": output_glb_path,
            "output_vrm": vrm_out,
            "has_vrm": True,
        }
    )
    logger.info("Applied humanoid template rig: glb=%s vrm=%s", output_glb_path, vrm_out)
    return output_glb_path, validation


def count_glb_morph_targets(glb_path: str) -> int:
    """
    Count morph target (blend shape) entries across all meshes in a GLB.

    Used to validate Phase 5 head-stitch exports keep template.vrm morphs.
    """
    import json
    import struct
    from pathlib import Path as _Path

    data = _Path(glb_path).read_bytes()
    if len(data) < 20:
        return 0
    magic, _version, total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:  # glTF
        return 0
    offset = 12
    gltf = None
    while offset < min(total_len, len(data)):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk)
            break
    if not gltf:
        return 0
    total = 0
    for mesh in gltf.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            targets = prim.get("targets") or []
            total += len(targets)
        weights = mesh.get("weights")
        if isinstance(weights, list) and weights and not any(
            (p.get("targets") for p in mesh.get("primitives") or [])
        ):
            total += len(weights)
    return total


def apply_humanoid_template_wrap(
    template_vrm_path: str,
    target_mesh_path: str,
    output_glb_path: str,
    *,
    output_vrm_path: Optional[str] = None,
    expect_headless_body: Optional[bool] = None,
    gnm_identity: bool = False,
    character_gender: Optional[str] = None,
    character_ethnicity: Optional[str] = None,
    gnm_seed: Optional[int] = None,
    gnm_bake_expressions: bool = False,
    gnm_replace_morphs: bool = False,
    face_likeness: bool = False,
    likeness_alpha: float = 0.65,
    likeness_image_path: Optional[str] = None,
    likeness_source: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Phase 5 head stitch: keep template.vrm head morphs + AIGC body below neck.

    Optional Phase A/B/4: GNM ethnicity identity, additive expression morphs,
    and face likeness via MeshMonk (RBF fallback).

    Likeness source (``likeness_source``):
      - ``auto`` (default): selfie MediaPipe mesh if ``likeness_image_path``, else body ROI
      - ``selfie``: require selfie → MediaPipe face mesh
      - ``body_roi``: crop top of AIGC body mesh (legacy)

    Exports GLB sibling for contract checks plus primary ``.vrm`` for Studio/Appearance.
    """
    import tempfile

    import numpy as np

    from core.utils.face_correspondence import (
        crop_head_mesh,
        deform_template_neutral,
        export_delta_npz,
        load_mesh_with_faces_from_glb,
        load_template_head_neutral,
        selfie_image_to_face_mesh,
        transfer_gnm_deltas_to_template,
    )
    from core.utils.gnm_head import (
        bake_expression_deltas,
        generate_identity_mesh,
        normalize_ethnicity,
        normalize_gender,
    )

    template_vrm_path = str(Path(template_vrm_path).resolve())
    target_mesh_path = str(Path(target_mesh_path).resolve())
    output_glb_path = str(Path(output_glb_path).resolve())
    vrm_out = str(
        Path(output_vrm_path).resolve()
        if output_vrm_path
        else Path(output_glb_path).with_suffix(".vrm")
    )

    if not Path(template_vrm_path).is_file():
        raise FileNotFoundError(f"Template VRM not found: {template_vrm_path}")
    if not Path(target_mesh_path).is_file():
        raise FileNotFoundError(f"Target mesh not found: {target_mesh_path}")

    output_path = Path(output_glb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(vrm_out).parent.mkdir(parents=True, exist_ok=True)

    gender = normalize_gender(character_gender)
    ethnicity = normalize_ethnicity(character_ethnicity)
    # GNM identity when flagged or when ethnicity is set (Studio ethnicity chip).
    use_gnm = bool(gnm_identity) or bool(ethnicity)

    wrap_meta: dict = {
        "gnm_identity": False,
        "gnm_bake_expressions": False,
        "face_likeness": False,
    }
    head_delta_npz = None
    expr_delta_npz = None

    if use_gnm or face_likeness or gnm_bake_expressions:
        head_info = load_template_head_neutral(template_vrm_path)
        template_neutral = head_info["vertices"]
        delta = None
        identity_coeffs = None
        engines: list[str] = []

        template_faces = head_info.get("faces")

        if use_gnm:
            gnm_mesh = generate_identity_mesh(
                gender=gender or "female",
                ethnicity=ethnicity or "white",
                seed=gnm_seed,
            )
            identity_coeffs = gnm_mesh["identity"]
            # Cross-topology (GNM ≠ VRM): dense MeshMonk shreds eyes/teeth/hair.
            # Mild RBF + reject-if-unsafe keeps morph accessories intact.
            gnm_alpha = 0.28
            delta = deform_template_neutral(
                gnm_mesh["vertices"],
                template_neutral,
                method="rbf",
                alpha=gnm_alpha,
                template_faces=template_faces,
                source_faces=gnm_mesh.get("faces"),
                engine_out=engines,
                reject_if_clamped=True,
                max_frac=0.12,
                rms_frac=0.04,
            )
            wrap_meta.update(
                {
                    "gnm_identity": True,
                    "character_gender": gender or "female",
                    "character_ethnicity": ethnicity or "white",
                    "gnm_seed": gnm_seed,
                    "gnm_alpha": gnm_alpha,
                    "gnm_warp_method": "rbf",
                }
            )
            logger.info(
                "GNM identity warp gender=%s ethnicity=%s delta_rms=%.5f",
                wrap_meta["character_gender"],
                wrap_meta["character_ethnicity"],
                float(np.sqrt((delta**2).mean())),
            )

        if face_likeness:
            try:
                src_mode = (likeness_source or "auto").strip().lower()
                if src_mode not in ("auto", "selfie", "body_roi"):
                    src_mode = "auto"
                head_verts = None
                head_faces = None
                resolved_source = "body_roi"
                selfie_path = (likeness_image_path or "").strip()
                want_selfie = src_mode == "selfie" or (
                    src_mode == "auto" and bool(selfie_path)
                )
                selfie_error = None

                if want_selfie:
                    if not selfie_path:
                        selfie_error = (
                            "likeness_source=selfie requires likeness_image_path"
                        )
                    else:
                        try:
                            selfie_mesh = selfie_image_to_face_mesh(selfie_path)
                            head_verts = selfie_mesh["vertices"]
                            head_faces = selfie_mesh.get("faces")
                            resolved_source = "selfie"
                        except Exception as selfie_exc:
                            selfie_error = str(selfie_exc)
                            logger.warning(
                                "Selfie likeness mesh failed (%s); falling back to body_roi",
                                selfie_exc,
                            )

                if head_verts is None:
                    # Always prefer a usable likeness mesh over a hard skip.
                    aigc = load_mesh_with_faces_from_glb(target_mesh_path)
                    head_verts, head_faces = crop_head_mesh(
                        aigc["vertices"],
                        aigc.get("faces"),
                        height_frac=0.38,
                    )
                    if selfie_error or (want_selfie and not selfie_path):
                        resolved_source = "body_roi_fallback"
                        wrap_meta["face_likeness_selfie_error"] = selfie_error or (
                            "likeness_source=selfie but no likeness_image_path"
                        )
                    else:
                        resolved_source = "body_roi"

                like_alpha = min(float(likeness_alpha), 0.45)
                like_delta = deform_template_neutral(
                    head_verts,
                    template_neutral if delta is None else (template_neutral + delta),
                    method="rbf",
                    alpha=like_alpha,
                    template_faces=template_faces,
                    source_faces=head_faces,
                    engine_out=engines,
                    reject_if_clamped=True,
                    max_frac=0.12,
                    rms_frac=0.04,
                )
                delta = like_delta if delta is None else (delta + like_delta)
                wrap_meta["face_likeness"] = True
                wrap_meta["likeness_alpha"] = like_alpha
                wrap_meta["likeness_source"] = resolved_source
                wrap_meta["likeness_warp_method"] = "rbf"
                logger.info(
                    "Face likeness warp source=%s alpha=%.2f delta_rms=%.5f",
                    resolved_source,
                    like_alpha,
                    float(np.sqrt((like_delta**2).mean())),
                )
            except Exception as exc:
                logger.warning("face_likeness skipped: %s", exc)
                wrap_meta["face_likeness"] = False
                wrap_meta["face_likeness_skip_reason"] = str(exc)

        if engines:
            wrap_meta["correspondence_engines"] = engines

        tmp_dir = tempfile.mkdtemp(prefix="gnm_wrap_")
        if delta is not None:
            delta_rms = float(np.sqrt((delta.astype(np.float64) ** 2).mean()))
            if delta_rms < 1e-8:
                wrap_meta["head_delta_skipped"] = "zero_after_sanitize"
                logger.info("Head delta skipped (zero after sanitize)")
            else:
                head_delta_npz = str(
                    export_delta_npz(
                        Path(tmp_dir) / "head_delta.npz",
                        delta,
                        mesh_name=head_info["mesh_name"] or "AvatarHead",
                        mesh_index=int(head_info["mesh_index"]),
                        prim_index=int(head_info["prim_index"]),
                        meta=wrap_meta,
                    )
                )

        if gnm_bake_expressions and identity_coeffs is not None:
            from core.utils.gnm_head import evaluate_mesh

            gnm_deltas = bake_expression_deltas(
                identity_coeffs,
                seed=gnm_seed,
            )
            gnm_neutral, _ = evaluate_mesh(identity=identity_coeffs, expression=None)
            transferred = transfer_gnm_deltas_to_template(
                gnm_deltas,
                gnm_neutral,
                template_neutral if delta is None else (template_neutral + delta),
            )
            # Prefix collisions with gnm_ unless replace is requested
            if not gnm_replace_morphs:
                protected = {
                    "blink",
                    "blinkleft",
                    "blinkright",
                    "aa",
                    "ih",
                    "ou",
                    "ee",
                    "oh",
                    "jawopen",
                    "jaw_drop",
                }
                remapped = {}
                for name, arr in transferred.items():
                    key = name.lower()
                    remapped[name if key not in protected else f"gnm_{name}"] = arr
                transferred = remapped
            expr_delta_npz = str(
                export_delta_npz(
                    Path(tmp_dir) / "expr_deltas.npz",
                    np.zeros_like(template_neutral),
                    mesh_name=head_info["mesh_name"] or "AvatarHead",
                    expression_deltas=transferred,
                    meta={"gnm_bake_expressions": True},
                )
            )
            wrap_meta["gnm_bake_expressions"] = True
            wrap_meta["gnm_expression_names"] = list(transferred.keys())

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "blender" / "apply_humanoid_template_wrap.py"
    addon = _vrm_addon_zip(repo_root)
    from core.utils.template_rig_alignment import DEFAULT_ARMATURE_YAW_RAD

    job = {
        "template_vrm": template_vrm_path,
        "target_mesh": target_mesh_path,
        "output_glb": output_glb_path,
        "output_vrm": vrm_out,
        "vrm_addon_zip": str(addon),
        "armature_yaw_rad": DEFAULT_ARMATURE_YAW_RAD,
    }
    # None → Blender auto-detect (full-body mannequin vs neck-open Body+Cloth).
    # True forces feet→neck scale (giraffe neck if the mesh still has a head).
    if expect_headless_body is not None:
        job["expect_headless_body"] = bool(expect_headless_body)
        if expect_headless_body:
            job["head_align_sink"] = 0.018
    if head_delta_npz:
        job["head_delta_npz"] = head_delta_npz
    if expr_delta_npz:
        job["expression_delta_npz"] = expr_delta_npz

    _run_blender_script(
        script,
        job,
        job_env_key="TEMPLATE_WRAP_JOB_JSON",
        ok_token="APPLY_HUMANOID_TEMPLATE_WRAP_OK",
        error_label="Humanoid template wrap (head stitch)",
    )
    if not output_path.exists():
        raise RuntimeError(f"Template-wrap GLB was not created: {output_glb_path}")
    if not Path(vrm_out).is_file():
        raise RuntimeError(f"Template-wrap VRM was not created: {vrm_out}")

    morph_count = count_glb_morph_targets(output_glb_path)
    if morph_count < 1:
        raise RuntimeError(
            "Phase 5 head stitch produced a GLB with no morph targets — "
            "template head shape keys were lost (see MESH_WRAP_ROADMAP.md)"
        )

    from core.utils.aigc_rig_contract import (
        format_contract_log,
        validate_aigc_rigged_glb,
    )

    contract = validate_aigc_rigged_glb(output_glb_path)
    logger.info(format_contract_log(contract))
    critical = {
        c
        for c in contract.codes
        if c
        in {
            "character_upside_down",
            "missing_skinned_mesh",
            "insufficient_joints",
        }
    }
    if critical:
        raise RuntimeError(
            "Humanoid template wrap failed API avatar contract: "
            + ", ".join(sorted(critical))
            + f" metrics={contract.metrics}"
        )
    if "character_facing_backwards" in contract.codes:
        logger.warning(
            "template_wrap head stitch: character_facing_backwards advisory "
            "(morphs kept; refine yaw in a follow-up if needed)"
        )

    wrap_status = "head_stitch"
    if wrap_meta.get("gnm_identity"):
        wrap_status = "head_stitch_gnm_identity"
    if wrap_meta.get("face_likeness"):
        wrap_status = f"{wrap_status}_likeness"

    validation = contract.to_dict()
    validation.update(
        {
            "wrap_status": wrap_status,
            "wrap_humanoid_only": True,
            "blend_shapes_on_generated_mesh": True,
            "morph_target_count": morph_count,
            "phase": 5,
            "output_glb": output_glb_path,
            "output_vrm": vrm_out,
            "has_vrm": True,
            **wrap_meta,
        }
    )
    logger.info(
        "Applied humanoid template wrap (head stitch): %s morphs=%s vrm=%s meta=%s",
        output_glb_path,
        morph_count,
        vrm_out,
        wrap_meta,
    )
    return output_glb_path, validation


def apply_appearance_component_rig(
    template_vrm_path: str,
    target_mesh_path: str,
    output_glb_path: str,
    *,
    appearance_slot: str = "Legs",
    output_vrm_path: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Fit an AIGC clothing mesh into an Appearance Editor VRM bone region.

    Runs ``scripts/blender/apply_appearance_component_rig.py``. Returns
    ``(output_glb_path, validation dict)`` with ``appearance_slot``.
    """
    template_vrm_path = str(Path(template_vrm_path).resolve())
    target_mesh_path = str(Path(target_mesh_path).resolve())
    output_glb_path = str(Path(output_glb_path).resolve())
    vrm_out = str(
        Path(output_vrm_path).resolve()
        if output_vrm_path
        else Path(output_glb_path).with_suffix(".vrm")
    )

    if not Path(template_vrm_path).is_file():
        raise FileNotFoundError(f"Appearance base VRM not found: {template_vrm_path}")
    if not Path(target_mesh_path).is_file():
        raise FileNotFoundError(f"Target clothing mesh not found: {target_mesh_path}")

    output_path = Path(output_glb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(vrm_out).parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "blender" / "apply_appearance_component_rig.py"
    addon = _vrm_addon_zip(repo_root)

    slot = str(appearance_slot or "Legs").strip() or "Legs"
    _run_blender_script(
        script,
        {
            "template_vrm": template_vrm_path,
            "target_mesh": target_mesh_path,
            "output_glb": output_glb_path,
            "output_vrm": vrm_out,
            "appearance_slot": slot,
            "vrm_addon_zip": str(addon) if addon.is_file() else "",
        },
        job_env_key="APPEARANCE_COMPONENT_JOB_JSON",
        ok_token="APPLY_APPEARANCE_COMPONENT_RIG_OK",
        error_label="Appearance component clothing fit",
    )
    if not output_path.exists():
        raise RuntimeError(
            f"Appearance-component GLB was not created: {output_glb_path}"
        )

    validation = {
        "appearance_slot": slot,
        "rig_mode": "appearance_component",
        "generation_method": "appearance_component_vrm_fit",
        "output_glb": output_glb_path,
        "output_vrm": vrm_out if Path(vrm_out).is_file() else None,
        "has_vrm": Path(vrm_out).is_file(),
    }
    logger.info(
        "Applied appearance component rig: slot=%s glb=%s vrm=%s",
        slot,
        output_glb_path,
        validation["output_vrm"],
    )
    return output_glb_path, validation


def _quat_mul_xyzw(a: list, b: list) -> list:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


_GLTF_YAW_PI_QUAT = [0.0, 1.0, 0.0, 0.0]


def _gltf_creature_faces_minus_z(gltf: dict) -> bool:
    """Return True when hips→head (or front vs back feet) points along glTF -Z."""
    import math

    nodes = gltf.get("nodes") or []
    if not nodes:
        return True

    def mat4_from_trs(t, r, s):
        x, y, z, w = r
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return [
            (1 - 2 * (yy + zz)) * s[0],
            (2 * (xy + wz)) * s[0],
            (2 * (xz - wy)) * s[0],
            0.0,
            (2 * (xy - wz)) * s[1],
            (1 - 2 * (xx + zz)) * s[1],
            (2 * (yz + wx)) * s[1],
            0.0,
            (2 * (xz + wy)) * s[2],
            (2 * (yz - wx)) * s[2],
            (1 - 2 * (xx + yy)) * s[2],
            0.0,
            t[0],
            t[1],
            t[2],
            1.0,
        ]

    def mat_mul(a, b):
        return [
            sum(a[k * 4 + r0] * b[c * 4 + k] for k in range(4))
            for c in range(4)
            for r0 in range(4)
        ]

    ident = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    world = [ident[:] for _ in nodes]

    def walk(idx, parent):
        node = nodes[idx]
        local = mat4_from_trs(
            node.get("translation", [0.0, 0.0, 0.0]),
            node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
            node.get("scale", [1.0, 1.0, 1.0]),
        )
        world[idx] = mat_mul(parent, local)
        for child in node.get("children") or []:
            walk(child, world[idx])

    for scene in gltf.get("scenes") or []:
        for root_idx in scene.get("nodes") or []:
            walk(root_idx, ident)

    by_name = {node.get("name"): i for i, node in enumerate(nodes) if node.get("name")}

    def world_pos(name):
        idx = by_name.get(name)
        if idx is None:
            return None
        m = world[idx]
        return (m[12], m[13], m[14])

    def forward_from(a, b):
        if a is None or b is None:
            return None
        fx, fy, fz = a[0] - b[0], 0.0, a[2] - b[2]
        mag = math.hypot(fx, fz)
        if mag < 1e-9:
            return None
        return (fx / mag, 0.0, fz / mag)

    def avg_pos(names):
        pts = [world_pos(n) for n in names]
        pts = [p for p in pts if p is not None]
        if not pts:
            return None
        return (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )

    front = avg_pos(["Front_Leg_Foot_L", "Front_Leg_Foot_R"])
    back = avg_pos(["Back_Leg_Foot_L", "Back_Leg_Foot_R"])
    fwd = forward_from(front, back) or forward_from(world_pos("Head"), world_pos("Hips"))
    if fwd is None:
        return True
    return fwd[2] <= -0.25


def _yaw_gltf_scene_roots_pi(glb_path: Path) -> None:
    """Rotate top-level scene nodes 180° on Y (glTF XZ). Fixes Mesh2Motion fox +Z rest pose."""
    gltf = _read_gltf_json(glb_path)
    for scene in gltf.get("scenes") or []:
        for idx in scene.get("nodes") or []:
            node = gltf["nodes"][idx]
            rot = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
            node["rotation"] = _quat_mul_xyzw(_GLTF_YAW_PI_QUAT, rot)
    _write_gltf_json(glb_path, gltf)


def _ensure_creature_template_glb_faces_minus_z(glb_path: Path) -> bool:
    """
    Verify creature GLB faces glTF -Z.

    Facing must be baked in Blender (post-parent yaw + transform_apply). A
    JSON-only root quaternion patch leaves inverse-bind matrices stale so the
    skeleton overlay and skinned mesh disagree (bones look \"backwards\").
    """
    gltf = _read_gltf_json(glb_path)
    if _gltf_creature_faces_minus_z(gltf):
        return False
    logger.warning(
        "Creature template GLB does not face glTF -Z after Blender bake: %s "
        "(refusing JSON root-yaw patch — re-check apply_creature_template_rig.py)",
        glb_path,
    )
    return False


def apply_creature_template_rig(
    skeleton_glb_path: str,
    target_mesh_path: str,
    output_glb_path: str,
    *,
    creature_template_id: str = "fox",
    armature_yaw_rad: float = 0.0,
) -> Tuple[str, dict]:
    """
    Rig target mesh using a Mesh2Motion creature skeleton template.

    Preserves target materials; uses Blender envelope weights.
    Returns ``(output_glb_path, rig_info dict)``.
    """
    skeleton_glb_path = str(Path(skeleton_glb_path).resolve())
    target_mesh_path = str(Path(target_mesh_path).resolve())
    output_glb_path = str(Path(output_glb_path).resolve())

    if not Path(skeleton_glb_path).is_file():
        raise FileNotFoundError(f"Creature skeleton GLB not found: {skeleton_glb_path}")
    if not Path(target_mesh_path).is_file():
        raise FileNotFoundError(f"Target mesh not found: {target_mesh_path}")

    output_path = Path(output_glb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "blender" / "apply_creature_template_rig.py"

    _run_blender_script(
        script,
        {
            "skeleton_glb": skeleton_glb_path,
            "target_mesh": target_mesh_path,
            "output_glb": output_glb_path,
            "creature_template_id": creature_template_id,
            "armature_yaw_rad": armature_yaw_rad,
        },
        job_env_key="CREATURE_RIG_JOB_JSON",
        ok_token="APPLY_CREATURE_TEMPLATE_RIG_OK",
        error_label="Creature template rig",
    )
    if not output_path.exists():
        raise RuntimeError(f"Creature template rig GLB was not created: {output_glb_path}")

    if _ensure_creature_template_glb_faces_minus_z(output_path):
        logger.info("Creature template rig: applied 180° Y root correction for glTF -Z forward")

    if target_mesh_path.lower().endswith((".glb", ".gltf")):
        _preserve_pbr_materials_from_source(Path(target_mesh_path), output_path)

    from core.utils.creature_template import (
        expected_bone_names,
        load_bone_profile,
        validate_creature_template,
    )
    from core.utils.unirig_glb_checks import analyze_glb

    errors = validate_creature_template(creature_template_id)
    if errors:
        raise RuntimeError(
            "Creature template assets invalid:\n  - " + "\n  - ".join(errors)
        )

    analysis = analyze_glb(output_glb_path)
    if not analysis.has_skin:
        raise RuntimeError("Creature template rig produced GLB without skin")

    profile = load_bone_profile(creature_template_id)
    expected = set(expected_bone_names(creature_template_id))
    rig_info = {
        "rig_mode": "creature_template",
        "creature_template_id": creature_template_id,
        "rig_type": "creature_template",
        "generation_method": "mesh2motion_creature_template",
        "placement_source": "auto",
        "bone_profile_version": profile.get("version", 1),
        "mesh2motion_skeleton_type": profile.get("mesh2motion_skeleton_type", creature_template_id),
        "expected_bone_count": len(expected),
        "has_skinning": True,
        "rig_quality": "ok",
        "quality_hints": [],
    }
    logger.info("Applied creature template rig (%s): %s", creature_template_id, output_glb_path)
    return output_glb_path, rig_info


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
