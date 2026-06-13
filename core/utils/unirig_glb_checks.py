"""
Regression checks for UniRig rig + textured GLB merge.

Reference model: TRELLIS textured **bird** (not eagle). Fixtures live under
``assets/example_autorig/regression/`` (``bird_trellis_textured.glb`` +
``bird_trellis_rig.fbx``).

Good merge keeps UniRig's remeshed proxy mesh (``character``) for skinning and
only projects source UVs/materials onto it.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class GlbAnalysis:
    path: Path
    vert_counts: tuple[int, ...]
    joint_counts: tuple[int, ...]
    scene_roots: tuple[str, ...]
    max_bone_translation: float
    has_images: bool
    has_base_color_texture: bool
    metallic_factor: Optional[float]
    roughness_factor: Optional[float]
    albedo_mean: Optional[float]
    albedo_std: Optional[float]

    @property
    def primary_vert_count(self) -> int:
        return max(self.vert_counts) if self.vert_counts else 0

    @property
    def has_skin(self) -> bool:
        return any(j > 0 for j in self.joint_counts)


@dataclass(frozen=True)
class UnirigMergeExpectations:
    """Expected metrics for bird_trellis regression fixture."""

    name: str = "bird_trellis"
    source_vert_count: int = 5829
    proxy_vert_count: int = 21188
    joint_count: int = 35
    max_bone_translation: float = 0.2703
    scene_roots: tuple[str, ...] = ("character", "Armature")
    forbidden_roots: tuple[str, ...] = ("geometry_0", "Node_0")
    albedo_mean: float = 77.0
    albedo_std: float = 58.2
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    vert_count_tolerance: float = 0.02
    bone_translation_tolerance: float = 0.02
    albedo_mean_tolerance: float = 8.0
    albedo_std_tolerance: float = 12.0


def _read_glb_chunks(path: Path) -> tuple[dict, bytes, int]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"Invalid GLB (too small): {path}")
    magic, _version, total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB file: {path}")
    offset = 12
    gltf: dict = {}
    bin_chunk = b""
    while offset < min(total_len, len(data)):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
    if not gltf:
        raise ValueError(f"GLB JSON chunk missing: {path}")
    return gltf, bin_chunk, total_len


def _decode_albedo_stats(gltf: dict, bin_chunk: bytes) -> tuple[Optional[float], Optional[float]]:
    images = gltf.get("images") or []
    if not images or not bin_chunk:
        return None, None
    img = images[0]
    if "bufferView" not in img:
        return None, None
    bv = gltf["bufferViews"][img["bufferView"]]
    start = bv.get("byteOffset", 0)
    png_bytes = bin_chunk[start : start + bv["byteLength"]]
    try:
        from PIL import Image
        import numpy as np

        rgb = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        return float(rgb.mean()), float(rgb.std())
    except Exception:
        return None, None


def analyze_glb(path: str | Path) -> GlbAnalysis:
    path = Path(path)
    gltf, bin_chunk, _total = _read_glb_chunks(path)
    nodes = gltf.get("nodes") or []
    meshes = gltf.get("meshes") or []
    skins = gltf.get("skins") or []

    vert_counts: list[int] = []
    for mesh in meshes:
        for prim in mesh.get("primitives") or []:
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None:
                vert_counts.append(gltf["accessors"][pos]["count"])

    scene_idx = gltf.get("scene", 0)
    root_indices = gltf["scenes"][scene_idx].get("nodes") or []
    scene_roots = tuple(nodes[i].get("name") or "" for i in root_indices)

    max_bone = 0.0
    joint_counts: list[int] = []
    for skin in skins:
        joints = skin.get("joints") or []
        joint_counts.append(len(joints))
        for ji in joints:
            t = nodes[ji].get("translation") or [0.0, 0.0, 0.0]
            dist = (t[0] ** 2 + t[1] ** 2 + t[2] ** 2) ** 0.5
            max_bone = max(max_bone, dist)

    mats = gltf.get("materials") or []
    metallic = roughness = None
    has_bct = False
    if mats:
        pbr = mats[0].get("pbrMetallicRoughness") or {}
        metallic = pbr.get("metallicFactor")
        roughness = pbr.get("roughnessFactor")
        has_bct = "baseColorTexture" in pbr

    albedo_mean, albedo_std = _decode_albedo_stats(gltf, bin_chunk)

    return GlbAnalysis(
        path=path,
        vert_counts=tuple(vert_counts),
        joint_counts=tuple(joint_counts),
        scene_roots=scene_roots,
        max_bone_translation=max_bone,
        has_images=bool(gltf.get("images")),
        has_base_color_texture=has_bct,
        metallic_factor=metallic,
        roughness_factor=roughness,
        albedo_mean=albedo_mean,
        albedo_std=albedo_std,
    )


def _approx(a: float, b: float, tol: float) -> bool:
    if b == 0:
        return abs(a - b) <= tol
    return abs(a - b) <= abs(b) * tol


def validate_unirig_merged_glb(
    source_mesh_path: str | Path,
    merged_glb_path: str | Path,
    *,
    expectations: Optional[UnirigMergeExpectations] = None,
    reference_fbx_glb_path: Optional[str | Path] = None,
) -> list[str]:
    """
    Return a list of validation error strings (empty == pass).

    Always checks generic anti-regression rules (no source-topology skinning).
    When ``expectations`` is set, also checks bird fixture metrics.
    When ``reference_fbx_glb_path`` is set, merged verts/joints must match it.
    """
    source_mesh_path = Path(source_mesh_path)
    merged_glb_path = Path(merged_glb_path)
    source = analyze_glb(source_mesh_path)
    merged = analyze_glb(merged_glb_path)
    errors: list[str] = []

    src_verts = source.primary_vert_count
    out_verts = merged.primary_vert_count

    if src_verts > 0 and out_verts <= int(src_verts * 1.2):
        errors.append(
            f"Merged mesh has {out_verts} verts (source {src_verts}) — "
            "likely skinning the upload mesh instead of UniRig proxy 'character'."
        )

    if "geometry_0" in merged.scene_roots:
        errors.append(
            "Scene root 'geometry_0' detected — source-mesh envelope export regression."
        )

    for bad in ("Node_0", "world", "Group"):
        if bad in merged.scene_roots:
            errors.append(f"Unexpected glTF import root '{bad}' in merged GLB.")

    if source.has_images:
        if not merged.has_images:
            errors.append("Source had embedded textures but merged GLB has no images.")
        if not merged.has_base_color_texture:
            errors.append("Merged GLB missing baseColorTexture.")
        if merged.albedo_mean is not None and merged.albedo_mean < 10.0:
            errors.append(
                f"Albedo texture appears black (mean={merged.albedo_mean:.2f})."
            )

    if not merged.has_skin:
        errors.append("Merged GLB has no skin — armature not exported.")

    if reference_fbx_glb_path is not None:
        ref = analyze_glb(reference_fbx_glb_path)
        if ref.primary_vert_count and not _approx(
            out_verts, ref.primary_vert_count, 0.02
        ):
            errors.append(
                f"Merged verts {out_verts} != fbx-only reference {ref.primary_vert_count}."
            )
        if ref.joint_counts and merged.joint_counts:
            if merged.joint_counts[0] != ref.joint_counts[0]:
                errors.append(
                    f"Merged joints {merged.joint_counts[0]} != "
                    f"fbx-only reference {ref.joint_counts[0]}."
                )
        if ref.max_bone_translation and not _approx(
            merged.max_bone_translation, ref.max_bone_translation, 0.05
        ):
            errors.append(
                f"Max bone translation {merged.max_bone_translation:.4f} != "
                f"reference {ref.max_bone_translation:.4f} (oversized/wrong rig)."
            )

    exp = expectations
    if exp is not None:
        if not _approx(out_verts, exp.proxy_vert_count, exp.vert_count_tolerance):
            errors.append(
                f"[{exp.name}] Expected ~{exp.proxy_vert_count} proxy verts, got {out_verts}."
            )
        if merged.joint_counts and merged.joint_counts[0] != exp.joint_count:
            errors.append(
                f"[{exp.name}] Expected {exp.joint_count} joints, got {merged.joint_counts[0]}."
            )
        if abs(merged.max_bone_translation - exp.max_bone_translation) > exp.bone_translation_tolerance:
            errors.append(
                f"[{exp.name}] Max bone translation "
                f"{merged.max_bone_translation:.4f} != ~{exp.max_bone_translation}."
            )
        if list(merged.scene_roots[: len(exp.scene_roots)]) != list(exp.scene_roots):
            errors.append(
                f"[{exp.name}] Scene roots {merged.scene_roots} != expected {exp.scene_roots}."
            )
        for forbidden in exp.forbidden_roots:
            if forbidden in merged.scene_roots:
                errors.append(f"[{exp.name}] Forbidden root '{forbidden}' present.")
        if merged.albedo_mean is not None:
            if abs(merged.albedo_mean - exp.albedo_mean) > exp.albedo_mean_tolerance:
                errors.append(
                    f"[{exp.name}] Albedo mean {merged.albedo_mean:.1f} != ~{exp.albedo_mean}."
                )
        if merged.albedo_std is not None:
            if abs(merged.albedo_std - exp.albedo_std) > exp.albedo_std_tolerance:
                errors.append(
                    f"[{exp.name}] Albedo std {merged.albedo_std:.1f} != ~{exp.albedo_std}."
                )
        if merged.metallic_factor != exp.metallic_factor:
            errors.append(
                f"[{exp.name}] metallicFactor {merged.metallic_factor} != {exp.metallic_factor}."
            )
        if merged.roughness_factor != exp.roughness_factor:
            errors.append(
                f"[{exp.name}] roughnessFactor {merged.roughness_factor} != {exp.roughness_factor}."
            )

    return errors


def assert_unirig_merged_glb(*args, **kwargs) -> None:
    errors = validate_unirig_merged_glb(*args, **kwargs)
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"UniRig merge regression failed:\n  - {joined}")


def bird_regression_fixture_paths(repo_root: Path | None = None) -> dict[str, Path]:
    root = repo_root or Path(__file__).resolve().parents[2]
    reg = root / "assets" / "example_autorig" / "regression"
    return {
        "source_glb": reg / "bird_trellis_textured.glb",
        "rig_fbx": reg / "bird_trellis_rig.fbx",
    }


def fixture_paths_available(repo_root: Path | None = None) -> bool:
    paths = bird_regression_fixture_paths(repo_root)
    return all(p.is_file() for p in paths.values())
