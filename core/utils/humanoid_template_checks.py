"""Regression checks for humanoid template rig outputs."""
from __future__ import annotations

from pathlib import Path

from core.utils.unirig_glb_checks import analyze_glb


def validate_template_rigged_glb(
    target_mesh_path: str | Path,
    rigged_glb_path: str | Path,
    *,
    min_joints: int = 40,
) -> list[str]:
    source = analyze_glb(target_mesh_path)
    rigged = analyze_glb(rigged_glb_path)
    errors: list[str] = []

    if not rigged.has_skin:
        errors.append("Rigged GLB has no skin — armature not exported.")
    if not rigged.joint_counts or rigged.joint_counts[0] < min_joints:
        joints = rigged.joint_counts[0] if rigged.joint_counts else 0
        errors.append(
            f"Expected >= {min_joints} joints for humanoid template rig, got {joints}."
        )
    if rigged.primary_vert_count < source.primary_vert_count * 0.8:
        errors.append(
            f"Rigged mesh lost vertices ({rigged.primary_vert_count} vs "
            f"source {source.primary_vert_count})."
        )
    if source.has_images and not rigged.has_base_color_texture:
        errors.append("Target had textures but rigged GLB missing baseColorTexture.")

    return errors
