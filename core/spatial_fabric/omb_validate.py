"""OMB spatial-fabric GLB validation helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import trimesh

OMB_TIER_LIMITS = {
    1: {"triangles": 500, "texture_px": 64},
    2: {"triangles": 2000, "texture_px": 128},
    3: {"triangles": 10000, "texture_px": 256},
    4: {"triangles": 150000, "texture_px": 1024},
    5: {"triangles": 150000, "texture_px": 2048},
}


@dataclass
class GlbStats:
    triangles: int
    file_size_bytes: int
    texture_max_dimension: int
    texture_total_pixels: int
    has_draco: bool
    format: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triangles": self.triangles,
            "file_size_bytes": self.file_size_bytes,
            "texture_max_dimension": self.texture_max_dimension,
            "texture_total_pixels": self.texture_total_pixels,
            "has_draco": self.has_draco,
            "format": self.format,
        }


def _estimate_texture_stats(scene: trimesh.Scene) -> Tuple[int, int]:
    max_dim = 0
    total_pixels = 0
    for geom in scene.geometry.values():
        visual = getattr(geom, "visual", None)
        material = getattr(visual, "material", None) if visual else None
        image = getattr(material, "image", None) if material else None
        if image is None:
            continue
        w, h = getattr(image, "size", (0, 0))
        max_dim = max(max_dim, w, h)
        total_pixels += w * h
    return max_dim, total_pixels


def analyze_glb_path(path: str) -> GlbStats:
    file_size = os.path.getsize(path)
    with open(path, "rb") as fh:
        header = fh.read(4096)
    has_draco = b"KHR_draco_mesh_compression" in header

    scene = trimesh.load(path, force="scene")
    triangles = 0
    tex_scene = trimesh.Scene()
    if isinstance(scene, trimesh.Scene):
        tex_scene = scene
        for geom in scene.geometry.values():
            if hasattr(geom, "faces"):
                triangles += int(len(geom.faces))
    elif hasattr(scene, "faces"):
        triangles = int(len(scene.faces))
        tex_scene.add_geometry(scene)

    tex_max, tex_total = _estimate_texture_stats(tex_scene)

    return GlbStats(
        triangles=triangles,
        file_size_bytes=file_size,
        texture_max_dimension=tex_max,
        texture_total_pixels=tex_total,
        has_draco=has_draco,
        format=os.path.splitext(path)[1].lower().lstrip(".") or "glb",
    )


def recommend_omb_tier(stats: GlbStats, use_pbr: bool = True) -> Dict[str, Any]:
    tier = 1
    reasons: List[str] = []

    for candidate in (1, 2, 3, 4, 5):
        limits = OMB_TIER_LIMITS[candidate]
        if stats.triangles <= limits["triangles"] and stats.texture_max_dimension <= limits[
            "texture_px"
        ]:
            tier = candidate
            break
        tier = min(candidate + 1, 5)

    if use_pbr and tier < 5:
        tier = min(tier + 1, 5)
        reasons.append("PBR materials bump tier by +1 per OMB modifiers")

    if stats.has_draco:
        reasons.append("Draco compression is not allowed for RP1 export")

    if stats.file_size_bytes > 64 * 1024 * 1024:
        reasons.append("File exceeds 64 MB RP1 absolute cap")

    limits = OMB_TIER_LIMITS[tier]
    within_tier = (
        stats.triangles <= limits["triangles"]
        and stats.texture_max_dimension <= limits["texture_px"]
        and not stats.has_draco
        and stats.file_size_bytes <= 64 * 1024 * 1024
    )

    return {
        "recommended_tier": tier,
        "within_recommended_tier": within_tier,
        "tier_limits": limits,
        "reasons": reasons,
    }
