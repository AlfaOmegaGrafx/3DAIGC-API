"""
Triangle mesh decimation helpers (trimesh quadric error metrics).

Preserves connectivity style (collapse edges) rather than rebuilding quads.
No external binary — uses ``trimesh.Trimesh.simplify_quadric_decimation``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import trimesh

logger = logging.getLogger(__name__)

DEFAULT_TARGET_FACES = 210_000


def resolve_target_face_count(
    mesh: trimesh.Trimesh,
    *,
    target_face_count: Optional[int] = None,
    target_vertex_count: Optional[int] = None,
    ratio: Optional[float] = None,
    default_target_faces: int = DEFAULT_TARGET_FACES,
) -> int:
    """
    Pick a face budget for quadric decimation.

    Priority: ``target_face_count`` > ``ratio`` > ``target_vertex_count``
    (tri meshes ≈ 2 faces per vertex) > ``default_target_faces``.
    Never increases face count.
    """
    current = max(1, int(len(mesh.faces)))
    if target_face_count is not None:
        target = int(target_face_count)
    elif ratio is not None:
        r = float(ratio)
        if not 0.0 < r <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        target = max(1, int(round(current * r)))
    elif target_vertex_count is not None:
        # Closed triangle meshes: faces ≈ 2 * vertices
        target = max(1, int(target_vertex_count) * 2)
    else:
        target = int(default_target_faces)

    target = max(4, min(target, current))
    return target


def decimate_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_face_count: Optional[int] = None,
    target_vertex_count: Optional[int] = None,
    ratio: Optional[float] = None,
    default_target_faces: int = DEFAULT_TARGET_FACES,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Decimate ``mesh`` with quadric error metrics.

    Returns ``(simplified_mesh, info)``.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")

    face_target = resolve_target_face_count(
        mesh,
        target_face_count=target_face_count,
        target_vertex_count=target_vertex_count,
        ratio=ratio,
        default_target_faces=default_target_faces,
    )

    before_faces = len(mesh.faces)
    before_verts = len(mesh.vertices)

    if face_target >= before_faces:
        logger.info(
            "Decimate skipped (already ≤ target): %s faces, target %s",
            before_faces,
            face_target,
        )
        simplified = mesh.copy()
    else:
        logger.info(
            "Decimating mesh: %s → %s faces (quadric)",
            before_faces,
            face_target,
        )
        simplified = mesh.simplify_quadric_decimation(face_count=face_target)

    info = {
        "backend": "trimesh_quadric",
        "target_face_count": face_target,
        "before_faces": before_faces,
        "before_vertices": before_verts,
        "after_faces": len(simplified.faces),
        "after_vertices": len(simplified.vertices),
        "skipped": face_target >= before_faces,
    }
    return simplified, info


def get_trimesh_decimate_info() -> Dict[str, Any]:
    return {
        "name": "trimesh_decimate",
        "license": "MIT (trimesh)",
        "backend": "simplify_quadric_decimation",
        "available": True,
        "binary": None,
    }
