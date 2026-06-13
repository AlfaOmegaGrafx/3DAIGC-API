"""
xatlas-based UV unwrapping (MIT license).

CPU-only atlas parametrization via the ``xatlas`` Python bindings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import trimesh
import xatlas

logger = logging.getLogger(__name__)


class XAtlasUnwrapper:
    """Parametrize a triangle mesh with xatlas and export OBJ/GLB with UVs."""

    def __init__(self, resolution: int = 1024):
        self.resolution = resolution

    def unwrap_mesh(
        self,
        mesh_path: Union[str, Path],
        output_path: Union[str, Path],
        *,
        output_format: str = "obj",
    ) -> Dict[str, Any]:
        mesh_path = Path(mesh_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load triangle mesh from {mesh_path}")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)

        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
        new_vertices = vertices[vmapping]
        new_faces = indices.astype(np.int64)

        out_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False,
        )
        out_mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uvs, dtype=np.float64))

        if output_format == "glb":
            out_mesh.export(output_path, file_type="glb")
        else:
            out_mesh.export(output_path, file_type="obj")

        num_charts = int(len(np.unique(vmapping)))
        logger.info(
            "xatlas UV: %d verts, %d faces, %d charts -> %s",
            len(new_vertices),
            len(new_faces),
            num_charts,
            output_path,
        )

        return {
            "output_mesh_path": str(output_path),
            "num_components": num_charts,
            "distortion": 0.0,
            "metadata": {
                "backend": "xatlas",
                "resolution": self.resolution,
                "input_vertices": int(len(vertices)),
                "input_faces": int(len(faces)),
                "output_vertices": int(len(new_vertices)),
                "output_faces": int(len(new_faces)),
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "xatlas",
            "license": "MIT",
            "backend": "xatlas",
            "device": "cpu",
        }
