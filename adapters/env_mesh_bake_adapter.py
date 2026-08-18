"""
Bake env-scan / multi-view world Gaussians → OMB-ready environment_mesh.glb.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.lingbot_3dgs_train import gsplat_available
from core.utils.world_env_mesh_bake import (
    EnvMeshBakeConfig,
    WorldBakeError,
    bake_world_env_mesh,
    world_has_bake_cameras,
)

logger = logging.getLogger(__name__)


class EnvMeshBakeAdapter(ImageToMeshModel):
    """gsplat depth → TSDF → decimate → GLB for RP1/OMB environment mesh."""

    FEATURE_TYPE = "environment_scan"
    MODEL_ID = "env_mesh_bake"

    def __init__(self, **kwargs):
        super().__init__(
            model_id=self.MODEL_ID,
            model_path="outputs/worlds",
            vram_requirement=10240,
            supported_output_formats=["glb", "json"],
            feature_type=self.FEATURE_TYPE,
            max_images=1,
        )
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        return {"gsplat": gsplat_available()}

    def _unload_model(self):
        pass

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "world_id": {
                    "type": "string",
                    "description": "Existing world job id under outputs/worlds/",
                    "required": True,
                },
                "target_face_count": {
                    "type": "integer",
                    "default": 300000,
                    "minimum": 1000,
                    "maximum": 500000,
                    "required": False,
                },
                "voxel_resolution": {
                    "type": "integer",
                    "default": 320,
                    "minimum": 64,
                    "maximum": 384,
                    "required": False,
                },
                "max_views": {
                    "type": "integer",
                    "default": 96,
                    "minimum": 2,
                    "maximum": 200,
                    "required": False,
                },
                "data_factor": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 8,
                    "required": False,
                },
                "quality": {
                    "type": "string",
                    "default": "photo",
                    "enum": ["draft", "balanced", "photo"],
                    "required": False,
                },
                "color_export": {
                    "type": "string",
                    "default": "vertex",
                    "enum": ["vertex", "atlas"],
                    "required": False,
                },
            }
        }

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not gsplat_available():
            raise RuntimeError("gsplat + CUDA required for env mesh bake")

        world_id = str(inputs.get("world_id") or inputs.get("job_id") or "").strip()
        world_dir = inputs.get("world_directory") or inputs.get("world_dir")
        if world_dir:
            root = Path(world_dir)
        elif world_id:
            root = Path("outputs") / "worlds" / world_id
        else:
            raise ValueError("Provide world_id or world_directory for bake-env-mesh")

        if not root.is_dir():
            raise FileNotFoundError(f"World package not found: {root}")
        if not world_has_bake_cameras(root):
            raise WorldBakeError(
                "Missing gs_dataset/ cameras+images — TripoSplat image-to-world "
                "cannot bake; use props for RP1 or LingBot env-scan Phase A."
            )

        overrides = {
            "target_face_count": inputs.get("target_face_count"),
            "voxel_resolution": inputs.get("voxel_resolution"),
            "max_views": inputs.get("max_views"),
            "data_factor": inputs.get("data_factor"),
            "color_export": inputs.get("color_export"),
            "write_collider": inputs.get("write_collider", True),
        }
        cfg = EnvMeshBakeConfig.from_quality(
            str(inputs.get("quality") or "photo"),
            **{k: v for k, v in overrides.items() if v is not None},
        )
        info = bake_world_env_mesh(root, cfg=cfg)
        job_id = world_id or root.name or str(uuid.uuid4())
        mesh_path = root / "environment_mesh.glb"
        return {
            "success": True,
            "job_id": job_id,
            "world_directory": str(root),
            "world_manifest_url": f"/api/v1/system/jobs/{job_id}/download?asset=manifest",
            "world_base_url": f"/api/v1/system/jobs/{job_id}/world/",
            "output_mesh_path": str(mesh_path),
            "mesh_url": str(mesh_path),
            "generation_info": {
                "pipeline": "env_mesh_bake",
                "env_mesh_bake": info,
            },
        }
