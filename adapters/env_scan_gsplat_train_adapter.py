"""
Phase B only — gsplat train on an existing env-scan world package.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.lingbot_3dgs_train import (
    PhaseBTrainConfig,
    gsplat_available,
    train_and_apply_phase_b,
)

logger = logging.getLogger(__name__)


class EnvScanGsplatTrainAdapter(ImageToMeshModel):
    """Train gsplat on ``outputs/worlds/<id>/gs_dataset`` and swap environment.ply."""

    FEATURE_TYPE = "environment_scan"
    MODEL_ID = "env_scan_gsplat_train"

    def __init__(self, **kwargs):
        super().__init__(
            model_id=self.MODEL_ID,
            model_path="outputs/worlds",
            vram_requirement=12288,
            supported_output_formats=["ply", "json"],
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
                    "description": "Existing env-scan job / world id under outputs/worlds/",
                    "required": True,
                },
                "max_steps": {
                    "type": "integer",
                    "default": 7000,
                    "minimum": 100,
                    "maximum": 50000,
                    "required": False,
                },
                "data_factor": {
                    "type": "integer",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 8,
                    "required": False,
                },
            }
        }

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not gsplat_available():
            raise RuntimeError("gsplat + CUDA required for Phase B training")

        world_id = str(inputs.get("world_id") or inputs.get("job_id") or "").strip()
        world_dir = inputs.get("world_directory") or inputs.get("world_dir")
        if world_dir:
            root = Path(world_dir)
        elif world_id:
            root = Path("outputs") / "worlds" / world_id
        else:
            raise ValueError("Provide world_id or world_directory for Phase B train")

        if not root.is_dir():
            raise FileNotFoundError(f"World package not found: {root}")
        if not (root / "gs_dataset").is_dir():
            raise FileNotFoundError(
                f"Missing gs_dataset under {root} — run Phase A refine_to_3dgs first"
            )

        steps = int(inputs.get("max_steps") or inputs.get("train_3dgs_steps") or 7000)
        data_factor = int(inputs.get("data_factor") or 4)
        max_images = inputs.get("max_images")
        cfg = PhaseBTrainConfig(
            max_steps=max(100, steps),
            data_factor=max(1, data_factor),
            max_images=int(max_images) if max_images else None,
            enable_densify=bool(inputs.get("enable_densify") or False),
        )
        info = train_and_apply_phase_b(root, cfg=cfg)
        job_id = world_id or root.name or str(uuid.uuid4())
        return {
            "success": True,
            "job_id": job_id,
            "world_directory": str(root),
            "world_manifest_url": f"/api/v1/system/jobs/{job_id}/download?asset=manifest",
            "world_base_url": f"/api/v1/system/jobs/{job_id}/world/",
            "output_splat_path": str(root / "environment.ply"),
            "output_mesh_path": str(root / "environment.ply"),
            "mesh_url": str(root / "environment.ply"),
            "generation_info": {
                "pipeline": "env_scan_gsplat_train",
                "gaussian_train": info,
            },
        }
