"""
Arc2Avatar adapter — single image → FLAME-linked head 3D Gaussian splat.

Upstream: https://github.com/dimgerogiannis/Arc2Avatar
Paper: https://arc2avatar.github.io/

Does NOT export VRM/mesh body. Output is point_cloud.ply for Spark.js.
Enable in models.yaml after thirdparty/Arc2Avatar + ARC2AVATAR_PYTHON + weights.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from utils.arc2avatar_pipeline_helper import (
    DEFAULT_ARC2AVATAR_ROOT,
    probe_arc2avatar_install,
    run_arc2avatar_train,
)

logger = logging.getLogger(__name__)


class Arc2AvatarNotReadyError(RuntimeError):
    """Raised when thirdparty install / weights / python env is incomplete."""


class Arc2AvatarHeadAdapter(ImageToMeshModel):
    """SDS train Arc2Avatar head splat from one face photo."""

    FEATURE_TYPE = "arc2avatar_head"
    MODEL_ID = "arc2avatar_head"

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 24576,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
        arc2avatar_root: Optional[str] = None,
        **kwargs,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = str(DEFAULT_ARC2AVATAR_ROOT)
        if feature_type is None:
            feature_type = self.FEATURE_TYPE
        if supported_output_formats is None:
            supported_output_formats = ["ply"]

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=supported_output_formats,
            feature_type=feature_type,
            max_images=1,
        )
        self.arc2avatar_root = Path(arc2avatar_root or DEFAULT_ARC2AVATAR_ROOT)
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        self._status_cache: Optional[Dict[str, Any]] = None

    def _load_model(self):
        status = probe_arc2avatar_install(self.arc2avatar_root)
        self._status_cache = status
        if not status.get("integrated"):
            raise Arc2AvatarNotReadyError(
                "Arc2Avatar not ready: " + "; ".join(status.get("blocking_reasons") or [])
            )
        return {"status": status}

    def _unload_model(self):
        self._status_cache = None

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "iterations": {
                    "type": "integer",
                    "description": "SDS iterations (upstream default 7000; lower for smoke)",
                    "default": 7000,
                    "minimum": 100,
                    "maximum": 20000,
                    "required": False,
                },
                "batch_size": {
                    "type": "integer",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 8,
                    "required": False,
                },
            }
        }

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" not in inputs:
            raise ValueError("image_path is required for Arc2Avatar head generation")

        image_path = Path(inputs["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        status = probe_arc2avatar_install(self.arc2avatar_root)
        if not status.get("integrated"):
            raise Arc2AvatarNotReadyError(
                "Arc2Avatar not ready: " + "; ".join(status.get("blocking_reasons") or [])
            )

        job_token = str(uuid.uuid4())[:8]
        out_dir = Path("outputs") / "arc2avatar" / f"job_{job_token}"
        iterations = inputs.get("iterations")
        if iterations is None and isinstance(inputs.get("model_parameters"), dict):
            iterations = inputs["model_parameters"].get("iterations")
        batch_size = int(
            inputs.get("batch_size")
            or (inputs.get("model_parameters") or {}).get("batch_size")
            or 4
        )

        logger.info(
            "Arc2Avatar train start image=%s out=%s iterations=%s",
            image_path,
            out_dir,
            iterations,
        )
        result = run_arc2avatar_train(
            image_path,
            out_dir,
            root=self.arc2avatar_root,
            iterations=int(iterations) if iterations is not None else None,
            batch_size=batch_size,
        )
        ply = result["output_splat_path"]
        return {
            "success": True,
            "output_splat_path": ply,
            "output_mesh_path": ply,
            "mesh_url": ply,
            "splat_dir": result.get("splat_dir"),
            "subject_dir": result.get("subject_dir"),
            "format": "ply",
            "generation_info": {
                "model": self.MODEL_ID,
                "pipeline": "arc2avatar_sds",
                "flame_based": True,
                "vrm_export": False,
                "elapsed_s": result.get("elapsed_s"),
                "train_log": result.get("train_log"),
                "source_ply": result.get("source_ply"),
            },
        }


# Backward-compatible names for the previous stub
def get_arc2avatar_status() -> Dict[str, Any]:
    status = probe_arc2avatar_install()
    return {
        **status,
        "model_id": Arc2AvatarHeadAdapter.MODEL_ID,
        "feature_type": Arc2AvatarHeadAdapter.FEATURE_TYPE,
        "output_formats": ["ply"],
        "blendshapes": True,
        "flame_based": True,
        "vrm_export": False,
        "documentation": "docs/ARC2AVATAR_TRACK.md",
    }


def run_arc2avatar_inference(**kwargs) -> Dict[str, Any]:
    adapter = Arc2AvatarHeadAdapter()
    return adapter._process_request(kwargs)
