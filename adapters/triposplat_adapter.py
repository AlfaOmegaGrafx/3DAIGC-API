"""
TripoSplat adapter — single image to 3D Gaussian splats (.ply / .splat).

Upstream: https://github.com/VAST-AI-Research/TripoSplat (MIT)
Weights: https://huggingface.co/VAST-AI/TripoSplat
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from utils.triposplat_pipeline_helper import (
    DEFAULT_CKPT_ROOT,
    DEFAULT_TRIPOSPLAT_ROOT,
    TripoSplatPipelineHelper,
)

logger = logging.getLogger(__name__)


class TripoSplatImageToSplatAdapter(ImageToMeshModel):
    """Generate 3D Gaussian splats from a single image using TripoSplat."""

    FEATURE_TYPE = "image_to_splat"
    MODEL_ID = "triposplat_image_to_splat"

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 16384,
        triposplat_root: Optional[str] = None,
        ckpt_root: Optional[str] = None,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = str(DEFAULT_TRIPOSPLAT_ROOT)
        if feature_type is None:
            feature_type = self.FEATURE_TYPE
        if supported_output_formats is None:
            supported_output_formats = ["ply", "splat"]

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=supported_output_formats,
            feature_type=feature_type,
            max_images=1,
        )

        self.triposplat_root = Path(triposplat_root or DEFAULT_TRIPOSPLAT_ROOT)
        self.ckpt_root = Path(ckpt_root or DEFAULT_CKPT_ROOT)
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        self.pipeline_helper: Optional[TripoSplatPipelineHelper] = None

    def _load_model(self):
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline_helper = TripoSplatPipelineHelper(
                triposplat_root=self.triposplat_root,
                ckpt_root=self.ckpt_root,
                device=device,
            )
            self.pipeline_helper.load()
            return {"pipeline": self.pipeline_helper}
        except Exception as exc:
            logger.exception("Failed to load TripoSplat")
            raise exc

    def _unload_model(self):
        if self.pipeline_helper is not None:
            self.pipeline_helper.unload()
            self.pipeline_helper = None

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" not in inputs:
            raise ValueError("image_path is required for image-to-splat generation")

        image_path = Path(inputs["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        output_format = inputs.get("output_format", "ply")
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        seed = int(inputs.get("seed", 42))
        steps = int(inputs.get("steps", 20))
        guidance_scale = float(inputs.get("guidance_scale", 3.0))
        shift = float(inputs.get("shift", 3.0))
        num_gaussians = int(inputs.get("num_gaussians", 131072))
        erode_radius = int(inputs.get("erode_radius", 1))

        if self.pipeline_helper is None:
            self._load_model()

        logger.info(
            "TripoSplat inference: image=%s gaussians=%s format=%s",
            image_path,
            num_gaussians,
            output_format,
        )

        gaussian = self.pipeline_helper.run(
            str(image_path),
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            shift=shift,
            num_gaussians=num_gaussians,
            erode_radius=erode_radius,
        )

        output_path = self.path_generator.generate_mesh_path(
            self.model_id,
            image_path.stem,
            output_format=output_format,
            subdirectory="splats",
        )

        if output_format == "ply":
            gaussian.save_ply(str(output_path))
        elif output_format == "splat":
            gaussian.save_splat(str(output_path))
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        num_splats = getattr(gaussian, "num_splats", None)
        if num_splats is None and hasattr(gaussian, "get_xyz"):
            try:
                num_splats = int(gaussian.get_xyz.shape[0])
            except Exception:
                num_splats = num_gaussians

        response = {
            "output_mesh_path": str(output_path),
            "output_splat_path": str(output_path),
            "success": True,
            "generation_info": {
                "model": self.model_id,
                "input_image": str(image_path),
                "output_format": output_format,
                "num_gaussians": num_gaussians,
                "num_splats": num_splats,
                "seed": seed,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "renderer_hint": "spark",
            },
        }

        self.status = ModelStatus.LOADED
        logger.info("TripoSplat completed: %s", output_path)
        return response

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "num_gaussians": {
                    "type": "integer",
                    "description": "Target Gaussian count (32768–262144, rounded to multiple of 32)",
                    "default": 131072,
                    "minimum": 32768,
                    "maximum": 262144,
                    "required": False,
                },
                "steps": {
                    "type": "integer",
                    "description": "Flow-matching sampler steps (10–50 recommended)",
                    "default": 20,
                    "minimum": 5,
                    "maximum": 100,
                    "required": False,
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "Classifier-free guidance strength",
                    "default": 3.0,
                    "minimum": 1.0,
                    "maximum": 10.0,
                    "required": False,
                },
                "shift": {
                    "type": "number",
                    "description": "Flow-matching timestep schedule shift",
                    "default": 3.0,
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "required": False,
                },
                "seed": {
                    "type": "integer",
                    "description": "RNG seed",
                    "default": 42,
                    "required": False,
                },
                "erode_radius": {
                    "type": "integer",
                    "description": "Alpha matte erosion after background removal",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 5,
                    "required": False,
                },
            }
        }

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {
            "input": self.supported_input_formats,
            "output": self.supported_output_formats,
        }
