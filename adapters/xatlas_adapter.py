"""
xatlas UV unwrapping adapter (MIT license).

Commercial-safe replacement for PartUV (no PartField weights).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.xatlas_utils import XAtlasUnwrapper
from core.models.base import ModelStatus
from core.models.uv_models import UVUnwrappingModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class XatlasUVUnwrappingAdapter(UVUnwrappingModel):
    """CPU xatlas atlas parametrization for mesh UVs."""

    MODEL_ID = "xatlas_uv_unwrapping"

    def __init__(
        self,
        model_id: str = "xatlas_uv_unwrapping",
        model_path: Optional[str] = None,
        vram_requirement: int = 512,
        resolution: int = 1024,
    ):
        if model_path is None:
            model_path = "pip:xatlas"

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_input_formats=["obj", "glb", "ply", "stl"],
            supported_output_formats=["obj", "glb"],
        )
        self.resolution = resolution
        self.unwrapper: Optional[XAtlasUnwrapper] = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        self.unwrapper = XAtlasUnwrapper(resolution=self.resolution)
        logger.info("xatlas UV unwrapper ready (CPU)")
        return self.unwrapper

    def _unload_model(self):
        self.unwrapper = None

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for UV unwrapping")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            output_format = inputs.get("output_format", "obj")
            if output_format not in ["obj", "glb"]:
                raise ValueError(f"Unsupported output format: {output_format}")

            if self.unwrapper is None:
                self._load_model()

            original_mesh = self.mesh_processor.load_mesh(mesh_path)
            original_stats = self.mesh_processor.get_mesh_stats(original_mesh)

            base_name = f"{self.model_id}_{mesh_path.stem}"
            output_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )

            result = self.unwrapper.unwrap_mesh(
                mesh_path,
                output_path,
                output_format=output_format,
            )

            output_mesh_path = Path(result["output_mesh_path"])
            output_mesh = self.mesh_processor.load_mesh(output_mesh_path)
            output_stats = self.mesh_processor.get_mesh_stats(output_mesh)

            metadata = result.get("metadata", {})
            generation_info = {
                "original_stats": original_stats,
                "output_stats": output_stats,
                "num_components": result.get("num_components", 0),
                "model_info": self.unwrapper.get_model_info(),
                "metadata": metadata,
            }
            info_path = self.path_generator.generate_info_path(output_mesh_path)
            self.mesh_processor.export_generation_info(generation_info, info_path)

            response = {
                "output_mesh_path": str(output_mesh_path),
                "packed_mesh_path": None,
                "individual_parts_dir": None,
                "generation_info_path": str(info_path),
                "num_components": result.get("num_components", 0),
                "distortion": result.get("distortion", 0.0),
                "success": True,
                "uv_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "output_format": output_format,
                    "original_vertices": original_stats["vertex_count"],
                    "original_faces": original_stats["face_count"],
                    "num_uv_components": result.get("num_components", 0),
                    "backend": "xatlas",
                },
            }

            logger.info("xatlas UV unwrapping completed: %s", output_mesh_path)
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error("xatlas UV unwrapping failed: %s", e)
            raise Exception(f"xatlas UV unwrapping failed: {e}") from e

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {"input": ["obj", "glb", "ply", "stl"], "output": ["obj", "glb"]}

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "output_format": {
                    "type": "string",
                    "description": "Output mesh format",
                    "default": "obj",
                    "enum": ["obj", "glb"],
                    "required": False,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Atlas resolution hint (reserved for future packing)",
                    "default": self.resolution,
                    "minimum": 256,
                    "maximum": 8192,
                    "required": False,
                },
            }
        }
