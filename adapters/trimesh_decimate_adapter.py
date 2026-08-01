"""
Trimesh quadric mesh-decimation adapter (MIT).

True triangle decimator for poly reduction without rebuilding topology.
Complements Instant Meshes / AutoRemesher (which remesh to quads).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.mesh_decimate_utils import (
    DEFAULT_TARGET_FACES,
    decimate_mesh,
    get_trimesh_decimate_info,
)
from core.models.base import ModelStatus
from core.models.retopo_models import MeshRetopologyModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class TrimeshDecimateAdapter(MeshRetopologyModel):
    """CPU triangle decimation via trimesh quadric error metrics."""

    MODEL_ID = "trimesh_decimate"
    DEFAULT_TARGET_FACES = DEFAULT_TARGET_FACES

    def __init__(
        self,
        model_id: str = "trimesh_decimate",
        model_path: Optional[str] = None,
        vram_requirement: int = 256,
        default_target_face_count: int = DEFAULT_TARGET_FACES,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path or "trimesh",
            vram_requirement=vram_requirement,
            target_vertex_count=default_target_face_count // 2,
        )
        self.default_target_face_count = default_target_face_count
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        # No weights / binary — validate trimesh API is present.
        import trimesh

        if not hasattr(trimesh.Trimesh, "simplify_quadric_decimation"):
            raise RuntimeError(
                "trimesh.simplify_quadric_decimation is unavailable in this install"
            )
        logger.info("Trimesh decimate ready (quadric)")
        return True

    def _unload_model(self):
        pass

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for mesh decimation")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            output_format = inputs.get("output_format", "glb")
            if output_format not in ["obj", "glb", "ply"]:
                raise ValueError(f"Unsupported output format: {output_format}")

            target_faces = inputs.get("target_face_count", None)
            target_vertices = inputs.get("target_vertex_count", None)
            ratio = inputs.get("ratio", inputs.get("decimate_ratio", None))

            original_mesh = self.mesh_processor.load_mesh(mesh_path)
            original_stats = self.mesh_processor.get_mesh_stats(original_mesh)

            simplified, decimate_info = decimate_mesh(
                original_mesh,
                target_face_count=target_faces,
                target_vertex_count=target_vertices,
                ratio=ratio,
                default_target_faces=self.default_target_face_count,
            )

            base_name = f"{self.model_id}_{mesh_path.stem}"
            output_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )
            self.mesh_processor.save_mesh(simplified, output_path, do_normalise=False)

            final_mesh = self.mesh_processor.load_mesh(output_path)
            output_stats = self.mesh_processor.get_mesh_stats(final_mesh)
            info_path = self.path_generator.generate_info_path(output_path)

            generation_info = {
                "original_stats": original_stats,
                "output_stats": output_stats,
                "decimate_info": decimate_info,
                "model_info": get_trimesh_decimate_info(),
            }
            self.mesh_processor.export_generation_info(generation_info, info_path)

            response = {
                "output_mesh_path": str(output_path),
                "generation_info_path": str(info_path),
                "original_stats": original_stats,
                "output_stats": output_stats,
                "success": True,
                "retopology_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "output_format": output_format,
                    "backend": "trimesh_quadric",
                    "operation": "decimate",
                    **decimate_info,
                },
            }

            logger.info("Trimesh decimate completed: %s", output_path)
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error("Trimesh decimate failed: %s", e)
            raise Exception(f"Trimesh decimate failed: {e}") from e

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {
            "input": ["obj", "glb", "ply", "stl"],
            "output": ["obj", "glb", "ply"],
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "target_face_count": {
                    "type": "integer",
                    "description": "Desired output triangle count (preferred)",
                    "default": self.default_target_face_count,
                    "minimum": 4,
                    "maximum": 5_000_000,
                    "required": False,
                },
                "target_vertex_count": {
                    "type": "integer",
                    "description": "Approx vertex budget (maps to ~2× faces for tri meshes)",
                    "minimum": 100,
                    "maximum": 2_000_000,
                    "required": False,
                },
                "ratio": {
                    "type": "number",
                    "description": "Keep this fraction of faces (0–1]; overrides default when set",
                    "minimum": 0.001,
                    "maximum": 1.0,
                    "required": False,
                },
                "output_format": {
                    "type": "string",
                    "description": "Output mesh format",
                    "default": "glb",
                    "enum": ["obj", "glb", "ply"],
                    "required": False,
                },
            }
        }
