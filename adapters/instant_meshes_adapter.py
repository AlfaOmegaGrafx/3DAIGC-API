"""
Instant Meshes retopology adapter (BSD-3-Clause).

Commercial-safe replacement for FastMesh. Requires a built ``instant-meshes``
binary; see ``scripts/install_instant_meshes.sh``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.instant_meshes_utils import (
    find_instant_meshes_binary,
    get_instant_meshes_info,
    run_instant_meshes,
)
from core.models.base import ModelStatus
from core.models.retopo_models import MeshRetopologyModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class InstantMeshesRetopologyAdapter(MeshRetopologyModel):
    """Quad-dominant retopology via Instant Meshes CLI (CPU)."""

    MODEL_ID = "instant_meshes_retopology"
    DEFAULT_TARGET_VERTICES = 4000

    def __init__(
        self,
        model_id: str = "instant_meshes_retopology",
        model_path: Optional[str] = None,
        vram_requirement: int = 512,
        default_target_vertex_count: int = DEFAULT_TARGET_VERTICES,
    ):
        if model_path is None:
            binary = find_instant_meshes_binary()
            model_path = str(binary) if binary else "thirdparty/instant-meshes/build/instant-meshes"

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            target_vertex_count=default_target_vertex_count,
        )
        self.default_target_vertex_count = default_target_vertex_count
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        binary = find_instant_meshes_binary()
        if binary is None:
            raise FileNotFoundError(
                "Instant Meshes binary not found. Run ./scripts/install_instant_meshes.sh "
                "or set INSTANT_MESHES_BIN to the executable path."
            )
        self.model_path = str(binary)
        logger.info("Instant Meshes binary: %s", binary)
        return binary

    def _unload_model(self):
        pass

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for mesh retopology")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            output_format = inputs.get("output_format", "obj")
            if output_format not in ["obj", "glb", "ply"]:
                raise ValueError(f"Unsupported output format: {output_format}")

            target_vertices = inputs.get("target_vertex_count", None)
            target_faces = inputs.get("target_face_count", None)
            threads = inputs.get("threads", None)
            smooth_iterations = int(inputs.get("smooth_iterations", 2))

            # Instant Meshes accepts only one of -v / -f / -s at a time.
            if target_faces is not None:
                target_vertices = None
            elif target_vertices is None:
                target_vertices = self.default_target_vertex_count

            original_mesh = self.mesh_processor.load_mesh(mesh_path)
            original_stats = self.mesh_processor.get_mesh_stats(original_mesh)

            base_name = f"{self.model_id}_{mesh_path.stem}"
            final_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )
            cli_out = final_path.with_suffix(".obj")

            run_instant_meshes(
                mesh_path,
                cli_out,
                target_vertex_count=target_vertices,
                target_face_count=target_faces,
                threads=threads,
                smooth_iterations=smooth_iterations,
            )

            if output_format == "obj":
                output_path = cli_out
            else:
                converted = self.mesh_processor.load_mesh(cli_out)
                self.mesh_processor.save_mesh(converted, final_path)
                output_path = final_path

            final_mesh = self.mesh_processor.load_mesh(output_path)
            output_stats = self.mesh_processor.get_mesh_stats(final_mesh)
            info_path = self.path_generator.generate_info_path(output_path)

            generation_info = {
                "original_stats": original_stats,
                "output_stats": output_stats,
                "target_vertex_count": target_vertices,
                "target_face_count": target_faces,
                "model_info": get_instant_meshes_info(),
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
                    "target_vertex_count": target_vertices,
                    "backend": "instant_meshes",
                },
            }

            logger.info("Instant Meshes retopology completed: %s", output_path)
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error("Instant Meshes retopology failed: %s", e)
            raise Exception(f"Instant Meshes retopology failed: {e}") from e

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {
            "input": ["obj", "glb", "ply", "stl"],
            "output": ["obj", "glb", "ply"],
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "target_vertex_count": {
                    "type": "integer",
                    "description": "Desired output vertex count (-v)",
                    "default": self.default_target_vertex_count,
                    "minimum": 100,
                    "maximum": 20000,
                    "required": False,
                },
                "target_face_count": {
                    "type": "integer",
                    "description": "Desired output face count (-f); overrides vertex target if set",
                    "minimum": 100,
                    "maximum": 500000,
                    "required": False,
                },
                "threads": {
                    "type": "integer",
                    "description": "Worker threads (-t)",
                    "minimum": 1,
                    "maximum": 128,
                    "required": False,
                },
                "smooth_iterations": {
                    "type": "integer",
                    "description": "Smoothing / reprojection steps (-S)",
                    "default": 2,
                    "minimum": 0,
                    "maximum": 10,
                    "required": False,
                },
                "output_format": {
                    "type": "string",
                    "description": "Output mesh format",
                    "default": "obj",
                    "enum": ["obj", "glb", "ply"],
                    "required": False,
                },
            }
        }
