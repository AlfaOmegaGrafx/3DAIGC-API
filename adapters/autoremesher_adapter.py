"""
AutoRemesher retopology adapter (MIT).

Better on organic / character meshes than Instant Meshes. Requires a built
``autoremesher`` binary; see ``scripts/install_autoremesher.sh``.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.autoremesher_utils import (
    AUTOREMESHER_INPUT_EXTS,
    find_autoremesher_binary,
    get_autoremesher_info,
    prepare_mesh_for_autoremesher,
    run_autoremesher,
    vertex_count_to_target_quads,
)
from core.models.base import ModelStatus
from core.models.retopo_models import MeshRetopologyModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class AutoRemesherRetopologyAdapter(MeshRetopologyModel):
    """Quad-dominant retopology via AutoRemesher CLI (CPU + OpenGL offscreen)."""

    MODEL_ID = "autoremesher_retopology"
    DEFAULT_TARGET_QUADS = 8000

    def __init__(
        self,
        model_id: str = "autoremesher_retopology",
        model_path: Optional[str] = None,
        vram_requirement: int = 512,
        default_target_quad_count: int = DEFAULT_TARGET_QUADS,
    ):
        if model_path is None:
            binary = find_autoremesher_binary()
            model_path = (
                str(binary) if binary else "thirdparty/autoremesher/autoremesher"
            )

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            target_vertex_count=default_target_quad_count,
        )
        self.default_target_quad_count = default_target_quad_count
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        binary = find_autoremesher_binary()
        if binary is None:
            raise FileNotFoundError(
                "AutoRemesher binary not found. Run ./scripts/install_autoremesher.sh "
                "or set AUTOREMESHER_BIN to the executable path."
            )
        self.model_path = str(binary)
        logger.info("AutoRemesher binary: %s", binary)
        return binary

    def _unload_model(self):
        pass

    def _cli_input_for_mesh(
        self, mesh_path: Path, mesh
    ) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
        """
        Always write a cleaned OBJ for AutoRemesher.

        The CLI only loads ``.obj`` and aborts on multi-component AIGC debris
        (``Found repeated halfedge`` → SIGABRT). Preprocess keeps the largest
        connected component.
        """
        cleaned, preprocess_meta = prepare_mesh_for_autoremesher(mesh)
        tmp_dir = Path(tempfile.mkdtemp(prefix="ar_in_"))
        cli_input = tmp_dir / f"{mesh_path.stem}.obj"
        self.mesh_processor.save_mesh(cleaned, cli_input, do_normalise=False)
        logger.info(
            "Prepared %s → %s for AutoRemesher (native: %s, preprocess=%s)",
            mesh_path,
            cli_input,
            sorted(AUTOREMESHER_INPUT_EXTS),
            preprocess_meta.get("preprocess"),
        )
        return cli_input, tmp_dir, preprocess_meta

    def _resolve_target_quads(self, inputs: Dict[str, Any]) -> int:
        if inputs.get("target_quad_count") is not None:
            return max(100, int(inputs["target_quad_count"]))
        if inputs.get("target_vertex_count") is not None:
            return vertex_count_to_target_quads(int(inputs["target_vertex_count"]))
        return self.default_target_quad_count

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tmp_dir: Optional[Path] = None
        try:
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for mesh retopology")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            output_format = inputs.get("output_format", "obj")
            if output_format not in ["obj", "glb", "ply"]:
                raise ValueError(f"Unsupported output format: {output_format}")

            target_quads = self._resolve_target_quads(inputs)
            edge_scaling = float(inputs.get("edge_scaling", 1.0))
            sharp_edge_degrees = float(inputs.get("sharp_edge_degrees", 90.0))
            smooth_normal_degrees = float(inputs.get("smooth_normal_degrees", 0.0))
            adaptivity = float(inputs.get("adaptivity", 1.0))

            original_mesh = self.mesh_processor.load_mesh(mesh_path)
            original_stats = self.mesh_processor.get_mesh_stats(original_mesh)

            base_name = f"{self.model_id}_{mesh_path.stem}"
            final_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )
            cli_out = final_path.with_suffix(".obj")
            cli_input, tmp_dir, preprocess_meta = self._cli_input_for_mesh(
                mesh_path, original_mesh
            )
            report_path = self.path_generator.generate_info_path(final_path).with_suffix(
                ".txt"
            )

            run_autoremesher(
                cli_input,
                cli_out,
                target_quads=target_quads,
                edge_scaling=edge_scaling,
                sharp_edge_degrees=sharp_edge_degrees,
                smooth_normal_degrees=smooth_normal_degrees,
                adaptivity=adaptivity,
                report_path=report_path,
            )

            if output_format == "obj":
                output_path = cli_out
            else:
                converted = self.mesh_processor.load_mesh(cli_out)
                self.mesh_processor.save_mesh(converted, final_path, do_normalise=False)
                output_path = final_path

            final_mesh = self.mesh_processor.load_mesh(output_path)
            output_stats = self.mesh_processor.get_mesh_stats(final_mesh)
            info_path = self.path_generator.generate_info_path(output_path)

            generation_info = {
                "original_stats": original_stats,
                "output_stats": output_stats,
                "target_quad_count": target_quads,
                "model_info": get_autoremesher_info(),
                "preprocess": preprocess_meta,
                "report_path": str(report_path) if report_path.is_file() else None,
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
                    "target_quad_count": target_quads,
                    "edge_scaling": edge_scaling,
                    "sharp_edge_degrees": sharp_edge_degrees,
                    "smooth_normal_degrees": smooth_normal_degrees,
                    "adaptivity": adaptivity,
                    "backend": "autoremesher",
                    "preprocess": preprocess_meta,
                },
            }

            logger.info("AutoRemesher retopology completed: %s", output_path)
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error("AutoRemesher retopology failed: %s", e)
            raise Exception(f"AutoRemesher retopology failed: {e}") from e
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

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
                    "description": "Approximate output vertex budget (maps to target quads)",
                    "default": self.default_target_quad_count,
                    "minimum": 100,
                    "maximum": 200000,
                    "required": False,
                },
                "target_quad_count": {
                    "type": "integer",
                    "description": "AutoRemesher --target-quads (overrides vertex budget if set)",
                    "minimum": 100,
                    "maximum": 200000,
                    "required": False,
                },
                "edge_scaling": {
                    "type": "number",
                    "description": "Edge scaling factor (--edge-scaling, 1.0–4.0)",
                    "default": 1.0,
                    "minimum": 1.0,
                    "maximum": 4.0,
                    "required": False,
                },
                "sharp_edge_degrees": {
                    "type": "number",
                    "description": "Sharp edge dihedral threshold in degrees (--sharp-edge)",
                    "default": 90.0,
                    "minimum": 30.0,
                    "maximum": 180.0,
                    "required": False,
                },
                "smooth_normal_degrees": {
                    "type": "number",
                    "description": "Smooth normal angle threshold in degrees (--smooth-normal)",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 180.0,
                    "required": False,
                },
                "adaptivity": {
                    "type": "number",
                    "description": "Curvature-adaptive quad density (--adaptivity, 0.0–1.0)",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
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
