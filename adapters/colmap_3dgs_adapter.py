"""
COLMAP + 3DGS reconstruction adapter (Phase 3).

Uses COLMAP sparse reconstruction when the ``colmap`` binary is installed.
Falls back to TripoSplat on the primary image when COLMAP is unavailable.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.mesh_models import ImageToMeshModel
from core.utils.colmap_3dgs_pipeline import colmap_available, reconstruct_photos_to_ply
from core.utils.file_utils import OutputPathGenerator
from core.utils.multi_image_input import (
    collect_local_image_paths,
    multi_image_generation_info,
    should_use_colmap_reconstruction,
)

logger = logging.getLogger(__name__)


class Colmap3dgsReconstructAdapter(ImageToMeshModel):
    """Multi-photo photogrammetry → merged splat (.ply / .splat)."""

    FEATURE_TYPE = "colmap_3dgs_reconstruct"
    MODEL_ID = "colmap_3dgs_reconstruct"

    def __init__(self, **kwargs):
        super().__init__(
            model_id=self.MODEL_ID,
            model_path="",
            vram_requirement=8192,
            supported_output_formats=["ply", "splat"],
            feature_type=self.FEATURE_TYPE,
            max_images=64,
        )
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        return {"colmap": colmap_available()}

    def _unload_model(self):
        pass

    def _image_paths_from_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        if inputs.get("image_paths"):
            return collect_local_image_paths(
                str(inputs["image_path"]),
                inputs["image_paths"][1:],
            )
        primary = str(inputs["image_path"])
        refs = inputs.get("reference_image_paths") or []
        return collect_local_image_paths(primary, refs)

    def _fallback_triposplat(self, inputs: Dict[str, Any], output_format: str) -> Dict[str, Any]:
        from adapters.triposplat_adapter import TripoSplatImageToSplatAdapter

        logger.warning(
            "COLMAP unavailable — falling back to TripoSplat on primary image only"
        )
        primary_inputs = dict(inputs)
        primary_inputs.pop("image_paths", None)
        primary_inputs.pop("reference_image_paths", None)
        primary_inputs["output_format"] = output_format
        adapter = TripoSplatImageToSplatAdapter()
        result = adapter._process_request(primary_inputs)
        gen = result.get("generation_info") or {}
        gen["multi_image_phase"] = "3_colmap_fallback_triposplat"
        gen["colmap_available"] = False
        result["generation_info"] = gen
        return result

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" not in inputs:
            raise ValueError("image_path is required")

        output_format = inputs.get("output_format", "ply")
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        image_paths = self._image_paths_from_inputs(inputs)
        if len(image_paths) < 3 and not should_use_colmap_reconstruction(
            image_paths, inputs
        ):
            raise ValueError(
                "COLMAP reconstruction requires at least 3 images "
                f"(got {len(image_paths)})"
            )

        if not colmap_available():
            return self._fallback_triposplat(inputs, output_format)

        stem = Path(image_paths[0]).stem
        output_path = self.path_generator.generate_mesh_path(
            self.model_id,
            stem,
            output_format="ply",
            subdirectory="splats",
        )
        workspace = output_path.parent / f"{stem}_colmap"

        metrics = reconstruct_photos_to_ply(image_paths, output_path, workspace=workspace)

        if output_format == "splat":
            splat_path = output_path.with_suffix(".splat")
            try:
                from utils.triposplat_pipeline_helper import TripoSplatPipelineHelper

                helper = TripoSplatPipelineHelper()
                helper.load()
                # Re-use TripoSplat exporter if we add ply→splat later; copy ply for now.
                shutil.copy2(output_path, splat_path)
                output_path = splat_path
            except Exception as exc:
                logger.warning("PLY→splat conversion skipped: %s", exc)

        return {
            "output_mesh_path": str(output_path),
            "output_splat_path": str(output_path),
            "success": True,
            "generation_info": {
                "model": self.model_id,
                "input_image": image_paths[0],
                "image_count": len(image_paths),
                "output_format": output_format,
                "renderer_hint": "spark",
                "colmap_available": True,
                **metrics,
                **multi_image_generation_info(
                    primary_file_id=inputs.get("image_file_id"),
                    reference_file_ids=inputs.get("reference_image_file_ids"),
                    phase="3_colmap_reconstruct",
                ),
            },
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {"parameters": {}}
