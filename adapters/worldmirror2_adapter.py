"""
WorldMirror 2.0 adapter — multi-view / video → 3D Gaussian splat (.ply / .splat).

Upstream: https://github.com/Tencent-Hunyuan/HY-World-2.0 (fork: AlfaOmegaGrafx/HY-World-2.0)
Weights: tencent/HY-World-2.0 (subfolder HY-WorldMirror-2.0)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.multi_image_input import (
    collect_local_image_paths,
    multi_image_generation_info,
)
from core.utils.worldmirror2_pipeline import (
    DEFAULT_PRETRAINED,
    DEFAULT_SUBFOLDER,
    ensure_hyworld_on_path,
    find_gaussians_ply,
    ply_to_splat,
    stage_images_directory,
    worldmirror2_available,
)

logger = logging.getLogger(__name__)


class WorldMirror2ReconstructAdapter(ImageToMeshModel):
    """Multi-photo feed-forward reconstruction → Gaussian splat PLY."""

    FEATURE_TYPE = "image_to_splat"
    MODEL_ID = "worldmirror2_reconstruct"

    def __init__(self, **kwargs):
        super().__init__(
            model_id=self.MODEL_ID,
            model_path=str(
                Path(__file__).resolve().parents[1]
                / "thirdparty"
                / "HY-World-2.0"
            ),
            vram_requirement=16384,
            supported_output_formats=["ply", "splat"],
            feature_type=self.FEATURE_TYPE,
            max_images=32,
        )
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        self.pipeline = None
        self.pretrained = kwargs.get("pretrained_model", DEFAULT_PRETRAINED)
        self.subfolder = kwargs.get("subfolder", DEFAULT_SUBFOLDER)

    def _load_model(self):
        if not worldmirror2_available():
            raise RuntimeError(
                "WorldMirror 2.0 unavailable — clone thirdparty/HY-World-2.0 "
                "and install deps (see scripts/install_worldmirror2_deps.sh)"
            )
        ensure_hyworld_on_path()
        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

        logger.info(
            "Loading WorldMirror 2.0 from %s (%s)",
            self.pretrained,
            self.subfolder,
        )
        self.pipeline = WorldMirrorPipeline.from_pretrained(
            self.pretrained,
            subfolder=self.subfolder,
            enable_bf16=True,
        )
        return {"pipeline": self.pipeline}

    def _unload_model(self):
        self.pipeline = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
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
            "WorldMirror unavailable — falling back to TripoSplat on primary image"
        )
        primary_inputs = dict(inputs)
        primary_inputs.pop("image_paths", None)
        primary_inputs.pop("reference_image_paths", None)
        primary_inputs["output_format"] = output_format
        adapter = TripoSplatImageToSplatAdapter()
        result = adapter._process_request(primary_inputs)
        gen = result.get("generation_info") or {}
        gen["multi_image_phase"] = "3_worldmirror_fallback_triposplat"
        gen["worldmirror_available"] = False
        result["generation_info"] = gen
        return result

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" not in inputs:
            raise ValueError("image_path is required")

        output_format = inputs.get("output_format", "ply")
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        image_paths = self._image_paths_from_inputs(inputs)
        if not image_paths:
            raise ValueError("At least one image is required for WorldMirror reconstruction")

        if not worldmirror2_available():
            return self._fallback_triposplat(inputs, output_format)

        if self.pipeline is None:
            self._load_model()

        params = inputs.get("model_parameters") or {}
        target_size = int(params.get("target_size", 952))
        apply_sky_mask = bool(params.get("apply_sky_mask", False))

        stem = Path(image_paths[0]).stem
        output_path = self.path_generator.generate_mesh_path(
            self.model_id,
            stem,
            output_format="ply",
            subdirectory="splats",
        )
        workspace = output_path.parent / f"{stem}_worldmirror2"
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        staged_dir = None
        try:
            staged_dir = stage_images_directory(image_paths)
            outdir = self.pipeline(
                str(staged_dir),
                strict_output_path=str(workspace),
                target_size=target_size,
                apply_sky_mask=apply_sky_mask,
                save_depth=False,
                save_normal=False,
                save_camera=False,
                save_points=False,
                save_colmap=False,
                save_rendered=False,
                save_gs=True,
                log_time=True,
            )
            if not outdir:
                raise RuntimeError("WorldMirror pipeline returned no output directory")

            gs_ply = find_gaussians_ply(Path(outdir))
            if gs_ply is None:
                raise FileNotFoundError(
                    f"gaussians.ply not found under WorldMirror output: {outdir}"
                )

            shutil.copy2(gs_ply, output_path)

            if output_format == "splat":
                splat_path = output_path.with_suffix(".splat")
                ply_to_splat(output_path, splat_path)
                output_path = splat_path

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
                    "worldmirror_available": True,
                    "worldmirror_output_dir": str(outdir),
                    **multi_image_generation_info(
                        primary_file_id=inputs.get("image_file_id"),
                        reference_file_ids=inputs.get("reference_image_file_ids"),
                        phase="3_worldmirror2_reconstruct",
                    ),
                },
            }
        except Exception as exc:
            logger.exception("WorldMirror reconstruction failed")
            if len(image_paths) >= 1:
                logger.warning("Falling back to TripoSplat after WorldMirror error: %s", exc)
                return self._fallback_triposplat(inputs, output_format)
            raise
        finally:
            if staged_dir and staged_dir.exists():
                shutil.rmtree(staged_dir, ignore_errors=True)

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "target_size": {
                    "type": "integer",
                    "description": "Max inference resolution (WorldMirror adaptive sizing)",
                    "default": 952,
                    "minimum": 518,
                    "maximum": 1920,
                    "required": False,
                },
                "apply_sky_mask": {
                    "type": "boolean",
                    "description": "Mask sky regions (requires optional ONNX/ZIM deps)",
                    "default": False,
                    "required": False,
                },
            }
        }
