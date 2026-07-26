"""
LingBot-Map environment scan adapter (Galaxy XR walk → digital twin).

Optional: only active when ``thirdparty/lingbot-map`` (or pip) is installed.
Does not replace ``opennexus_image_to_world``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.lingbot_map_pipeline import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MAX_FRAMES,
    HARD_CAP_MAX_FRAMES,
    clamp_env_scan_frame_budget,
    lingbot_map_available,
    lingbot_map_status,
    run_environment_scan,
)

logger = logging.getLogger(__name__)


class LingBotMapEnvironmentScanAdapter(ImageToMeshModel):
    """Streaming RGB walk-scan → metric world package (point cloud + manifest)."""

    FEATURE_TYPE = "lingbot_map_environment_scan"
    MODEL_ID = "lingbot_map_environment_scan"

    def __init__(self, **kwargs):
        super().__init__(
            model_id=self.MODEL_ID,
            model_path="thirdparty/lingbot-map",
            vram_requirement=16384,
            supported_output_formats=["ply", "json"],
            feature_type=self.FEATURE_TYPE,
            max_images=2048,
        )
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        status = lingbot_map_status()
        if not status["available"]:
            logger.warning(
                "LingBot-Map not installed — jobs will fail until: %s",
                status["install_hint"],
            )
        return status

    def _unload_model(self):
        pass

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "world_name": {
                    "type": "string",
                    "description": "Display name for the scanned world",
                    "required": False,
                },
                "max_frames": {
                    "type": "integer",
                    "description": (
                        f"Max frames to sample (cap {HARD_CAP_MAX_FRAMES}). "
                        "Long walks use windowed CPU-resident inference on GB10."
                    ),
                    "default": DEFAULT_MAX_FRAMES,
                    "minimum": 3,
                    "maximum": HARD_CAP_MAX_FRAMES,
                    "required": False,
                },
                "frame_stride": {
                    "type": "integer",
                    "description": "Keep every Nth frame when extracting from video",
                    "default": DEFAULT_FRAME_STRIDE,
                    "minimum": 1,
                    "maximum": 30,
                    "required": False,
                },
                "metric_calibration": {
                    "type": "object",
                    "description": (
                        "1:1 scale: mode=reference_length|two_points|player_height|auto_bbox "
                        "with true_meters / recon_length as needed"
                    ),
                    "required": False,
                },
            }
        }

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not lingbot_map_available():
            raise RuntimeError(
                "LingBot-Map is not installed on this server. "
                "Run: bash scripts/install_lingbot_map.sh "
                "(default image-to-world / TripoSplat paths are unchanged)."
            )

        job_id = str(inputs.get("job_id") or inputs.get("world_id") or uuid.uuid4())
        video_path = inputs.get("video_path")
        frame_dir = inputs.get("frame_dir")
        image_paths: Optional[List[str]] = None
        if inputs.get("image_paths"):
            image_paths = list(inputs["image_paths"])
        elif inputs.get("image_path") and inputs.get("reference_image_paths"):
            image_paths = [str(inputs["image_path"])] + [
                str(p) for p in inputs["reference_image_paths"]
            ]
        elif inputs.get("image_path") and not video_path and not frame_dir:
            # Single image is not enough — require video or multi-frame
            raise ValueError(
                "environment scan needs a walk video (video_path), frame_dir, "
                "or ≥3 images (image_paths). Capture with Galaxy XR outward cameras "
                "while walking the space."
            )

        metric_calibration = inputs.get("metric_calibration")
        max_frames, stride = clamp_env_scan_frame_budget(
            int(inputs.get("max_frames") or DEFAULT_MAX_FRAMES),
            int(inputs.get("frame_stride") or DEFAULT_FRAME_STRIDE),
        )
        refine_to_3dgs = bool(inputs.get("refine_to_3dgs") or False)

        result = run_environment_scan(
            job_id=job_id,
            video_path=video_path,
            frame_dir=frame_dir,
            image_paths=image_paths,
            metric_calibration=metric_calibration,
            world_name=inputs.get("world_name") or inputs.get("object_name"),
            max_frames=max_frames,
            stride=stride,
            output_root=Path("outputs") / "worlds" / job_id,
            refine_to_3dgs=refine_to_3dgs,
        )

        return {
            "success": True,
            "job_id": job_id,
            "world_directory": result["world_directory"],
            "world_manifest_url": result.get("world_manifest_url"),
            "world_base_url": result.get("world_base_url"),
            "output_splat_path": result.get("output_splat_path"),
            "output_mesh_path": result.get("output_splat_path"),
            "mesh_url": result.get("output_splat_path"),
            "generation_info": {
                "pipeline": "lingbot_map_environment_scan",
                "metric_calibration": result.get("metric_calibration"),
                "lingbot_map": lingbot_map_status(),
                "one_to_one_meters": bool(
                    (result.get("metric_calibration") or {}).get("one_to_one")
                ),
                "gaussian_refine": result.get("gaussian_refine"),
            },
        }
