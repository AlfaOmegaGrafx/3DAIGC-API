"""
Image-to-World adapter — DGX-local explorable environment pipeline.

Environment: TripoSplat Gaussian splat (.ply)
Props: optional TRELLIS.2 meshes from normalized bbox crops on the source image
Output: world.manifest.json + assets under outputs/worlds/{job_id}/
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel
from utils.triposplat_pipeline_helper import TripoSplatPipelineHelper

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_MANIFEST_VERSION = 1


def _crop_bbox(image: Image.Image, bbox: List[float]) -> Image.Image:
    """Crop using normalized bbox [x, y, w, h] in 0–1."""
    if len(bbox) < 4:
        raise ValueError("bbox must have 4 values: x, y, w, h (normalized 0–1)")
    w, h = image.size
    x0 = int(max(0, min(1, bbox[0])) * w)
    y0 = int(max(0, min(1, bbox[1])) * h)
    x1 = int(max(0, min(1, bbox[0] + bbox[2])) * w)
    y1 = int(max(0, min(1, bbox[1] + bbox[3])) * h)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return image.crop((x0, y0, x1, y1))


def _bbox_to_world_position(bbox: List[float], scale: float = 2.0) -> List[float]:
    """Map normalized bbox center to rough world coordinates (y-up)."""
    cx = bbox[0] + bbox[2] * 0.5
    cy = bbox[1] + bbox[3] * 0.5
    x = (cx - 0.5) * scale
    z = (0.5 - cy) * scale
    return [round(x, 3), 0.0, round(z, 3)]


class ImageToWorldAdapter(ImageToMeshModel):
    """Generate a World Package from a single reference image."""

    FEATURE_TYPE = "image_to_world"
    MODEL_ID = "opennexus_image_to_world"

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 20480,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = str(REPO_ROOT / "outputs" / "worlds")
        if feature_type is None:
            feature_type = self.FEATURE_TYPE
        if supported_output_formats is None:
            supported_output_formats = ["json"]

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=supported_output_formats,
            feature_type=feature_type,
            max_images=1,
        )
        self.triposplat_helper: Optional[TripoSplatPipelineHelper] = None

    def _load_model(self):
        return {"ready": True}

    def _unload_model(self):
        if self.triposplat_helper is not None:
            self.triposplat_helper.unload()
            self.triposplat_helper = None

    def _run_triposplat(self, image_path: Path, output_ply: Path, params: Dict[str, Any]) -> None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.triposplat_helper is None:
            self.triposplat_helper = TripoSplatPipelineHelper(device=device)
        self.triposplat_helper.load()
        gaussian = self.triposplat_helper.run(
            str(image_path),
            seed=int(params.get("seed", 42)),
            steps=int(params.get("steps", 20)),
            guidance_scale=float(params.get("guidance_scale", 3.0)),
            shift=float(params.get("shift", 3.0)),
            num_gaussians=int(params.get("num_gaussians", 131072)),
            erode_radius=int(params.get("erode_radius", 1)),
        )
        gaussian.save_ply(str(output_ply))
        self.triposplat_helper.unload()
        self.triposplat_helper = None

    def _run_trellis2_prop(
        self, crop_path: Path, output_glb: Path, mesh_model: str, params: Dict[str, Any]
    ) -> None:
        from adapters.trellis2_adapter import Trellis2ImageToTexturedMeshAdapter

        adapter = Trellis2ImageToTexturedMeshAdapter()
        adapter.load(0)
        try:
            result = adapter._process_request(
                {
                    "image_path": str(crop_path),
                    "output_format": "glb",
                    "texture_size": int(params.get("texture_size", 1024)),
                    "decimation_target": int(params.get("decimation_target", 100000)),
                }
            )
            src = Path(result["output_mesh_path"])
            if src.resolve() != output_glb.resolve():
                shutil.copy2(src, output_glb)
        finally:
            adapter.unload()

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" not in inputs:
            raise ValueError("image_path is required for image-to-world generation")

        image_path = Path(inputs["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        world_id = inputs.get("world_id") or f"world-{uuid.uuid4().hex[:12]}"
        world_name = inputs.get("world_name") or image_path.stem
        prop_regions: List[Dict[str, Any]] = inputs.get("prop_regions") or []
        splat_params = inputs.get("splat_parameters") or {}
        prop_mesh_model = inputs.get(
            "prop_mesh_model_preference", "trellis2_image_to_textured_mesh"
        )
        prop_mesh_params = inputs.get("prop_mesh_parameters") or {}

        job_tag = str(inputs.get("job_id") or uuid.uuid4().hex[:12])
        world_dir = REPO_ROOT / "outputs" / "worlds" / job_tag
        props_dir = world_dir / "props"
        world_dir.mkdir(parents=True, exist_ok=True)
        props_dir.mkdir(parents=True, exist_ok=True)

        ref_copy = world_dir / f"reference{image_path.suffix.lower() or '.png'}"
        shutil.copy2(image_path, ref_copy)

        env_ply = world_dir / "environment.ply"
        logger.info("Image-to-world: generating environment splat for %s", image_path)
        self._run_triposplat(image_path, env_ply, splat_params)

        props_manifest: List[Dict[str, Any]] = []
        if prop_regions:
            base_image = Image.open(image_path).convert("RGB")
            for index, region in enumerate(prop_regions):
                prop_id = region.get("id") or f"prop_{index + 1}"
                bbox = region.get("bbox")
                if not bbox or len(bbox) < 4:
                    logger.warning("Skipping prop %s: invalid bbox", prop_id)
                    continue
                try:
                    crop = _crop_bbox(base_image, bbox)
                    crop_path = props_dir / f"{prop_id}_crop.png"
                    crop.save(crop_path)
                    glb_path = props_dir / f"{prop_id}.glb"
                    logger.info("Image-to-world: mesh prop %s from crop %s", prop_id, crop_path)
                    if prop_mesh_model.startswith("trellis2"):
                        self._run_trellis2_prop(crop_path, glb_path, prop_mesh_model, prop_mesh_params)
                    else:
                        logger.warning(
                            "Unsupported prop_mesh_model_preference %s; skipping %s",
                            prop_mesh_model,
                            prop_id,
                        )
                        continue
                    props_manifest.append(
                        {
                            "id": prop_id,
                            "role": region.get("role", "interactable"),
                            "mesh_url": f"props/{prop_id}.glb",
                            "transform": {
                                "position": region.get("position")
                                or _bbox_to_world_position(bbox),
                                "rotation_y": float(region.get("rotation_y", 0)),
                                "scale": float(region.get("scale", 1)),
                            },
                            "interaction": region.get(
                                "interaction",
                                {"type": "grabbable", "collider": "auto_bbox"},
                            ),
                            "generation": {
                                "model": prop_mesh_model,
                                "crop_ref": f"props/{prop_id}_crop.png",
                            },
                        }
                    )
                except Exception as exc:
                    logger.exception("Prop %s failed: %s", prop_id, exc)

        manifest = {
            "id": world_id,
            "version": WORLD_MANIFEST_VERSION,
            "name": world_name,
            "source_image": ref_copy.name,
            "coordinate_system": "y-up",
            "spawn": inputs.get(
                "spawn",
                {"position": [0, 0, 0], "rotation_y": 0, "player_height": 1.6},
            ),
            "environment": {
                "type": "gaussian_splat",
                "url": "environment.ply",
                "format": "ply",
                "renderer": "spark",
                "transform": {"position": [0, 0, 0], "rotation": [1, 0, 0, 0], "scale": 1},
            },
            "props": props_manifest,
            "audio": {},
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pipeline": "image-to-world-v1",
                "prop_count": len(props_manifest),
            },
        }

        manifest_path = world_dir / "world.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        world_base_url = f"/api/v1/system/jobs/{job_tag}/world/"

        self.status = ModelStatus.LOADED
        return {
            "success": True,
            "world_id": world_id,
            "world_manifest_path": str(manifest_path),
            "world_manifest_url": f"/api/v1/system/jobs/{job_tag}/download?asset=manifest",
            "world_base_url": world_base_url,
            "output_splat_path": str(env_ply),
            "output_mesh_path": str(env_ply),
            "world_directory": str(world_dir),
            "pipeline": "image-to-world",
            "feature": "image_to_world",
            "generation_info": {
                "world_id": world_id,
                "prop_count": len(props_manifest),
                "environment_format": "ply",
            },
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "world_name": {
                    "type": "string",
                    "description": "Display name for the generated world",
                    "required": False,
                },
                "prop_regions": {
                    "type": "array",
                    "description": "Optional prop bbox crops [{ id, bbox: [x,y,w,h] }]",
                    "required": False,
                },
                "prop_mesh_model_preference": {
                    "type": "string",
                    "description": "Model for prop mesh generation",
                    "default": "trellis2_image_to_textured_mesh",
                    "required": False,
                },
                "splat_parameters": {
                    "type": "object",
                    "description": "TripoSplat params (steps, guidance_scale, num_gaussians, etc.)",
                    "required": False,
                },
            }
        }
