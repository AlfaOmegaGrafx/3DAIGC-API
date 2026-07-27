"""
World generation API — image → explorable splat environment + optional mesh props.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.dependencies import get_current_user_or_none, get_file_store, get_scheduler
from api.routers.mesh_generation import (
    MeshGenerationResponse,
    process_file_input,
    validate_model_preference,
)
from core.file_store import FileStore
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/world-generation", tags=["world_generation"])


class PropRegion(BaseModel):
    id: str = Field(..., description="Unique prop identifier")
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Normalized crop [x, y, w, h] in 0–1",
    )
    role: str = Field("interactable", description="Prop role in the world")
    position: Optional[List[float]] = Field(None, description="Optional world position override")
    rotation_y: float = Field(0, description="Yaw rotation in radians")
    scale: float = Field(1, description="Uniform scale")
    interaction: Optional[Dict[str, Any]] = None


class ImageToWorldRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    image_file_id: Optional[str] = None
    world_id: Optional[str] = Field(None, description="Stable world package id")
    world_name: Optional[str] = Field(None, description="Display name")
    model_preference: str = Field(
        "opennexus_image_to_world",
        description="World generation orchestrator model",
    )
    prop_regions: List[PropRegion] = Field(
        default_factory=list,
        description="Optional interactable prop regions (normalized bbox crops)",
    )
    prop_mesh_model_preference: str = Field(
        "trellis2_image_to_textured_mesh",
        description="Mesh model for prop generation",
    )
    splat_parameters: Optional[Dict[str, Any]] = None
    prop_mesh_parameters: Optional[Dict[str, Any]] = None
    spawn: Optional[Dict[str, Any]] = None

    @field_validator("image_file_id")
    @classmethod
    def validate_inputs(cls, v, info):
        image_path = info.data.get("image_path")
        image_base64 = info.data.get("image_base64")
        inputs_provided = sum(bool(x) for x in [image_path, image_base64, v])
        if inputs_provided == 0:
            raise ValueError(
                "One of image_path, image_base64, or image_file_id must be provided"
            )
        if inputs_provided > 1:
            raise ValueError(
                "Only one of image_path, image_base64, or image_file_id should be provided"
            )
        return v

    model_config = ConfigDict(protected_namespaces=("settings_",))


@router.post("/image-to-world", response_model=MeshGenerationResponse)
async def image_to_world(
    request: ImageToWorldRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user=Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """Generate a World Package: TripoSplat environment + optional TRELLIS.2 props."""
    try:
        user_id = current_user.user_id if current_user else None

        validate_model_preference(
            request.model_preference, "image_to_world", scheduler
        )

        image_file_path = await process_file_input(
            file_path=request.image_path,
            base64_data=request.image_base64,
            file_id=request.image_file_id,
            input_type="image",
            file_store=file_store,
        )

        prop_regions = [r.model_dump() for r in request.prop_regions]

        job_request = JobRequest(
            feature="image_to_world",
            inputs={
                "image_path": image_file_path,
                "world_id": request.world_id,
                "world_name": request.world_name,
                "prop_regions": prop_regions,
                "prop_mesh_model_preference": request.prop_mesh_model_preference,
                "splat_parameters": request.splat_parameters or {},
                "prop_mesh_parameters": request.prop_mesh_parameters or {},
                "spawn": request.spawn,
            },
            model_preference=request.model_preference,
            priority=1,
            metadata={"feature_type": "image_to_world"},
            user_id=user_id,
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Image-to-world job queued successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling image-to-world job: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


class MetricCalibration(BaseModel):
    """1:1 meter scale for physical-replica metaverse anchoring."""

    mode: str = Field(
        "reference_length",
        description="reference_length | two_points | player_height | auto_bbox",
    )
    true_meters: Optional[float] = Field(
        None,
        description="Real-world length in meters (door width, wall span, player height)",
    )
    recon_length: Optional[float] = Field(
        None,
        description="Same length measured in reconstruction units (optional if two_points)",
    )
    recon_height: Optional[float] = Field(
        None, description="For player_height mode: height in recon units"
    )
    player_height_meters: Optional[float] = Field(
        1.6, description="Real player height (default 1.6 m)"
    )
    point_a: Optional[List[float]] = Field(None, description="3D point A in recon space")
    point_b: Optional[List[float]] = Field(None, description="3D point B in recon space")


class EnvironmentScanRequest(BaseModel):
    """Galaxy XR / walk-video environment scan → metric world package (LingBot-Map)."""

    video_path: Optional[str] = None
    video_file_id: Optional[str] = None
    frame_dir: Optional[str] = None
    image_file_ids: Optional[List[str]] = Field(
        None, description="Ordered walk frames (≥3) if not using video"
    )
    world_id: Optional[str] = None
    world_name: Optional[str] = None
    model_preference: str = Field(
        "lingbot_map_environment_scan",
        description="Environment scan model (default LingBot-Map)",
    )
    metric_calibration: Optional[MetricCalibration] = Field(
        None,
        description=(
            "Required for 1:1 physical replica. Prefer measuring a door/wall: "
            "mode=reference_length with true_meters + recon_length, or two_points."
        ),
    )
    # Up to 600; long walks use CPU-resident + windowed LingBot inference (GB10-safe).
    max_frames: int = Field(600, ge=3, le=600)
    frame_stride: int = Field(1, ge=1, le=30)
    refine_to_3dgs: bool = Field(
        False,
        description=(
            "Phase A: convert colored point cloud → Spark-compatible isotropic "
            "Gaussian PLY and export COLMAP (gs_dataset/) for Phase B gsplat train. "
            "Point cloud kept as environment.points.ply."
        ),
    )
    train_3dgs: bool = Field(
        False,
        description=(
            "Phase B: after Phase A, train gsplat on gs_dataset/ and replace "
            "environment.ply with optimized Gaussians (implies refine_to_3dgs). "
            "Long-running on GB10 — prefer POST /train-3dgs on an existing world."
        ),
    )
    train_3dgs_steps: int = Field(
        7000,
        ge=100,
        le=50000,
        description="Phase B training steps when train_3dgs is true",
    )

    model_config = ConfigDict(protected_namespaces=("settings_",))


class Train3dgsRequest(BaseModel):
    """Phase B only — train gsplat on an existing env-scan world with gs_dataset/."""

    world_id: str = Field(..., description="Existing env-scan job id under outputs/worlds/")
    max_steps: int = Field(7000, ge=100, le=50000)
    data_factor: int = Field(4, ge=1, le=8)
    max_images: Optional[int] = Field(
        None, description="Optional image cap (even stride) for faster/smoke trains"
    )
    model_preference: str = Field("env_scan_gsplat_train")

    model_config = ConfigDict(protected_namespaces=("settings_",))


@router.post("/environment-scan", response_model=MeshGenerationResponse)
async def environment_scan(
    request: EnvironmentScanRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user=Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Walk-scan a physical space (Galaxy XR outward cameras / phone video) into a
    World Package with optional 1:1 metric scale for metaverse anchoring.

    Does not change the default image-to-world (TripoSplat) path.
    Requires LingBot-Map installed: ``bash scripts/install_lingbot_map.sh``.
    """
    try:
        user_id = current_user.user_id if current_user else None
        validate_model_preference(
            request.model_preference, "environment_scan", scheduler
        )

        video_path = None
        image_paths: List[str] = []

        if request.video_file_id or request.video_path:
            video_path = await process_file_input(
                file_path=request.video_path,
                file_id=request.video_file_id,
                input_type="video",
                file_store=file_store,
            )
        elif request.image_file_ids:
            for fid in request.image_file_ids:
                image_paths.append(
                    await process_file_input(
                        file_id=fid, input_type="image", file_store=file_store
                    )
                )
        elif request.frame_dir:
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide video_file_id / video_path, frame_dir, or image_file_ids (≥3)",
            )

        inputs: Dict[str, Any] = {
            "world_id": request.world_id,
            "world_name": request.world_name,
            "max_frames": request.max_frames,
            "frame_stride": request.frame_stride,
            "refine_to_3dgs": request.refine_to_3dgs,
            "train_3dgs": request.train_3dgs,
            "train_3dgs_steps": request.train_3dgs_steps,
            "metric_calibration": (
                request.metric_calibration.model_dump(exclude_none=True)
                if request.metric_calibration
                else None
            ),
        }
        if video_path:
            inputs["video_path"] = video_path
        if request.frame_dir:
            inputs["frame_dir"] = request.frame_dir
        if image_paths:
            inputs["image_path"] = image_paths[0]
            inputs["image_paths"] = image_paths

        job_request = JobRequest(
            feature="environment_scan",
            inputs=inputs,
            model_preference=request.model_preference,
            priority=1,
            metadata={"feature_type": "environment_scan"},
            user_id=user_id,
        )
        job_id = await scheduler.schedule_job(job_request)
        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Environment scan job queued (LingBot-Map + metric scale)",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling environment-scan job: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/train-3dgs", response_model=MeshGenerationResponse)
async def train_env_scan_3dgs(
    request: Train3dgsRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user=Depends(get_current_user_or_none),
):
    """
    Phase B gsplat train on an existing environment-scan world package.

    Requires ``outputs/worlds/<world_id>/gs_dataset/`` from Phase A (``refine_to_3dgs``).
    """
    try:
        user_id = current_user.user_id if current_user else None
        world_root = Path("outputs") / "worlds" / request.world_id
        if not world_root.is_dir():
            raise HTTPException(
                status_code=404, detail=f"World not found: {request.world_id}"
            )
        if not (world_root / "gs_dataset").is_dir():
            raise HTTPException(
                status_code=400,
                detail="Missing gs_dataset/ — run Phase A refine_to_3dgs first",
            )

        validate_model_preference(
            request.model_preference, "environment_scan", scheduler
        )
        inputs: Dict[str, Any] = {
            "world_id": request.world_id,
            "world_directory": str(world_root),
            "max_steps": request.max_steps,
            "data_factor": request.data_factor,
        }
        if request.max_images is not None:
            inputs["max_images"] = request.max_images

        job_request = JobRequest(
            feature="environment_scan",
            inputs=inputs,
            model_preference=request.model_preference,
            priority=1,
            metadata={"feature_type": "environment_scan", "phase": "B_gsplat"},
            user_id=user_id,
        )
        job_id = await scheduler.schedule_job(job_request)
        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message=f"Phase B gsplat train queued for world {request.world_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling train-3dgs job: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
