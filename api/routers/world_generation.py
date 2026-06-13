"""
World generation API — image → explorable splat environment + optional mesh props.
"""

import logging
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
