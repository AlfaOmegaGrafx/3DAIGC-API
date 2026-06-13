"""
Gaussian splat generation API endpoints.

Image → 3D Gaussian splats (.ply / .splat) for Spark.js / WebXR rendering.
"""

import logging
from typing import Optional

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

router = APIRouter(prefix="/splat-generation", tags=["splat_generation"])


class ImageToSplatRequest(BaseModel):
    """Request for image-to-Gaussian-splat generation."""

    image_path: Optional[str] = Field(None, description="Path to input image (local)")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_file_id: Optional[str] = Field(None, description="Uploaded image file ID")
    output_format: str = Field(
        "ply", description="Output splat format (ply or splat)"
    )
    model_preference: str = Field(
        "triposplat_image_to_splat",
        description="Splat generation model",
    )
    model_parameters: Optional[dict] = Field(
        None,
        description="Model-specific parameters (num_gaussians, steps, seed, etc.)",
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed = ["ply", "splat"]
        if v not in allowed:
            raise ValueError(f"Output format must be one of: {allowed}")
        return v

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


@router.post("/image-to-splat", response_model=MeshGenerationResponse)
async def image_to_splat(
    request: ImageToSplatRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user=Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """Generate 3D Gaussian splats from a single image (TripoSplat)."""
    try:
        user_id = current_user.user_id if current_user else None

        validate_model_preference(
            request.model_preference, "image_to_splat", scheduler
        )

        image_file_path = await process_file_input(
            file_path=request.image_path,
            base64_data=request.image_base64,
            file_id=request.image_file_id,
            input_type="image",
            file_store=file_store,
        )

        job_request = JobRequest(
            feature="image_to_splat",
            inputs={
                "image_path": image_file_path,
                "output_format": request.output_format,
                **(request.model_parameters or {}),
            },
            model_preference=request.model_preference,
            priority=1,
            metadata={"feature_type": "image_to_splat"},
            user_id=user_id,
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Image-to-splat generation job queued successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling image-to-splat job: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")
