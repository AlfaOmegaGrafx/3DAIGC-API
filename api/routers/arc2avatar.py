"""
Arc2Avatar head splat API.

POST /api/v1/arc2avatar/image-to-head
GET  /api/v1/arc2avatar/status
"""
from __future__ import annotations

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
from adapters.arc2avatar_adapter import get_arc2avatar_status
from core.file_store import FileStore
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arc2avatar", tags=["arc2avatar"])


class ImageToHeadRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Local image path")
    image_base64: Optional[str] = Field(None, description="Base64 image")
    image_file_id: Optional[str] = Field(None, description="Uploaded file id")
    output_format: str = Field("ply", description="Output format (ply)")
    model_preference: str = Field("arc2avatar_head", description="Model id")
    model_parameters: Optional[dict] = Field(
        None,
        description="iterations, batch_size, etc.",
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        if v not in ("ply",):
            raise ValueError("output_format must be ply")
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


@router.get("/status")
async def arc2avatar_status():
    """Install / weights / python readiness (does not start a job)."""
    return get_arc2avatar_status()


@router.post("/image-to-head", response_model=MeshGenerationResponse)
async def image_to_head(
    request: ImageToHeadRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user=Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """Queue Arc2Avatar SDS head splat job (minutes–hours per subject)."""
    try:
        user_id = current_user.user_id if current_user else None
        validate_model_preference(
            request.model_preference, "arc2avatar_head", scheduler
        )

        image_file_path = await process_file_input(
            file_path=request.image_path,
            base64_data=request.image_base64,
            file_id=request.image_file_id,
            input_type="image",
            file_store=file_store,
        )

        inputs = {
            "image_path": image_file_path,
            "output_format": request.output_format,
        }
        if request.model_parameters:
            inputs["model_parameters"] = request.model_parameters
            if "iterations" in request.model_parameters:
                inputs["iterations"] = request.model_parameters["iterations"]
            if "batch_size" in request.model_parameters:
                inputs["batch_size"] = request.model_parameters["batch_size"]

        job_request = JobRequest(
            feature="arc2avatar_head",
            inputs=inputs,
            model_preference=request.model_preference,
            user_id=user_id or "",
            priority=1,
            timeout_seconds=21600,
        )
        job_id = await scheduler.submit_job(job_request)
        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Arc2Avatar head job queued (SDS train — long running)",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Arc2Avatar queue failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
