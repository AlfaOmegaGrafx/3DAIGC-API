"""Motion generation API — text-to-motion (Kimodo) for VRM playback in OpenNexus3DStudio."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.dependencies import get_current_user_or_none, get_scheduler
from api.routers.mesh_generation import MeshGenerationResponse, validate_model_preference
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/motion-generation", tags=["motion_generation"])


class TextToMotionRequest(BaseModel):
    """Text prompt → skeletal motion clip (studio_motion.json for uploaded VRM)."""

    text_prompt: str = Field(..., description="Natural language motion description")
    duration: float = Field(5.0, description="Clip length in seconds", ge=1.0, le=30.0)
    output_format: str = Field(
        "studio_motion",
        description="Primary artifact: studio_motion (VRM), npz, or bvh",
    )
    export_bvh: bool = Field(False, description="Also write SOMA BVH alongside studio_motion")
    model_preference: str = Field(
        "kimodo_text_to_motion",
        description="Motion generation model",
    )
    model_parameters: Optional[dict] = Field(
        None,
        description="Optional: diffusion_steps, seed",
    )

    model_config = ConfigDict(protected_namespaces=("settings_",))

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed = ["studio_motion", "npz", "bvh"]
        if v not in allowed:
            raise ValueError(f"output_format must be one of: {allowed}")
        return v


@router.post("/text-to-motion", response_model=MeshGenerationResponse)
async def text_to_motion(
    request: TextToMotionRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user=Depends(get_current_user_or_none),
):
    """Generate a motion clip from text (Kimodo SOMA → studio_motion.json)."""
    try:
        user_id = current_user.user_id if current_user else None

        validate_model_preference(
            request.model_preference, "text_to_motion", scheduler
        )

        params = dict(request.model_parameters or {})
        job_request = JobRequest(
            feature="text_to_motion",
            inputs={
                "text_prompt": request.text_prompt,
                "duration": request.duration,
                "output_format": request.output_format,
                "export_bvh": request.export_bvh,
                "diffusion_steps": params.get("diffusion_steps", 100),
                "seed": params.get("seed"),
            },
            model_preference=request.model_preference,
            priority=1,
            metadata={
                "text_prompt": request.text_prompt[:200],
                "duration": request.duration,
            },
            user_id=user_id,
        )

        job_id = await scheduler.schedule_job(job_request)
        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Text-to-motion job queued (Kimodo)",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling text-to-motion job: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
