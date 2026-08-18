"""Image generation API — text-to-image (Krea) + image-edit (Mage-Flow-Edit)."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

router = APIRouter(prefix="/image-generation", tags=["image_generation"])


class TextToImageRequest(BaseModel):
    """Text prompt → raster image (PNG/WebP)."""

    text_prompt: str = Field(..., description="Natural language image description")
    width: int = Field(1024, ge=512, le=2048, description="Output width (px)")
    height: int = Field(1024, ge=512, le=2048, description="Output height (px)")
    output_format: str = Field("png", description="png or webp")
    model_preference: str = Field(
        "krea2_turbo_text_to_image",
        description="Text-to-image model (default: Krea 2 Turbo local weights)",
    )
    model_parameters: Optional[dict] = Field(
        None,
        description="Optional: num_inference_steps, guidance_scale, seed",
    )

    model_config = ConfigDict(protected_namespaces=("settings_",))

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed = ["png", "webp"]
        if v not in allowed:
            raise ValueError(f"output_format must be one of: {allowed}")
        return v


@router.post("/text-to-image", response_model=MeshGenerationResponse)
async def text_to_image(
    request: TextToImageRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user=Depends(get_current_user_or_none),
):
    """Generate an image from text using locally hosted Krea 2 weights (no api.krea.ai)."""
    try:
        user_id = current_user.user_id if current_user else None

        validate_model_preference(
            request.model_preference, "text_to_image", scheduler
        )

        params = dict(request.model_parameters or {})
        inputs = {
            "text_prompt": request.text_prompt,
            "width": request.width,
            "height": request.height,
            "output_format": request.output_format,
            "seed": params.get("seed"),
        }
        if params.get("num_inference_steps") is not None:
            inputs["num_inference_steps"] = params["num_inference_steps"]
        if params.get("guidance_scale") is not None:
            inputs["guidance_scale"] = params["guidance_scale"]

        job_request = JobRequest(
            feature="text_to_image",
            inputs=inputs,
            model_preference=request.model_preference,
            priority=1,
            metadata={
                "text_prompt": request.text_prompt[:200],
                "width": request.width,
                "height": request.height,
            },
            user_id=user_id,
        )

        job_id = await scheduler.schedule_job(job_request)
        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Text-to-image job queued (Krea 2 local inference)",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling text-to-image job: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


class ImageEditRequest(BaseModel):
    """Image + edit instruction → edited raster image (PNG/WebP)."""

    text_prompt: str = Field(..., description="Edit instruction (natural language)")
    image_path: Optional[str] = Field(None, description="Path to input image")
    image_base64: Optional[str] = Field(None, description="Base64-encoded input image")
    image_file_id: Optional[str] = Field(None, description="Uploaded file id")
    output_format: str = Field("png", description="png or webp")
    model_preference: str = Field(
        "mage_flow_edit_turbo",
        description="Image-edit model (default: Mage-Flow-Edit-Turbo)",
    )
    model_parameters: Optional[dict] = Field(
        None,
        description="Optional: num_inference_steps, guidance_scale, max_size, seed, width, height",
    )

    model_config = ConfigDict(protected_namespaces=("settings_",))

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed = ["png", "webp"]
        if v not in allowed:
            raise ValueError(f"output_format must be one of: {allowed}")
        return v

    @model_validator(mode="after")
    def validate_image_source(self):
        provided = sum(
            bool(x) for x in [self.image_path, self.image_base64, self.image_file_id]
        )
        if provided == 0:
            raise ValueError(
                "One of image_path, image_base64, or image_file_id must be provided"
            )
        if provided > 1:
            raise ValueError(
                "Only one of image_path, image_base64, or image_file_id should be provided"
            )
        return self


@router.post("/image-edit", response_model=MeshGenerationResponse)
async def image_edit(
    request: ImageEditRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    file_store: Optional[FileStore] = Depends(get_file_store),
    current_user=Depends(get_current_user_or_none),
):
    """Instruction-based image edit via Mage-Flow-Edit-Turbo (isolated venv)."""
    try:
        user_id = current_user.user_id if current_user else None

        validate_model_preference(request.model_preference, "image_edit", scheduler)

        image_file_path = await process_file_input(
            file_path=request.image_path,
            base64_data=request.image_base64,
            file_id=request.image_file_id,
            input_type="image",
            file_store=file_store,
        )

        params = dict(request.model_parameters or {})
        inputs = {
            "image_path": image_file_path,
            "text_prompt": request.text_prompt,
            "output_format": request.output_format,
            "seed": params.get("seed"),
        }
        for key in (
            "num_inference_steps",
            "guidance_scale",
            "max_size",
            "width",
            "height",
        ):
            if params.get(key) is not None:
                inputs[key] = params[key]

        job_request = JobRequest(
            feature="image_edit",
            inputs=inputs,
            model_preference=request.model_preference,
            priority=1,
            metadata={
                "text_prompt": request.text_prompt[:200],
                "image_path": image_file_path,
            },
            user_id=user_id,
        )

        job_id = await scheduler.schedule_job(job_request)
        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Image-edit job queued (Mage-Flow-Edit-Turbo)",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling image-edit job: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
