"""
Mesh UV unwrapping API endpoints.

Provides endpoints for generating optimized UV coordinates for 3D meshes.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.dependencies import get_current_user_or_none, get_file_store, get_scheduler
from api.routers.file_upload import resolve_file_id_async
from core.file_store import FileStore
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mesh-uv-unwrapping", tags=["mesh_uv_unwrapping"])


def validate_model_preference(
    model_preference: str, feature: str, scheduler: MultiprocessModelScheduler
) -> None:
    """
    Validate that the model preference is available for the given feature.

    Args:
        model_preference: The preferred model ID
        feature: The feature type for the job
        scheduler: The model scheduler instance

    Raises:
        HTTPException: If the model preference is invalid
    """
    if not scheduler.validate_model_preference(model_preference, feature):
        available_models = scheduler.get_available_models(feature)
        feature_models = available_models.get(feature, [])

        if not feature_models:
            raise HTTPException(
                status_code=400,
                detail=f"No models available for feature '{feature}'. Please check if models are registered.",
            )

        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_preference}' is not available for feature '{feature}'. "
            f"Available models: {feature_models}",
        )


# Request models
class MeshUVUnwrappingRequest(BaseModel):
    """Request for mesh UV unwrapping"""

    mesh_path: Optional[str] = Field(None, description="Path to the input mesh file")
    mesh_file_id: Optional[str] = Field(
        None, description="File ID from upload endpoint"
    )
    distortion_threshold: float = Field(
        1.25, description="Maximum allowed distortion", ge=1.0, le=5.0
    )
    pack_method: str = Field(
        "blender",
        description="UV packing method (blender, uvpackmaster, or none)",
    )
    save_individual_parts: bool = Field(
        True, description="Save individual part meshes"
    )
    save_visuals: bool = Field(False, description="Save visualization images")
    output_format: str = Field("obj", description="Output format for UV unwrapped mesh")
    model_preference: str = Field(
        "xatlas_uv_unwrapping",
        description=(
            "UV model: xatlas_uv_unwrapping (MIT, commercial) or "
            "partuv_uv_unwrapping (non-commercial, disabled by default)"
        ),
    )
    model_parameters: Optional[dict] = Field(
        None, 
        description="Model-specific parameters (query /system/models/{model_id}/parameters for schema)"
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed_formats = ["obj", "glb"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @field_validator("pack_method")
    @classmethod
    def validate_pack_method(cls, v):
        # currently uvpackmaster is not installed, if needed, install it according to the official website 
        allowed_methods = ["blender"]
        if v not in allowed_methods:
            raise ValueError(f"Pack method must be one of: {allowed_methods}")
        return v

    @field_validator("mesh_file_id")
    @classmethod
    def validate_inputs(cls, v, info):
        mesh_path = info.data.get("mesh_path")

        inputs_provided = sum(bool(x) for x in [mesh_path, v])

        if inputs_provided == 0:
            raise ValueError("One of mesh_path or mesh_file_id must be provided")
        if inputs_provided > 1:
            raise ValueError("Only one of mesh_path or mesh_file_id should be provided")
        return v

    model_config = ConfigDict(protected_namespaces=("settings_",))


# Response models
class MeshUVUnwrappingResponse(BaseModel):
    """Response for mesh UV unwrapping requests"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


@router.post("/unwrap-mesh", response_model=MeshUVUnwrappingResponse)
async def unwrap_mesh(
    request: MeshUVUnwrappingRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user = Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Generate UV coordinates for a 3D mesh.

    This endpoint creates optimized UV layouts using part-based unwrapping
    to minimize distortion and create efficient texture maps.

    Args:
        request: Mesh UV unwrapping parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the UV unwrapping task
    """
    try:
        # Validate model preference
        validate_model_preference(
            request.model_preference, "uv_unwrapping", scheduler
        )

        # Process mesh input
        mesh_file_path = None

        if request.mesh_file_id:
            # Handle file ID (uses Redis in multi-worker mode)
            mesh_file_path = await resolve_file_id_async(request.mesh_file_id, file_store)
            if not mesh_file_path:
                raise HTTPException(
                    status_code=404, detail="Mesh file not found or expired"
                )
        else:
            mesh_file_path = request.mesh_path

        # Validate mesh file exists
        if not mesh_file_path:
            raise HTTPException(
                status_code=400, detail="Mesh path or file ID must be provided"
            )

        if not Path(mesh_file_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Mesh file not found: {mesh_file_path}"
            )

        # Create job request
        user_id = current_user.user_id if current_user else None
        job_request = JobRequest(
            feature="uv_unwrapping",
            inputs={
                "mesh_path": mesh_file_path,
                "distortion_threshold": request.distortion_threshold,
                "pack_method": request.pack_method,
                "save_individual_parts": request.save_individual_parts,
                "save_visuals": request.save_visuals,
                "output_format": request.output_format,
                **(request.model_parameters or {}),
            },
            model_preference=request.model_preference,
            user_id=user_id,
            priority=1,
            metadata={"feature_type": "uv_unwrapping"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshUVUnwrappingResponse(
            job_id=job_id,
            status="queued",
            message="Mesh UV unwrapping job queued successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling mesh UV unwrapping job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to schedule job: {str(e)}"
        )


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported input and output formats for mesh UV unwrapping.

    Returns:
        Dictionary of supported formats
    """
    return {
        "input_formats": ["obj", "glb"],
        "output_formats": ["obj"],  # PartUV primarily outputs OBJ with UV
    }


@router.get("/pack-methods")
async def get_pack_methods():
    """
    Get available UV packing methods.

    Returns:
        Dictionary of packing methods with descriptions
    """
    return {
        "pack_methods": {
            "blender": {
                "description": "Default packing method using bpy",
            },
            # NOT INSTALLED 
            # "uvpackmaster": {
            #     "description": "Professional packing with part grouping support",
            # },
            # "none": {
            #     "description": "No packing - outputs unwrapped UV charts without arrangement",
            # },
        }
    }


@router.get("/available-models")
async def get_available_models(
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Get available UV unwrapping models and their specifications.

    Args:
        scheduler: Model scheduler dependency

    Returns:
        Dictionary of available models with their details
    """
    try:
        available_models = scheduler.get_available_models("uv_unwrapping")
        models_info = available_models.get("uv_unwrapping", [])

        # Add additional model descriptions
        models_details = {}
        for model_id in models_info:
            if "partuv" in model_id.lower():
                models_details[model_id] = {
                    "description": "PartUV - Part-based UV unwrapping with minimal distortion",
                }
            else:
                models_details[model_id] = {
                    "description": "UV unwrapping model",
                }

        return {
            "available_models": models_info,
            "models_details": models_details,
        }

    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get available models: {str(e)}"
        )

