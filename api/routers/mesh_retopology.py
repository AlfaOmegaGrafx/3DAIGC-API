"""
Mesh retopology API endpoints.

Provides endpoints for optimizing mesh topology through retopology operations.
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

router = APIRouter(prefix="/mesh-retopology", tags=["mesh_retopology"])


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
class MeshRetopologyRequest(BaseModel):
    """Request for mesh retopology"""

    mesh_path: Optional[str] = Field(None, description="Path to the input mesh file")
    mesh_file_id: Optional[str] = Field(
        None, description="File ID from upload endpoint"
    )
    target_vertex_count: Optional[int] = Field(
        None, description="Target number of vertices", ge=100, le=20000
    )
    poly_type: Optional[str] = Field(
        "tri", description="Specification of the polygon type"
    )
    output_format: str = Field("obj", description="Output format for retopologized mesh")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    model_preference: str = Field(
        "instant_meshes_retopology",
        description=(
            "Retopology model: instant_meshes_retopology (commercial, BSD-3), "
            "or fastmesh_v1k/v4k_retopology (non-commercial, disabled by default)"
        ),
    )
    model_parameters: Optional[dict] = Field(
        None, 
        description="Model-specific parameters (query /system/models/{model_id}/parameters for schema)"
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed_formats = ["obj", "glb", "ply"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @field_validator("poly_type")
    @classmethod
    def validate_poly_type(cls, v):
        allowed_poly_types = ["tri", "quad"]
        if v not in allowed_poly_types:
            raise ValueError(f"Polygon type must be one of: {allowed_poly_types}")
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
class MeshRetopologyResponse(BaseModel):
    """Response for mesh retopology requests"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


@router.post("/retopologize-mesh", response_model=MeshRetopologyResponse)
async def retopologize_mesh(
    request: MeshRetopologyRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user = Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Retopologize a 3D mesh to optimize its topology.

    This endpoint optimizes mesh topology by converting high-resolution meshes
    to lower-polygon versions while preserving shape and features.

    Args:
        request: Mesh retopology parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the retopology task
    """
    try:
        # Validate model preference
        validate_model_preference(
            request.model_preference, "mesh_retopology", scheduler
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
            feature="mesh_retopology",
            inputs={
                "mesh_path": mesh_file_path,
                "target_vertex_count": request.target_vertex_count,
                "output_format": request.output_format,
                "seed": request.seed,
                **(request.model_parameters or {}),
            },
            model_preference=request.model_preference,
            priority=1,
            metadata={"feature_type": "mesh_retopology"},
            user_id=user_id,
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshRetopologyResponse(
            job_id=job_id,
            status="queued",
            message="Mesh retopology job queued successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling mesh retopology job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to schedule job: {str(e)}"
        )


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported input and output formats for mesh retopology.

    Returns:
        Dictionary of supported formats
    """
    return {
        "input_formats": ["obj", "glb", "ply", "stl"],
        "output_formats": ["obj", "glb", "ply"],
    }


@router.get("/available-models")
async def get_available_models(
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Get available retopology models and their specifications.

    Args:
        scheduler: Model scheduler dependency

    Returns:
        Dictionary of available models with their details
    """
    try:
        available_models = scheduler.get_available_models("mesh_retopology")
        models_info = available_models.get("mesh_retopology", [])

        # Add additional model descriptions
        models_details = {}
        for model_id in models_info:
            if "v1k" in model_id.lower():
                models_details[model_id] = {
                    "description": "FastMesh V1K - Generates meshes with ~1000 vertices",
                    "target_vertices": 1000,
                }
            elif "v4k" in model_id.lower():
                models_details[model_id] = {
                    "description": "FastMesh V4K - Generates meshes with ~4000 vertices",
                    "target_vertices": 4000,
                }
            else:
                models_details[model_id] = {
                    "description": "Mesh retopology model",
                    "target_vertices": None,
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

