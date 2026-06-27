"""
Mesh editing API endpoints.

Provides endpoints for local 3D mesh editing using VoxHammer,
supporting both text-guided and image-guided editing workflows.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.dependencies import get_current_user_or_none, get_file_store, get_scheduler
from api.object_name import ObjectNamed, enrich_job_inputs, enrich_job_metadata
from api.routers.file_upload import resolve_file_id_async
from core.file_store import FileStore
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler
from core.utils.mask_generator import MaskGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mesh-editing", tags=["mesh_editing"])


# Mask parameter models
class MaskBoundingBox(BaseModel):
    """Bounding box mask parameters"""
    center: List[float] = Field(..., description="Center point [x, y, z]", min_items=3, max_items=3)
    dimensions: List[float] = Field(..., description="Box dimensions [width, height, depth]", min_items=3, max_items=3)


class MaskEllipsoid(BaseModel):
    """Ellipsoid mask parameters"""
    center: List[float] = Field(..., description="Center point [x, y, z]", min_items=3, max_items=3)
    radii: List[float] = Field(..., description="Ellipsoid radii [rx, ry, rz]", min_items=3, max_items=3)


# Request models
class TextMeshEditingRequest(ObjectNamed):
    """Request for text-guided mesh editing"""
    
    mesh_path: Optional[str] = Field(None, description="Path to input mesh file (for local files)")
    mesh_base64: Optional[str] = Field(None, description="Base64 encoded mesh data")
    mesh_file_id: Optional[str] = Field(None, description="File ID from upload endpoint")
    
    # Mask parameters - one of these must be provided
    mask_bbox: Optional[MaskBoundingBox] = Field(None, description="Bounding box mask parameters")
    mask_ellipsoid: Optional[MaskEllipsoid] = Field(None, description="Ellipsoid mask parameters")
    
    source_prompt: str = Field(..., description="Text describing the original mesh region")
    target_prompt: str = Field(..., description="Text describing the desired edited mesh region")
    
    num_views: int = Field(150, description="Number of rendering views", ge=50, le=300)
    resolution: int = Field(512, description="Rendering resolution", ge=256, le=1024)
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field("voxhammer_text_mesh_editing", description="Model name for editing")
    model_parameters: Optional[dict] = Field(
        None, 
        description="Model-specific parameters (query /system/models/{model_id}/parameters for schema)"
    )
    
    model_config = ConfigDict(protected_namespaces=("settings_",))
    
    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed_formats = ["glb"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v
    
    @field_validator("mask_ellipsoid")
    @classmethod
    def validate_mask(cls, v, info):
        mesh_path = info.data.get("mesh_path")
        mesh_base64 = info.data.get("mesh_base64")
        mesh_file_id = info.data.get("mesh_file_id")
        mask_bbox = info.data.get("mask_bbox")
        
        # Validate mesh input
        mesh_inputs = sum(bool(x) for x in [mesh_path, mesh_base64, mesh_file_id])
        if mesh_inputs == 0:
            raise ValueError("One of mesh_path, mesh_base64, or mesh_file_id must be provided")
        if mesh_inputs > 1:
            raise ValueError("Only one of mesh_path, mesh_base64, or mesh_file_id should be provided")
        
        # Validate mask input
        mask_inputs = sum(bool(x) for x in [mask_bbox, v])
        if mask_inputs == 0:
            raise ValueError("One of mask_bbox or mask_ellipsoid must be provided")
        if mask_inputs > 1:
            raise ValueError("Only one of mask_bbox or mask_ellipsoid should be provided")
        
        return v


class ImageMeshEditingRequest(ObjectNamed):
    """Request for image-guided mesh editing"""
    
    mesh_path: Optional[str] = Field(None, description="Path to input mesh file")
    mesh_base64: Optional[str] = Field(None, description="Base64 encoded mesh data")
    mesh_file_id: Optional[str] = Field(None, description="File ID from upload endpoint")
    
    source_image_path: Optional[str] = Field(None, description="Path to source reference image")
    source_image_base64: Optional[str] = Field(None, description="Base64 encoded source image")
    source_image_file_id: Optional[str] = Field(None, description="Source image file ID")
    
    target_image_path: Optional[str] = Field(None, description="Path to target reference image")
    target_image_base64: Optional[str] = Field(None, description="Base64 encoded target image")
    target_image_file_id: Optional[str] = Field(None, description="Target image file ID")
    
    mask_image_path: Optional[str] = Field(None, description="Path to 2D mask image")
    mask_image_base64: Optional[str] = Field(None, description="Base64 encoded mask image")
    mask_image_file_id: Optional[str] = Field(None, description="Mask image file ID")
    
    # Mask parameters
    mask_bbox: Optional[MaskBoundingBox] = Field(None, description="Bounding box mask parameters")
    mask_ellipsoid: Optional[MaskEllipsoid] = Field(None, description="Ellipsoid mask parameters")
    
    num_views: int = Field(150, description="Number of rendering views", ge=50, le=300)
    resolution: int = Field(512, description="Rendering resolution", ge=256, le=1024)
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field("voxhammer_image_mesh_editing", description="Model name for editing")
    model_parameters: Optional[dict] = Field(
        None, 
        description="Model-specific parameters (query /system/models/{model_id}/parameters for schema)"
    )
    
    model_config = ConfigDict(protected_namespaces=("settings_",))
    
    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed_formats = ["glb"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v
    
    @field_validator("mask_ellipsoid")
    @classmethod
    def validate_inputs(cls, v, info):
        # Validate all inputs
        mesh_path = info.data.get("mesh_path")
        mesh_base64 = info.data.get("mesh_base64")
        mesh_file_id = info.data.get("mesh_file_id")
        
        source_image_path = info.data.get("source_image_path")
        source_image_base64 = info.data.get("source_image_base64")
        source_image_file_id = info.data.get("source_image_file_id")
        
        target_image_path = info.data.get("target_image_path")
        target_image_base64 = info.data.get("target_image_base64")
        target_image_file_id = info.data.get("target_image_file_id")
        
        mask_image_path = info.data.get("mask_image_path")
        mask_image_base64 = info.data.get("mask_image_base64")
        mask_image_file_id = info.data.get("mask_image_file_id")
        
        mask_bbox = info.data.get("mask_bbox")
        
        # Validate mesh input
        mesh_inputs = sum(bool(x) for x in [mesh_path, mesh_base64, mesh_file_id])
        if mesh_inputs != 1:
            raise ValueError("Exactly one mesh input must be provided")
        
        # Validate image inputs
        source_inputs = sum(bool(x) for x in [source_image_path, source_image_base64, source_image_file_id])
        target_inputs = sum(bool(x) for x in [target_image_path, target_image_base64, target_image_file_id])
        mask_inputs_img = sum(bool(x) for x in [mask_image_path, mask_image_base64, mask_image_file_id])
        
        if source_inputs != 1:
            raise ValueError("Exactly one source image input must be provided")
        if target_inputs != 1:
            raise ValueError("Exactly one target image input must be provided")
        if mask_inputs_img != 1:
            raise ValueError("Exactly one mask image input must be provided")
        
        # Validate 3D mask input
        mask_inputs_3d = sum(bool(x) for x in [mask_bbox, v])
        if mask_inputs_3d != 1:
            raise ValueError("Exactly one 3D mask (bbox or ellipsoid) must be provided")
        
        return v


# Response model
class MeshEditingResponse(BaseModel):
    """Response for mesh editing requests"""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


# Helper function to process file inputs (reused from mesh_generation)
async def process_file_input(
    file_path: Optional[str] = None,
    base64_data: Optional[str] = None,
    file_id: Optional[str] = None,
    input_type: str = "file",
    file_store: Optional[FileStore] = None,
) -> str:
    """Process various file input formats and return the processed file path"""
    from api.routers.mesh_generation import process_file_input as _process
    return await _process(file_path, base64_data, file_id, upload_file=None, input_type=input_type, file_store=file_store)


# Endpoints
@router.post("/text-mesh-editing", response_model=MeshEditingResponse)
async def text_mesh_editing(
    request: TextMeshEditingRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user = Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Edit a 3D mesh using text guidance.
    
    Args:
        request: Text-guided mesh editing parameters
        scheduler: Model scheduler dependency
        current_user: Current authenticated user
        file_store: File store dependency
    
    Returns:
        Job information for the mesh editing task
    """
    try:
        user_id = current_user.user_id if current_user else None
        
        # Process mesh input
        mesh_file_path = await process_file_input(
            file_path=request.mesh_path,
            base64_data=request.mesh_base64,
            file_id=request.mesh_file_id,
            input_type="mesh",
            file_store=file_store,
        )
        
        # Determine mask type and parameters
        if request.mask_bbox:
            mask_type = "bbox"
            mask_center = request.mask_bbox.center
            mask_params = request.mask_bbox.dimensions
        else:  # mask_ellipsoid
            mask_type = "ellipsoid"
            mask_center = request.mask_ellipsoid.center
            mask_params = request.mask_ellipsoid.radii
        
        # Validate mask parameters
        is_valid, error_msg = MaskGenerator.validate_mask_params(
            mask_type, mask_center, mask_params
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid mask parameters: {error_msg}")
        
        job_request = JobRequest(
            feature="text_mesh_editing",
            inputs=enrich_job_inputs(
                {
                    "mesh_path": mesh_file_path,
                    "mask_type": mask_type,
                    "mask_center": mask_center,
                    "mask_params": mask_params,
                    "source_prompt": request.source_prompt,
                    "target_prompt": request.target_prompt,
                    "num_views": request.num_views,
                    "resolution": request.resolution,
                    "output_format": request.output_format,
                    **(request.model_parameters or {}),
                },
                request.object_name,
            ),
            model_preference=request.model_preference,
            priority=1,
            metadata=enrich_job_metadata("text_mesh_editing", request.object_name),
            user_id=user_id,
        )
        
        job_id = await scheduler.schedule_job(job_request)
        
        return MeshEditingResponse(
            job_id=job_id,
            status="queued",
            message="Text-guided mesh editing job queued successfully",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling text mesh editing job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/image-mesh-editing", response_model=MeshEditingResponse)
async def image_mesh_editing(
    request: ImageMeshEditingRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user = Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Edit a 3D mesh using image guidance.
    
    Args:
        request: Image-guided mesh editing parameters
        scheduler: Model scheduler dependency
        current_user: Current authenticated user
        file_store: File store dependency
    
    Returns:
        Job information for the mesh editing task
    """
    try:
        user_id = current_user.user_id if current_user else None
        
        # Process mesh input
        mesh_file_path = await process_file_input(
            file_path=request.mesh_path,
            base64_data=request.mesh_base64,
            file_id=request.mesh_file_id,
            input_type="mesh",
            file_store=file_store,
        )
        
        # Process image inputs
        source_image_path = await process_file_input(
            file_path=request.source_image_path,
            base64_data=request.source_image_base64,
            file_id=request.source_image_file_id,
            input_type="source_image",
            file_store=file_store,
        )
        
        target_image_path = await process_file_input(
            file_path=request.target_image_path,
            base64_data=request.target_image_base64,
            file_id=request.target_image_file_id,
            input_type="target_image",
            file_store=file_store,
        )
        
        mask_image_path = await process_file_input(
            file_path=request.mask_image_path,
            base64_data=request.mask_image_base64,
            file_id=request.mask_image_file_id,
            input_type="mask_image",
            file_store=file_store,
        )
        
        # Determine mask type and parameters
        if request.mask_bbox:
            mask_type = "bbox"
            mask_center = request.mask_bbox.center
            mask_params = request.mask_bbox.dimensions
        else:  # mask_ellipsoid
            mask_type = "ellipsoid"
            mask_center = request.mask_ellipsoid.center
            mask_params = request.mask_ellipsoid.radii
        
        # Validate mask parameters
        is_valid, error_msg = MaskGenerator.validate_mask_params(
            mask_type, mask_center, mask_params
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid mask parameters: {error_msg}")
        
        job_request = JobRequest(
            feature="image_mesh_editing",
            inputs=enrich_job_inputs(
                {
                    "mesh_path": mesh_file_path,
                    "mask_type": mask_type,
                    "mask_center": mask_center,
                    "mask_params": mask_params,
                    "source_image_path": source_image_path,
                    "target_image_path": target_image_path,
                    "mask_image_path": mask_image_path,
                    "num_views": request.num_views,
                    "resolution": request.resolution,
                    "output_format": request.output_format,
                    **(request.model_parameters or {}),
                },
                request.object_name,
            ),
            model_preference=request.model_preference,
            priority=1,
            metadata=enrich_job_metadata("image_mesh_editing", request.object_name),
            user_id=user_id,
        )
        
        job_id = await scheduler.schedule_job(job_request)
        
        return MeshEditingResponse(
            job_id=job_id,
            status="queued",
            message="Image-guided mesh editing job queued successfully",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling image mesh editing job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


# Utility endpoints
@router.get("/supported-masks")
async def get_supported_masks():
    """Get list of supported mask types and their parameters"""
    return {
        "mask_types": {
            "bbox": {
                "name": "Bounding Box",
                "parameters": {
                    "center": "[x, y, z] - Center point of the box",
                    "dimensions": "[width, height, depth] - Size of the box",
                },
            },
            "ellipsoid": {
                "name": "Ellipsoid",
                "parameters": {
                    "center": "[x, y, z] - Center point of the ellipsoid",
                    "radii": "[rx, ry, rz] - Radii in each dimension",
                },
            },
        },
        "constraints": {
            "center_range": "[-10.0, 10.0] (typical mesh space)",
            "dimension_range": "[0.1, 5.0] (positive values)",
            "radii_range": "[0.1, 5.0] (positive values)",
        },
    }

