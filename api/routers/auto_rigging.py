"""
Auto-rigging API endpoints.

Provides endpoints for automatically adding bone structures to 3D meshes.
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.dependencies import get_current_user_or_none, get_file_store, get_scheduler
from api.object_name import ObjectNamed, enrich_job_inputs, enrich_job_metadata
from api.routers.file_upload import resolve_file_id_async
from core.file_store import FileStore
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler
from core.utils.humanoid_template import (
    get_template,
    load_template_manifest,
    template_paths_available,
    validate_humanoid_template,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auto-rigging", tags=["auto_rigging"])


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
class AutoRigRequest(ObjectNamed):
    """Request for auto-rigging"""

    mesh_path: Optional[str] = Field(None, description="Path to the input mesh file")
    mesh_file_id: Optional[str] = Field(
        None, description="File ID from upload endpoint"
    )
    mesh_job_id: Optional[str] = Field(
        None,
        description="Completed mesh job whose output should be rigged (no re-upload)",
    )
    rig_mode: str = Field("skeleton", description="Rig mode for auto-rigging")
    humanoid_template_id: Optional[str] = Field(
        None,
        description="Humanoid VRM template id when rig_mode is 'template' (default: template)",
    )
    creature_template_id: Optional[str] = Field(
        None,
        description="Creature template id when rig_mode is 'creature_template' (default: fox)",
    )
    appearance_slot: Optional[str] = Field(
        None,
        description="Appearance Editor slot when rig_mode is 'appearance_component'",
    )
    output_format: str = Field("fbx", description="Output format for rigged mesh")
    model_preference: str = Field(
        "unirig_auto_rig", description="Name of the auto-rigging model to use"
    )
    model_parameters: Optional[dict] = Field(
        None, 
        description="Model-specific parameters (query /system/models/{model_id}/parameters for schema)"
    )
    likeness_image_file_id: Optional[str] = Field(
        None,
        description=(
            "Optional selfie / face photo file id for template_wrap face_likeness "
            "(MediaPipe mesh when likeness_source is selfie or auto)"
        ),
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed_formats = ["fbx", "glb", "vrm"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @model_validator(mode="after")
    def validate_mesh_source(self):
        provided = sum(
            bool(x) for x in [self.mesh_path, self.mesh_file_id, self.mesh_job_id]
        )
        if provided == 0:
            raise ValueError(
                "One of mesh_path, mesh_file_id, or mesh_job_id must be provided"
            )
        if provided > 1:
            raise ValueError(
                "Only one of mesh_path, mesh_file_id, or mesh_job_id should be provided"
            )
        return self

    model_config = ConfigDict(protected_namespaces=("settings_",))


def reject_non_humanoid_template_wrap(
    rig_mode: str, model_preference: Optional[str]
) -> Optional[str]:
    """
    MeshMonk / template_wrap / Phase 5 head stitch is humanoid UniRig only.

    Returns an error detail string when the combination is invalid, else None.
    Creatures use SkinTokens / creature_template + client creatureFaceRetarget.
    """
    if str(rig_mode or "").lower() != "template_wrap":
        return None
    pref = (model_preference or "").strip()
    if pref == "creature_template_auto_rig":
        return (
            "template_wrap is humanoid-only (UniRig + template.vrm head stitch). "
            "Use rig_mode=creature_template for Mesh2Motion creatures, "
            "or SkinTokens for non-humanoid AIGC; face motion uses "
            "client creatureFaceRetarget, not MeshMonk."
        )
    if pref and pref != "unirig_auto_rig":
        if pref == "skintokens_auto_rig":
            return (
                "template_wrap cannot use SkinTokens. "
                "Use unirig_auto_rig for humanoid head stitch / wrap, "
                "or drop wrap for SkinTokens."
            )
    return None


_MESH_RESULT_KEYS = (
    "output_mesh_path",
    "output_mesh_path_glb",
    "rigged_mesh_path",
    "mesh_path",
    "output_path",
    "file_path",
)


def _looks_like_mesh_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".glb", ".gltf", ".fbx", ".obj", ".vrm"))


async def resolve_mesh_path_from_job(scheduler, job_id: str) -> str:
    """Resolve a completed job's on-disk mesh so generate-rig can skip re-upload."""
    job = await scheduler.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Source mesh job not found")
    status = str(job.get("status") or "").lower()
    if status not in {"completed", "success", "done", "succeeded"}:
        raise HTTPException(
            status_code=400,
            detail=f"Source mesh job is not completed (status={status})",
        )
    result = job.get("result") or {}
    candidates = []
    for key in _MESH_RESULT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for path in candidates:
        if os.path.isfile(path) and _looks_like_mesh_file(path):
            return path
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise HTTPException(
        status_code=404,
        detail="Source mesh job has no readable output mesh on disk",
    )


# Response models
class AutoRigResponse(BaseModel):
    """Response for auto-rigging requests"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


@router.post("/generate-rig", response_model=AutoRigResponse)
async def generate_rig(
    request: AutoRigRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
    current_user = Depends(get_current_user_or_none),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Generate bone structure for a 3D mesh.

    Args:
        request: Auto-rigging parameters
        scheduler: Model scheduler dependency
        current_user: Current authenticated user (required if auth enabled)

    Returns:
        Job information for the auto-rigging task
    """
    user_id = current_user.user_id if current_user else None
    
    allowed_modes = [
        "skeleton",
        "skin",
        "full",
        "template",
        "template_wrap",
        "appearance_component",
        "creature_template",
    ]
    rig_mode_normalized = request.rig_mode.lower()
    if rig_mode_normalized not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid rig mode. Allowed: skeleton, skin, full, template, "
                "template_wrap, appearance_component, creature_template"
            ),
        )

    wrap_reject = reject_non_humanoid_template_wrap(
        rig_mode_normalized, request.model_preference
    )
    if wrap_reject:
        raise HTTPException(status_code=400, detail=wrap_reject)

    try:
        # Validate model preference
        validate_model_preference(request.model_preference, "auto_rig", scheduler)

        # Process mesh input
        mesh_file_path = None

        if request.mesh_file_id:
            # Handle file ID (uses Redis in multi-worker mode)
            mesh_file_path = await resolve_file_id_async(request.mesh_file_id, file_store)
            if not mesh_file_path:
                raise HTTPException(
                    status_code=404, detail="Mesh file not found or expired"
                )
        elif request.mesh_job_id:
            mesh_file_path = await resolve_mesh_path_from_job(
                scheduler, request.mesh_job_id
            )
        else:
            mesh_file_path = request.mesh_path

        # Validate mesh file exists
        if not mesh_file_path:
            raise HTTPException(
                status_code=400, detail="Mesh path or file ID must be provided"
            )

        likeness_image_path = None
        mp = dict(request.model_parameters or {})
        # Do not let model_parameters clobber the resolved selfie path.
        mp.pop("likeness_image_path", None)
        likeness_file_id = (
            request.likeness_image_file_id
            or mp.pop("likeness_image_file_id", None)
        )
        if likeness_file_id:
            likeness_image_path = await resolve_file_id_async(
                str(likeness_file_id), file_store
            )
            if not likeness_image_path:
                raise HTTPException(
                    status_code=404,
                    detail="Likeness selfie file not found or expired",
                )

        # Validate rig type
        job_inputs = enrich_job_inputs(
            {
                "rig_mode": rig_mode_normalized,
                "mesh_path": mesh_file_path,
                "output_format": request.output_format,
                **(
                    {"humanoid_template_id": request.humanoid_template_id}
                    if request.humanoid_template_id
                    else {}
                ),
                **(
                    {"creature_template_id": request.creature_template_id}
                    if request.creature_template_id
                    else {}
                ),
                **(
                    {"appearance_slot": request.appearance_slot}
                    if request.appearance_slot
                    else {}
                ),
                **mp,
                **(
                    {"likeness_image_path": likeness_image_path}
                    if likeness_image_path
                    else {}
                ),
            },
            request.object_name,
        )
        job_request = JobRequest(
            feature="auto_rig",
            inputs=job_inputs,
            model_preference=request.model_preference,
            priority=1,
            metadata=enrich_job_metadata("auto_rig", request.object_name),
            user_id=user_id,
        )

        job_id = await scheduler.schedule_job(job_request)

        return AutoRigResponse(
            job_id=job_id,
            status="queued",
            message="Auto-rigging job queued successfully",
        )
    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error(f"Error scheduling auto-rig job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.get("/humanoid-templates/{template_id}/manifest")
async def get_humanoid_template_manifest(template_id: str):
    """
    Return regression manifest + live VRM analysis for template.vrm.
    Used by OpenNexus3DStudio VRM export and expression planning.
    """
    try:
        spec = get_template(template_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manifest = load_template_manifest(template_id)
    errors = validate_humanoid_template(template_id) if template_paths_available(template_id) else [
        f"Template VRM missing: {spec.vrm_path}"
    ]

    return {
        "template_id": template_id,
        "vrm_path": str(spec.vrm_path),
        "skeleton_fbx_path": str(spec.skeleton_fbx_path),
        "available": template_paths_available(template_id),
        "validation_errors": errors,
        "expected": manifest.get("expected", {}),
        "description": manifest.get(
            "description",
            "Master humanoid VRM (template.vrm) with facial blend shapes",
        ),
        "blend_shapes_on_generated_mesh": True,
        "wrap_status": "head_stitch",
        "wrap_humanoid_only": True,
        "documentation": "/docs/AVATAR_PIPELINE.md",
        "note": (
            "template_wrap Phase 5 keeps template.vrm head morphs + AIGC body. "
            "MeshMonk likeness (Phase 4) is deferred. Creatures use creatureFaceRetarget."
        ),
    }


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported input and output formats for auto-rigging.

    Returns:
        Dictionary of supported formats
    """
    return {"input_formats": ["obj", "glb", "fbx", "vrm"], "output_formats": ["fbx", "glb", "vrm"]}
