"""
Spatial fabric / RP1 publishing endpoints for OMB-compliant GLB assets.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from api.dependencies import get_scheduler
from core.spatial_fabric.msf_publish import publish_glb_to_msf
from core.spatial_fabric.omb_validate import analyze_glb_path, recommend_omb_tier
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

router = APIRouter(prefix="/spatial-fabric", tags=["spatial_fabric"])


class PublishRequest(BaseModel):
    job_id: str = Field(..., description="Completed 3DAIGC job id with mesh output")
    asset_name: Optional[str] = Field(None, description="Published object filename stem")
    use_pbr: bool = Field(True, description="Apply OMB PBR tier modifier")


def _spatial_config() -> Dict[str, str]:
    return {
        "public_base_url": os.environ.get("MSF_PUBLIC_BASE_URL", "").rstrip("/"),
        "fabric_msf_url": os.environ.get("MSF_FABRIC_MSF_URL", "").rstrip("/"),
        "company_id": os.environ.get("RP1_COMPANY_ID", ""),
        "objects_dir": os.environ.get(
            "MSF_OBJECTS_DIR", "/home/sifr/MSF_Map_Svc/dist/web/objects"
        ),
    }


def _resolve_job_mesh_path(job_status: Dict[str, Any]) -> str:
    result = job_status.get("result") or {}
    for key in (
        "output_mesh_path",
        "mesh_path",
        "output_path",
        "file_path",
    ):
        path = result.get(key)
        if path and os.path.isfile(path):
            return path
    raise HTTPException(status_code=404, detail="No mesh output file on completed job")


@router.get("/config", summary="Spatial fabric integration config")
async def get_spatial_fabric_config():
    cfg = _spatial_config()
    return {
        "enabled": bool(cfg["public_base_url"]),
        "public_base_url": cfg["public_base_url"] or None,
        "fabric_msf_url": cfg["fabric_msf_url"] or None,
        "company_id": cfg["company_id"] or None,
        "omb_guidelines": "https://omb.wiki/en/spatial-fabric/model-guidelines",
    }


@router.get("/assets/{job_id}", summary="Resolve public GLB URL for a completed job")
async def get_spatial_asset(
    job_id: str,
    request: Request,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    job_status = await scheduler.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_status.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed (status={job_status.get('status')})",
        )

    cfg = _spatial_config()
    download_url = f"{request.base_url}api/v1/system/jobs/{job_id}/download"

    try:
        mesh_path = _resolve_job_mesh_path(job_status)
        stats = analyze_glb_path(mesh_path)
        tier = recommend_omb_tier(stats)
    except HTTPException:
        stats = None
        tier = None
        mesh_path = None

    return {
        "job_id": job_id,
        "download_url": download_url,
        "mesh_path": mesh_path,
        "stats": stats.to_dict() if stats else None,
        "omb": tier,
        "fabric_msf_url": cfg["fabric_msf_url"] or None,
    }


@router.post("/validate-glb", summary="Validate GLB against OMB tier budgets")
async def validate_glb(
    file: UploadFile = File(...),
    use_pbr: bool = Query(True),
):
    suffix = os.path.splitext(file.filename or "upload.glb")[1] or ".glb"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        stats = analyze_glb_path(tmp_path)
        tier = recommend_omb_tier(stats, use_pbr=use_pbr)
        return {"stats": stats.to_dict(), "omb": tier}
    finally:
        os.unlink(tmp_path)


@router.post("/publish", summary="Publish job GLB to MSF object library")
async def publish_to_spatial_fabric(
    body: PublishRequest,
    request: Request,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    cfg = _spatial_config()
    if not cfg["public_base_url"]:
        raise HTTPException(
            status_code=503,
            detail="MSF_PUBLIC_BASE_URL is not configured on the API server",
        )

    job_status = await scheduler.get_job_status(body.job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_status.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed (status={job_status.get('status')})",
        )

    mesh_path = _resolve_job_mesh_path(job_status)
    stats = analyze_glb_path(mesh_path)
    tier = recommend_omb_tier(stats, use_pbr=body.use_pbr)

    asset_name = body.asset_name or f"job-{body.job_id}"
    published = publish_glb_to_msf(
        mesh_path,
        asset_name=asset_name,
        objects_dir=cfg["objects_dir"],
        public_base_url=cfg["public_base_url"],
    )

    return {
        "job_id": body.job_id,
        "published": published,
        "stats": stats.to_dict(),
        "omb": tier,
        "fabric_msf_url": cfg["fabric_msf_url"] or None,
        "scene_assembler_url": f"{cfg['public_base_url']}/",
        "download_url": f"{request.base_url}api/v1/system/jobs/{body.job_id}/download",
    }

@router.post("/publish-glb", summary="Upload and publish GLB to MSF object library")
async def publish_glb_upload(
    file: UploadFile = File(...),
    asset_name: Optional[str] = Query(None),
    use_pbr: bool = Query(True),
):
    cfg = _spatial_config()
    if not cfg["public_base_url"]:
        raise HTTPException(
            status_code=503,
            detail="MSF_PUBLIC_BASE_URL is not configured on the API server",
        )

    suffix = os.path.splitext(file.filename or "upload.glb")[1] or ".glb"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        stats = analyze_glb_path(tmp_path)
        tier = recommend_omb_tier(stats, use_pbr=use_pbr)
        stem = asset_name or os.path.splitext(file.filename or "viewport-export")[0]
        published = publish_glb_to_msf(
            tmp_path,
            asset_name=stem,
            objects_dir=cfg["objects_dir"],
            public_base_url=cfg["public_base_url"],
        )
    finally:
        os.unlink(tmp_path)

    return {
        "published": published,
        "stats": stats.to_dict(),
        "omb": tier,
        "fabric_msf_url": cfg["fabric_msf_url"] or None,
        "scene_assembler_url": f"{cfg['public_base_url']}/",
    }

