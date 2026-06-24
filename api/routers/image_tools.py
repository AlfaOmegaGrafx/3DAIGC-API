"""Image preprocessing tools (background removal preview for OpenNexus3DStudio)."""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from core.utils.birefnet_removal import remove_background_from_bytes
from core.utils.file_utils import SUPPORTED_IMAGE_FORMATS, validate_file_extension

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/image-tools", tags=["image_tools"])


@router.post("/remove-background")
async def remove_background(
    file: UploadFile = File(..., description="Source image (JPEG/PNG/WebP)"),
    erode_radius: int = Form(1, ge=0, le=8, description="Shrink matte edge (pixels)"),
):
    """
    Remove image background with BiRefNet (same family as TRELLIS / TripoSplat preprocess).
    Returns PNG with alpha for studio preview before image-to-3D jobs.
    """
    filename = file.filename or "image.png"
    if not validate_file_extension(filename, SUPPORTED_IMAGE_FORMATS):
        raise HTTPException(status_code=400, detail="Unsupported image format")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        png_bytes = remove_background_from_bytes(raw, erode_radius=erode_radius)
    except FileNotFoundError as exc:
        logger.warning("Background removal unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Background removal model is not available on this server.",
        ) from exc
    except Exception as exc:
        logger.exception("Background removal failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{stem}_nobg.png"'},
    )
