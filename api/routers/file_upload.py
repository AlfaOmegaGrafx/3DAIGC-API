"""
File upload API endpoints.

Provides endpoints for uploading images and meshes with unique identifiers
that can be used in other API endpoints.

Supports two deployment modes:
- Single-worker mode: Uses in-memory storage (fast, but not shared across workers)
- Multi-worker mode: Uses Redis-backed FileStore (shared across all workers)
"""

import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from api.dependencies import get_file_store
from core.file_store import FileStore
from core.utils.file_utils import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_MESH_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    FileUploadError,
    save_upload_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/file-upload", tags=["file_upload"])

# Configuration
UPLOAD_BASE_DIR = Path("uploads")
UPLOAD_BASE_DIR.mkdir(exist_ok=True)

# In-memory file metadata storage (fallback for single-worker mode)
# In multi-worker mode, FileStore (Redis) is used instead
_local_file_metadata: Dict[str, Dict] = {}


class FileUploadResponse(BaseModel):
    """Response for file upload requests"""

    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Type of file (image/mesh)")
    file_size_mb: float = Field(..., description="File size in MB")
    upload_time: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiration time")


class FileMetadataResponse(BaseModel):
    """Response for file metadata requests"""

    file_id: str = Field(..., description="Unique identifier for the file")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Type of file (image/mesh)")
    file_size_mb: float = Field(..., description="File size in MB")
    upload_time: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiration time")
    is_available: bool = Field(..., description="Whether the file is still available")


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate file type based on extension"""
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_types


def generate_file_id() -> str:
    """Generate unique file identifier"""
    return str(uuid.uuid4())


# ============================================================================
# Storage Layer Abstraction
# ============================================================================

async def store_file_metadata_impl(
    file_store: Optional[FileStore],
    file_id: str,
    file_info: Dict,
) -> None:
    """
    Store file metadata using Redis (multi-worker) or in-memory (single-worker).
    """
    metadata = {
        "file_id": file_id,
        "filename": file_info["original_filename"],
        "file_path": file_info["file_path"],
        "file_type": file_info["file_type"],
        "file_size_mb": file_info["file_size_mb"],
        "upload_time": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
        "is_available": True,
    }
    
    if file_store is not None:
        # Multi-worker mode: use Redis
        await file_store.store_file_metadata(file_id, metadata)
    else:
        # Single-worker mode: use in-memory dict
        _local_file_metadata[file_id] = metadata


async def get_file_metadata_impl(
    file_store: Optional[FileStore],
    file_id: str,
) -> Optional[Dict]:
    """
    Get file metadata from Redis (multi-worker) or in-memory (single-worker).
    """
    if file_store is not None:
        return await file_store.get_file_metadata(file_id)
    else:
        return _local_file_metadata.get(file_id)


async def delete_file_metadata_impl(
    file_store: Optional[FileStore],
    file_id: str,
) -> bool:
    """
    Delete file metadata from Redis (multi-worker) or in-memory (single-worker).
    """
    if file_store is not None:
        return await file_store.delete_file_metadata(file_id)
    else:
        if file_id in _local_file_metadata:
            _local_file_metadata[file_id]["is_available"] = False
            return True
        return False


async def list_file_metadata_impl(
    file_store: Optional[FileStore],
    file_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """
    List file metadata from Redis (multi-worker) or in-memory (single-worker).
    """
    if file_store is not None:
        return await file_store.list_file_metadata(file_type=file_type, limit=limit)
    else:
        files = []
        for metadata in _local_file_metadata.values():
            if file_type and metadata.get("file_type") != file_type:
                continue
            if len(files) >= limit:
                break
            files.append(metadata)
        return files


async def count_files_impl(
    file_store: Optional[FileStore],
    file_type: Optional[str] = None,
) -> int:
    """
    Count files from Redis (multi-worker) or in-memory (single-worker).
    """
    if file_store is not None:
        return await file_store.count_files(file_type=file_type)
    else:
        if file_type:
            return sum(
                1 for m in _local_file_metadata.values() 
                if m.get("file_type") == file_type
            )
        return len(_local_file_metadata)


async def get_file_path_impl(
    file_store: Optional[FileStore],
    file_id: str,
) -> Optional[str]:
    """
    Get file path by ID, checking if file exists on disk.
    """
    metadata = await get_file_metadata_impl(file_store, file_id)
    if metadata and metadata.get("is_available", True):
        file_path = metadata.get("file_path")
        if file_path and os.path.exists(file_path):
            return file_path
    return None


# ============================================================================
# File Upload Logic
# ============================================================================

async def upload_file_with_validation(
    file: UploadFile,
    file_type: str,
    allowed_extensions: List[str],
    file_store: Optional[FileStore],
    max_size_mb: int = 100,
) -> Dict:
    """Upload file with validation and return metadata"""

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not validate_file_type(file.filename, allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed extensions: {allowed_extensions}",
        )

    # Generate file ID and create directory
    file_id = generate_file_id()
    upload_dir = (
        UPLOAD_BASE_DIR / file_type / file_id[:2]
    )  # Use first 2 chars for subdirectory
    upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save file
        file_info = await save_upload_file(
            file, str(upload_dir), max_size_mb=max_size_mb, validate_content=True
        )

        # Store metadata (uses Redis in multi-worker mode, in-memory otherwise)
        await store_file_metadata_impl(file_store, file_id, file_info)

        logger.info(f"Uploaded {file_type} file {file_id}: {file.filename}")

        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_type,
            "file_size_mb": file_info["file_size_mb"],
            "upload_time": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24),
        }
    except FileUploadError as e:
        logger.error(f"Failed to upload {file_type} file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Failed to upload {file_type} file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/image", response_model=FileUploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="Image file to upload"),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Upload an image file.

    Returns a unique file ID that can be used in other API endpoints.
    Supported formats: PNG, JPG, JPEG, WebP, BMP, TIFF
    """
    return await upload_file_with_validation(
        file, "image", SUPPORTED_IMAGE_FORMATS, file_store, max_size_mb=50
    )


@router.post("/mesh", response_model=FileUploadResponse)
async def upload_mesh(
    file: UploadFile = File(..., description="Mesh file to upload"),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Upload a mesh file.

    Returns a unique file ID that can be used in other API endpoints.
    Supported formats: GLB, OBJ, FBX, PLY, STL, GLTF
    """
    return await upload_file_with_validation(
        file, "mesh", SUPPORTED_MESH_FORMATS, file_store, max_size_mb=200
    )


@router.post("/video", response_model=FileUploadResponse)
async def upload_video(
    file: UploadFile = File(..., description="Walk-scan video (Galaxy XR / phone)"),
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Upload a walk-around video for environment scan (LingBot-Map).

    Supported formats: MP4, WebM, MOV, MKV, AVI, M4V. Max 2 GB.
    """
    return await upload_file_with_validation(
        file, "video", SUPPORTED_VIDEO_FORMATS, file_store, max_size_mb=2048
    )


@router.get("/metadata/{file_id}", response_model=FileMetadataResponse)
async def get_file_metadata_endpoint(
    file_id: str,
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Get metadata for an uploaded file.

    Args:
        file_id: Unique file identifier

    Returns:
        File metadata including availability status
    """
    metadata = await get_file_metadata_impl(file_store, file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    # Check if file still exists on disk
    file_path = metadata.get("file_path")
    is_available = metadata.get("is_available", True) and (
        file_path and os.path.exists(file_path)
    )

    # Parse datetime strings if they're strings (from Redis)
    upload_time = metadata.get("upload_time")
    expires_at = metadata.get("expires_at")
    
    if isinstance(upload_time, str):
        upload_time = datetime.fromisoformat(upload_time)
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)

    return FileMetadataResponse(
        file_id=metadata["file_id"],
        filename=metadata["filename"],
        file_type=metadata["file_type"],
        file_size_mb=metadata["file_size_mb"],
        upload_time=upload_time,
        expires_at=expires_at,
        is_available=is_available,
    )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    Delete an uploaded file.

    Args:
        file_id: Unique file identifier

    Returns:
        Deletion confirmation
    """
    metadata = await get_file_metadata_impl(file_store, file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Remove file from disk
        file_path = metadata.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Mark as unavailable / delete metadata
        await delete_file_metadata_impl(file_store, file_id)

        logger.info(f"Deleted file {file_id}: {metadata.get('filename')}")

        return {
            "file_id": file_id,
            "message": "File deleted successfully",
            "filename": metadata.get("filename"),
        }

    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.get("/list")
async def list_uploaded_files(
    file_type: Optional[str] = None,
    limit: int = 100,
    file_store: Optional[FileStore] = Depends(get_file_store),
):
    """
    List uploaded files.

    Args:
        file_type: Optional filter by file type (image/mesh)
        limit: Maximum number of files to return

    Returns:
        List of uploaded files
    """
    files_metadata = await list_file_metadata_impl(file_store, file_type, limit)
    total_count = await count_files_impl(file_store, file_type)
    
    files = []
    for metadata in files_metadata:
        file_path = metadata.get("file_path")
        is_available = metadata.get("is_available", True) and (
            file_path and os.path.exists(file_path)
        )
        
        # Parse datetime strings if they're strings (from Redis)
        upload_time = metadata.get("upload_time")
        if isinstance(upload_time, str):
            upload_time = datetime.fromisoformat(upload_time)
        
        files.append({
            "file_id": metadata.get("file_id"),
            "filename": metadata.get("filename"),
            "file_type": metadata.get("file_type"),
            "file_size_mb": metadata.get("file_size_mb"),
            "upload_time": upload_time,
            "is_available": is_available,
        })

    return {"files": files, "count": len(files), "total_files": total_count}


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported file formats for upload.

    Returns:
        Dictionary of supported formats and limits
    """
    return {
        "image": {
            "formats": SUPPORTED_IMAGE_FORMATS,
            "max_size_mb": 50,
            "description": "Supported image formats for upload",
        },
        "mesh": {
            "formats": SUPPORTED_MESH_FORMATS,
            "max_size_mb": 200,
            "description": "Supported mesh formats for upload",
        },
        "retention": {
            "default_hours": 24,
            "description": "Files are automatically deleted after 24 hours",
        },
    }


# ============================================================================
# Utility Functions (for use by other modules)
# ============================================================================

async def resolve_file_id_async(
    file_id: str,
    file_store: Optional[FileStore] = None,
) -> Optional[str]:
    """
    Resolve a file ID to its actual file path (async version).

    This function can be imported and used by other modules
    to convert file IDs to file paths.
    
    Args:
        file_id: The unique file identifier
        file_store: Optional FileStore instance (for multi-worker mode)
        
    Returns:
        The file path if found and available, None otherwise
    """
    return await get_file_path_impl(file_store, file_id)


def resolve_file_id(file_id: str) -> Optional[str]:
    """
    Resolve a file ID to its actual file path (sync version, single-worker only).

    This function uses the in-memory storage and should only be used
    in single-worker mode. For multi-worker mode, use resolve_file_id_async.
    
    Args:
        file_id: The unique file identifier
        
    Returns:
        The file path if found and available, None otherwise
    """
    metadata = _local_file_metadata.get(file_id)
    if metadata and metadata.get("is_available", True):
        file_path = metadata.get("file_path")
        if file_path and os.path.exists(file_path):
            return file_path
    return None
