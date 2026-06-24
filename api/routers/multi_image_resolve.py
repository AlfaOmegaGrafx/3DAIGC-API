"""Resolve uploaded file IDs to local paths for multi-image jobs."""
from __future__ import annotations

from typing import List, Optional

from fastapi import HTTPException

from api.routers.file_upload import resolve_file_id_async
from core.file_store import FileStore
from core.utils.multi_image_input import normalize_reference_image_file_ids


async def resolve_reference_image_paths(
    reference_file_ids: Optional[List[str]],
    file_store: Optional[FileStore],
    *,
    primary_file_id: Optional[str] = None,
) -> List[str]:
    """Resolve reference image file IDs to absolute paths (deduped, capped)."""
    ids = normalize_reference_image_file_ids(
        reference_file_ids, primary_file_id=primary_file_id
    )
    paths: List[str] = []
    for fid in ids:
        resolved = await resolve_file_id_async(fid, file_store)
        if not resolved:
            raise HTTPException(
                status_code=404,
                detail=f"Reference image file not found or expired: {fid}",
            )
        paths.append(resolved)
    return paths


async def collect_multiview_image_paths(
    primary_path: str,
    reference_file_ids: Optional[List[str]],
    file_store: Optional[FileStore],
    *,
    primary_file_id: Optional[str] = None,
) -> List[str]:
    """Primary first, then unique reference paths."""
    refs = await resolve_reference_image_paths(
        reference_file_ids, file_store, primary_file_id=primary_file_id
    )
    out: List[str] = [primary_path]
    seen = {primary_path}
    for p in refs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out
