"""Shared helpers for multi-image job inputs (Phase 1+ roadmap)."""
from __future__ import annotations

from typing import Iterable, List, Optional

MAX_REFERENCE_IMAGES = 7
MAX_TOTAL_IMAGES = MAX_REFERENCE_IMAGES + 1


def normalize_reference_image_file_ids(
    reference_ids: Optional[Iterable[str]],
    *,
    primary_file_id: Optional[str] = None,
) -> List[str]:
    """
    Dedupe reference file IDs and drop the primary if duplicated.

    Returns at most ``MAX_REFERENCE_IMAGES`` ids.
    """
    primary = (primary_file_id or "").strip()
    seen: set[str] = set()
    out: List[str] = []
    for raw in reference_ids or []:
        fid = str(raw).strip()
        if not fid or fid == primary or fid in seen:
            continue
        seen.add(fid)
        out.append(fid)
        if len(out) >= MAX_REFERENCE_IMAGES:
            break
    return out


def collect_local_image_paths(
    primary_path: str,
    reference_paths: Optional[Iterable[str]] = None,
) -> List[str]:
    """Primary first, unique local paths (for worker-side job inputs)."""
    out: List[str] = [primary_path]
    seen = {primary_path}
    for raw in reference_paths or []:
        p = str(raw).strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out[:MAX_TOTAL_IMAGES]


def should_use_multiview_mesh(image_paths: List[str], inputs: dict) -> bool:
    if inputs.get("use_multiview_mesh") is False:
        return False
    if inputs.get("use_multiview_mesh") is True and len(image_paths) >= 2:
        return True
    return len(image_paths) >= 2


def should_use_colmap_reconstruction(image_paths: List[str], inputs: dict) -> bool:
    if inputs.get("reconstruction_mode") == "generative":
        return False
    if inputs.get("reconstruction_mode") == "colmap":
        return len(image_paths) >= 3
    return len(image_paths) >= 3


def should_use_worldmirror_reconstruction(image_paths: List[str], inputs: dict) -> bool:
    if inputs.get("reconstruction_mode") == "generative":
        return False
    if inputs.get("reconstruction_mode") == "worldmirror":
        return len(image_paths) >= 1
    return len(image_paths) >= 2


def multi_image_generation_info(
    *,
    primary_file_id: Optional[str],
    reference_file_ids: Optional[Iterable[str]] = None,
    phase: str = "1_primary_only",
) -> dict:
    """Metadata block attached to splat/mesh job results."""
    refs = normalize_reference_image_file_ids(
        reference_file_ids, primary_file_id=primary_file_id
    )
    return {
        "multi_image_phase": phase,
        "primary_image_file_id": primary_file_id,
        "reference_image_file_ids": refs,
        "reference_count": len(refs),
        "total_image_count": 1 + len(refs),
    }
