"""Multi-image job helpers (Phase 1–3 — docs/MULTI_IMAGE_SPLAT_ROADMAP.md)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

MAX_REFERENCE_IMAGES = 7
MAX_TOTAL_IMAGES = MAX_REFERENCE_IMAGES + 1


def normalize_reference_image_file_ids(
    reference_file_ids: Optional[Sequence[str]],
    *,
    primary_file_id: Optional[str] = None,
    max_refs: int = MAX_REFERENCE_IMAGES,
) -> List[str]:
    """Dedupe reference file IDs, drop primary, cap count."""
    out: List[str] = []
    seen = set()
    primary = (primary_file_id or "").strip()
    for raw in reference_file_ids or []:
        fid = (raw or "").strip()
        if not fid or fid == primary or fid in seen:
            continue
        seen.add(fid)
        out.append(fid)
        if len(out) >= max_refs:
            break
    return out


def collect_local_image_paths(
    primary_path: str,
    reference_paths: Optional[Iterable[str]] = None,
    *,
    max_total: int = MAX_TOTAL_IMAGES,
) -> List[str]:
    """Primary first, then unique existing reference paths."""
    primary = str(primary_path)
    out: List[str] = [primary]
    seen = {primary}
    for raw in reference_paths or []:
        p = str(raw)
        if p in seen:
            continue
        if not Path(p).is_file():
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= max_total:
            break
    return out


def _flag(inputs: dict, key: str, default: bool = False) -> bool:
    val = inputs.get(key)
    if val is None:
        return default
    return bool(val)


def should_use_multiview_mesh(image_paths: Sequence[str], inputs: dict) -> bool:
    """TRELLIS v1 multiview when 2+ views and not explicitly disabled."""
    if inputs.get("reconstruction_mode") == "single":
        return False
    if _flag(inputs, "use_multiview_mesh", default=True) is False:
        return False
    return len(image_paths) >= 2


def should_use_colmap_reconstruction(image_paths: Sequence[str], inputs: dict) -> bool:
    """COLMAP path when 3+ views or explicitly requested."""
    mode = inputs.get("reconstruction_mode")
    if mode == "generative":
        return False
    if mode == "colmap":
        return len(image_paths) >= 1
    if _flag(inputs, "use_colmap_reconstruction", default=False):
        return len(image_paths) >= 3
    return len(image_paths) >= 3


def should_use_worldmirror_reconstruction(image_paths: Sequence[str], inputs: dict) -> bool:
    if inputs.get("reconstruction_mode") == "generative":
        return False
    if inputs.get("reconstruction_mode") == "worldmirror":
        return len(image_paths) >= 1
    return len(image_paths) >= 2


def multi_image_generation_info(
    image_paths: Sequence[str],
    *,
    phase: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "multi_image_count": len(image_paths),
        "multi_image_phase": phase,
    }
    if extra:
        info.update(extra)
    return info
