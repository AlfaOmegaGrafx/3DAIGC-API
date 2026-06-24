"""Publish GLB assets to the local MSF Map Service static object tree."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, Optional


def _slugify(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-")
    return base or "asset"


def publish_glb_to_msf(
    source_path: str,
    *,
    asset_name: str,
    objects_dir: Optional[str] = None,
    public_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    objects_dir = objects_dir or os.environ.get(
        "MSF_OBJECTS_DIR", "/home/sifr/MSF_Map_Svc/dist/web/objects"
    )
    public_base_url = (public_base_url or os.environ.get("MSF_PUBLIC_BASE_URL", "")).rstrip("/")

    os.makedirs(objects_dir, exist_ok=True)
    slug = _slugify(asset_name)
    if not slug.lower().endswith(".glb"):
        slug = f"{slug}.glb"

    dest_path = os.path.join(objects_dir, slug)
    shutil.copy2(source_path, dest_path)

    object_url = f"{public_base_url}/objects/{slug}" if public_base_url else f"/objects/{slug}"

    return {
        "object_name": slug,
        "object_path": dest_path,
        "object_url": object_url,
        "resource_path": f"/objects/{slug}",
    }
