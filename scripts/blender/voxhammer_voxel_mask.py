"""Blender entrypoint: VoxHammer voxel masking from mask GLB (needs bpy)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

root = Path(os.environ["DAIGC_ROOT"])
vox_root = root / "thirdparty" / "VoxHammer"
sys.path.insert(0, str(vox_root))

from voxhammer.delete_region_voxel import process_delete_ply  # noqa: E402


def main() -> None:
    job_path = os.environ["VOXHAMMER_JOB_JSON"]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    params = job["params"]
    process_delete_ply(
        params["mask_glb_path"],
        params["render_dir"],
        filter_method=params.get("filter_method", "volume"),
        voxel_size=params.get("voxel_size", 1 / 64),
    )


if __name__ == "__main__":
    main()
