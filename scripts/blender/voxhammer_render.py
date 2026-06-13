"""Blender entrypoint: VoxHammer multi-view rendering (needs bpy)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

root = Path(os.environ["DAIGC_ROOT"])
vox_root = root / "thirdparty" / "VoxHammer"
sys.path.insert(0, str(vox_root))

from voxhammer.bpy_render import render_3d_model  # noqa: E402


def main() -> None:
    job_path = os.environ["VOXHAMMER_JOB_JSON"]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    render_3d_model(**job["params"])


if __name__ == "__main__":
    main()
