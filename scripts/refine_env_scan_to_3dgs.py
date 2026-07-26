#!/usr/bin/env python3
"""
Phase A: convert an existing LingBot environment-scan world (XYZRGB PLY)
into a Spark-compatible isotropic Gaussian PLY.

Usage:
  cd /home/sifr/3DAIGC-API
  source .venv/bin/activate   # or whatever venv the API uses
  python scripts/refine_env_scan_to_3dgs.py \\
    outputs/worlds/897b3ea3-b819-4d39-b49a-5d0be253d592

Optional COLMAP export (Phase B prep) when cameras + frames exist:
  python scripts/refine_env_scan_to_3dgs.py WORLD_DIR \\
    --frames WORLD_DIR/_work/frames_flat \\
    --cameras WORLD_DIR/cameras_aligned.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Env-scan point cloud → Gaussian PLY (Phase A)")
    parser.add_argument("world_dir", type=Path, help="World package directory (has environment.ply)")
    parser.add_argument("--frames", type=Path, default=None, help="frames_flat dir for COLMAP export")
    parser.add_argument("--cameras", type=Path, default=None, help="cameras_aligned.npz")
    parser.add_argument("--no-colmap", action="store_true", help="Skip COLMAP / gs_dataset export")
    args = parser.parse_args()

    world_dir = args.world_dir.resolve()
    if not (world_dir / "environment.ply").is_file():
        print(f"ERROR: missing {world_dir / 'environment.ply'}", file=sys.stderr)
        return 1

    frames = args.frames
    if frames is None:
        cand = world_dir / "_work" / "frames_flat"
        if cand.is_dir():
            frames = cand

    cameras = args.cameras
    if cameras is None:
        for cand in (
            world_dir / "cameras_aligned.npz",
            world_dir / "_work" / "lingbot_out" / "cameras_aligned.npz",
        ):
            if cand.is_file():
                cameras = cand
                break

    from core.utils.lingbot_3dgs_refine import refine_point_cloud_world_to_gaussian

    info = refine_point_cloud_world_to_gaussian(
        world_dir,
        frames_dir=frames,
        cameras_npz=cameras,
        export_colmap=not args.no_colmap,
    )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
