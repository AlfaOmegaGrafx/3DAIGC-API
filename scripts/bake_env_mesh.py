#!/usr/bin/env python3
"""CLI: bake env-scan world Gaussians → environment_mesh.glb."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("world_id", help="outputs/worlds/<id>")
    p.add_argument(
        "--quality",
        default="photo",
        choices=("draft", "balanced", "photo"),
        help="Bake quality preset (default: photo)",
    )
    p.add_argument("--target-faces", type=int, default=None)
    p.add_argument("--voxel-res", type=int, default=None)
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--data-factor", type=int, default=None)
    p.add_argument(
        "--color-export",
        default=None,
        choices=("vertex", "atlas"),
        help="vertex=COLOR_0 studio; atlas=OMB face-island PBR",
    )
    args = p.parse_args()

    from core.utils.world_env_mesh_bake import EnvMeshBakeConfig, bake_world_env_mesh

    root = Path("outputs") / "worlds" / args.world_id
    cfg = EnvMeshBakeConfig.from_quality(
        args.quality,
        target_face_count=args.target_faces,
        voxel_resolution=args.voxel_res,
        max_views=args.max_views,
        data_factor=args.data_factor,
        color_export=args.color_export,
    )
    info = bake_world_env_mesh(root, cfg=cfg)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
