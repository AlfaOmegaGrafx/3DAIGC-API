#!/usr/bin/env python3
"""
Phase B: train gsplat on an env-scan world ``gs_dataset/`` and replace environment.ply.

Usage:
  cd /home/sifr/3DAIGC-API
  source venv/bin/activate
  python scripts/train_env_scan_3dgs.py outputs/worlds/<JOB_ID>
  python scripts/train_env_scan_3dgs.py outputs/worlds/<JOB_ID> --max-steps 7000
  python scripts/train_env_scan_3dgs.py outputs/worlds/<JOB_ID> --max-steps 200 --max-images 24  # smoke
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
    parser = argparse.ArgumentParser(description="Env-scan Phase B gsplat train")
    parser.add_argument("world_dir", type=Path, help="World package with gs_dataset/")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--data-factor", type=int, default=2, help="Image downsample (2 → 1/2)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=800_000)
    parser.add_argument("--sh-degree", type=int, default=0, help="0 = DC only (Spark-compatible)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--enable-densify",
        action="store_true",
        help="Allow gsplat clone/split (off by default; caused floaters on LingBot poses)",
    )
    args = parser.parse_args()

    world_dir = args.world_dir.resolve()
    if not (world_dir / "gs_dataset").is_dir():
        print(f"ERROR: missing {world_dir / 'gs_dataset'}", file=sys.stderr)
        return 1

    from core.utils.lingbot_3dgs_train import PhaseBTrainConfig, train_and_apply_phase_b

    cfg = PhaseBTrainConfig(
        max_steps=args.max_steps,
        data_factor=args.data_factor,
        max_images=args.max_images,
        max_points=args.max_points,
        sh_degree=max(0, args.sh_degree),
        device=args.device,
        enable_densify=args.enable_densify,
        refine_stop_iter=min(6000, max(500, args.max_steps - 100)) if args.enable_densify else 50_000,
        refine_start_iter=500 if args.enable_densify else 50_000,
    )
    info = train_and_apply_phase_b(world_dir, cfg=cfg)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
