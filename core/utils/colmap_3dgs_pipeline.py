"""
COLMAP sparse reconstruction → Gaussian PLY export (Phase 3).

Requires the ``colmap`` binary on PATH (``sudo apt install colmap`` on Ubuntu).
When COLMAP is unavailable, use TripoSplat on the primary image instead.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def colmap_available() -> bool:
    return shutil.which("colmap") is not None


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    logger.info("COLMAP: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"COLMAP command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )


def run_colmap_sparse_reconstruction(
    image_paths: List[str],
    workspace: Path,
) -> Path:
    """
    Run feature extraction + matching + mapper.

    Returns path to sparse model ``sparse/0``.
    """
    if not colmap_available():
        raise RuntimeError(
            "COLMAP is not installed. On DGX/Ubuntu run: sudo apt install colmap"
        )
    if len(image_paths) < 3:
        raise ValueError("COLMAP reconstruction requires at least 3 images")

    workspace = Path(workspace)
    images_dir = workspace / "images"
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(image_paths):
        src_path = Path(src)
        if not src_path.is_file():
            raise FileNotFoundError(f"Image not found: {src}")
        dest = images_dir / f"view_{i:03d}{src_path.suffix.lower() or '.jpg'}"
        if not dest.exists():
            dest.write_bytes(src_path.read_bytes())

    if database.exists():
        database.unlink()

    _run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.single_camera",
            "1",
        ],
        cwd=workspace,
    )
    _run(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database),
        ],
        cwd=workspace,
    )
    _run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        cwd=workspace,
    )

    model0 = sparse_dir / "0"
    if not model0.is_dir():
        raise RuntimeError(f"COLMAP mapper produced no model at {model0}")

    _run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(model0),
            "--output_path",
            str(model0),
            "--output_type",
            "TXT",
        ],
        cwd=workspace,
    )
    return model0


def _read_points3d_txt(model_dir: Path) -> np.ndarray:
    points_file = model_dir / "points3D.txt"
    if not points_file.is_file():
        raise RuntimeError(f"Missing {points_file} after COLMAP export")
    rows: List[List[float]] = []
    for line in points_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        rows.append([x, y, z])
    if not rows:
        raise RuntimeError("COLMAP sparse model has no points")
    return np.asarray(rows, dtype=np.float32)


def _center_and_scale_points(points: np.ndarray) -> np.ndarray:
    """Roughly normalize to ~2 m tall scene for Spark.js viewing."""
    pts = points.copy()
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) / 2.0
    pts -= center
    height = max(float(mx[1] - mn[1]), 1e-3)
    scale = 1.8 / height
    pts *= scale
    return pts


def export_colmap_points_to_ply(model_dir: Path, output_ply: Path) -> int:
    """
    Export COLMAP sparse points as a minimal Gaussian-compatible PLY.

    Each point becomes a small isotropic Gaussian (position + white color).
    """
    points = _read_points3d_txt(model_dir)
    points = _center_and_scale_points(points)
    n = len(points)
    colors = np.full((n, 3), 220, dtype=np.uint8)
    scales = np.full((n, 3), 0.02, dtype=np.float32)

    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    with output_ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            sx, sy, sz = scales[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {sx:.6f} {sy:.6f} {sz:.6f}\n")

    return n


def reconstruct_photos_to_ply(
    image_paths: List[str],
    output_ply: Path,
    *,
    workspace: Optional[Path] = None,
) -> dict:
    """Full COLMAP sparse → PLY path."""
    workspace = Path(workspace or output_ply.parent / "colmap_workspace")
    model_dir = run_colmap_sparse_reconstruction(image_paths, workspace)
    count = export_colmap_points_to_ply(model_dir, output_ply)
    return {
        "point_count": count,
        "image_count": len(image_paths),
        "colmap_model": str(model_dir),
        "workspace": str(workspace),
    }
