"""
LingBot env-scan → 3D Gaussian Splatting (Spark.js) conversion.

Phase A (this module, shipped): colored point cloud → isotropic Gaussian PLY
  that Spark can load, plus COLMAP-text camera export for training.

Phase B (continue here): gsplat train on exported frames+cameras, then replace
  ``environment.ply`` with the optimized splat. See
  ``docs/LINGBOT_MAP_ENVIRONMENT_SCAN.md`` § 3DGS refinement.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 0-th order SH basis constant (Inria / gsplat convention)
_SH_C0 = 0.28209479177387814


def _as_scale_xyz_tuple(scale: Any) -> Tuple[float, float, float]:
    if isinstance(scale, (list, tuple)) and len(scale) >= 3:
        return float(scale[0]), float(scale[1]), float(scale[2])
    if isinstance(scale, dict):
        return (
            float(scale.get("x", scale.get("sx", 1.0))),
            float(scale.get("y", scale.get("sy", 1.0))),
            float(scale.get("z", scale.get("sz", 1.0))),
        )
    v = float(scale) if scale is not None else 1.0
    return v, v, v


def rgb_to_sh_dc(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert uchar RGB [0,255] → f_dc_0..2 (spherical-harmonics DC)."""
    rgb = np.asarray(rgb_u8, dtype=np.float64).reshape(-1, 3) / 255.0
    return ((rgb - 0.5) / _SH_C0).astype(np.float32)


def inverse_sigmoid(x: float) -> float:
    x = min(max(float(x), 1e-4), 1.0 - 1e-4)
    return math.log(x / (1.0 - x))


def estimate_point_adaptive_scales(
    verts: np.ndarray,
    *,
    k: int = 4,
    scale_mult: float = 0.5,
    min_scale: float = 0.0008,
    max_scale: float = 0.004,
) -> np.ndarray:
    """
    Per-point isotropic stddev from local spacing.

    Fixed ``0.012`` (~1.2 cm) was ~4–8× median nearest-neighbor distance on
    Office scans, so Spark looked soft vs the source photo. Adaptive scales
    keep coverage without melting edges.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    n = int(xyz.shape[0])
    if n < 2:
        return np.full((n,), float(min_scale), dtype=np.float32)
    k_eff = int(max(1, min(k, n - 1)))
    tree = cKDTree(xyz)
    # Query in chunks to bound peak RAM on large clouds.
    nn_mean = np.empty(n, dtype=np.float64)
    chunk = 250_000
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        dists, _ = tree.query(xyz[i0:i1], k=k_eff + 1, workers=-1)
        if k_eff == 1:
            nn_mean[i0:i1] = dists[:, 1]
        else:
            nn_mean[i0:i1] = dists[:, 1:].mean(axis=1)
    scales = np.clip(nn_mean * float(scale_mult), float(min_scale), float(max_scale))
    return scales.astype(np.float32)


def point_cloud_to_gaussian_ply(
    verts: np.ndarray,
    colors: np.ndarray,
    output_ply: Path,
    *,
    scale: float | None = None,
    opacity: float = 0.92,
    max_points: int = 750_000,
    scale_mult: float = 0.5,
    min_scale: float = 0.0008,
    max_scale: float = 0.004,
) -> int:
    """
    Pack XYZRGB points as Spark-compatible isotropic Gaussians (no training).

    Properties match TripoSplat / WorldMirror exports consumed by Spark.js.

    ``scale=None`` (default) → per-point adaptive stddev from kNN spacing.
    Pass a float to force a fixed isotropic scale (legacy / tests).
    """
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = int(verts.shape[0])
    if n < 10:
        raise RuntimeError(f"Too few points for Gaussian conversion ({n})")
    if n > max_points:
        stride = max(1, (n + max_points - 1) // max_points)
        verts = verts[::stride]
        colors = colors[::stride]
        n = int(verts.shape[0])

    f_dc = rgb_to_sh_dc(colors)
    opac = np.full((n,), inverse_sigmoid(opacity), dtype=np.float32)
    if scale is None:
        lin_scales = estimate_point_adaptive_scales(
            verts,
            scale_mult=scale_mult,
            min_scale=min_scale,
            max_scale=max_scale,
        )
    else:
        lin_scales = np.full((n,), float(max(scale, 1e-6)), dtype=np.float32)
    # log-scale (gsplat / Inria store log of stddev)
    log_s = np.log(np.maximum(lin_scales, 1e-6)).astype(np.float32)
    scales = np.stack([log_s, log_s, log_s], axis=1)
    # Identity quaternion w,x,y,z
    rots = np.zeros((n, 4), dtype=np.float32)
    rots[:, 0] = 1.0
    normals = np.zeros((n, 3), dtype=np.float32)

    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    packed = np.empty(n, dtype=dtype)
    packed["x"] = verts[:, 0]
    packed["y"] = verts[:, 1]
    packed["z"] = verts[:, 2]
    packed["nx"] = normals[:, 0]
    packed["ny"] = normals[:, 1]
    packed["nz"] = normals[:, 2]
    packed["f_dc_0"] = f_dc[:, 0]
    packed["f_dc_1"] = f_dc[:, 1]
    packed["f_dc_2"] = f_dc[:, 2]
    packed["opacity"] = opac
    packed["scale_0"] = scales[:, 0]
    packed["scale_1"] = scales[:, 1]
    packed["scale_2"] = scales[:, 2]
    packed["rot_0"] = rots[:, 0]
    packed["rot_1"] = rots[:, 1]
    packed["rot_2"] = rots[:, 2]
    packed["rot_3"] = rots[:, 3]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    )
    with output_ply.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(packed.tobytes())
    return n


def apply_gravity_to_c2w(
    c2w: np.ndarray,
    *,
    rotation_3x3: np.ndarray,
    y_flipped: bool = False,
    x_mirrored: bool = True,
    y_offset: float = 0.0,
    level_3x3: np.ndarray | None = None,
    yaw_3x3: np.ndarray | None = None,
    y_seat_final: float = 0.0,
) -> np.ndarray:
    """
    Apply the same uprighting used on the point cloud to camera-to-world (3x4 or 4x4).
    ``p' = p @ R.T`` on points ⇒ ``R_c2w' = R @ R_c2w`` for rotation part, t' = t @ R.T.
    """
    R = np.asarray(rotation_3x3, dtype=np.float64)
    mats = np.asarray(c2w, dtype=np.float64)
    single = mats.ndim == 2
    if single:
        mats = mats[None, ...]
    out = []
    Rl = np.asarray(level_3x3, dtype=np.float64) if level_3x3 is not None else None
    Ry = np.asarray(yaw_3x3, dtype=np.float64) if yaw_3x3 is not None else None
    for m in mats:
        if m.shape == (3, 4):
            M = np.eye(4)
            M[:3, :4] = m
        else:
            M = m.copy()
        Rw = M[:3, :3]
        tw = M[:3, 3]
        Rw2 = R @ Rw
        tw2 = tw @ R.T
        if y_flipped:
            # Reflect Y (same as verts[:,1] *= -1 before seating)
            S = np.diag([1.0, -1.0, 1.0])
            Rw2 = S @ Rw2
            tw2 = tw2 @ S
        if x_mirrored:
            S = np.diag([-1.0, 1.0, 1.0])
            Rw2 = S @ Rw2
            tw2 = tw2 @ S
        tw2 = tw2.copy()
        tw2[1] -= y_offset
        if Rl is not None:
            Rw2 = Rl @ Rw2
            tw2 = tw2 @ Rl.T
        if Ry is not None:
            Rw2 = Ry @ Rw2
            tw2 = tw2 @ Ry.T
        tw2 = tw2.copy()
        tw2[1] -= float(y_seat_final or 0.0)
        M2 = np.eye(4)
        M2[:3, :3] = Rw2
        M2[:3, 3] = tw2
        out.append(M2[:3, :4] if m.shape == (3, 4) else M2)
    arr = np.stack(out, axis=0)
    return arr[0] if single else arr


def export_colmap_text_from_lingbot(
    *,
    frames_dir: Path,
    extrinsic_c2w: np.ndarray,
    intrinsic: np.ndarray,
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    out_dir: Path,
    image_size: Tuple[int, int] = (518, 518),
) -> Path:
    """
    Write COLMAP TXT model (cameras.txt / images.txt / points3D.txt) + image copies.

    Uses LingBot poses (no SfM). Ready for Phase B gsplat / nerfstudio ingestion.
    """
    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    sparse = out_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    frames = sorted(Path(frames_dir).glob("frame_*.*"))
    ext = np.asarray(extrinsic_c2w)
    if ext.ndim == 2:
        ext = ext[None, ...]
    K = np.asarray(intrinsic)
    if K.ndim == 3:
        K0 = K[0]
    else:
        K0 = K
    fx = float(K0[0, 0])
    fy = float(K0[1, 1]) if K0.shape[0] > 1 else fx
    cx = float(K0[0, 2]) if K0.shape[1] > 2 else image_size[0] / 2
    cy = float(K0[1, 2]) if K0.shape[0] > 1 and K0.shape[1] > 2 else image_size[1] / 2
    w, h = int(image_size[0]), int(image_size[1])

    n_img = min(len(frames), int(ext.shape[0]))

    # Always persist matrix poses. Gravity X-mirror yields det(R)=-1, which
    # cannot be represented as a COLMAP quaternion (silent corruption → muddy 3DGS).
    poses = np.zeros((n_img, 4, 4), dtype=np.float32)
    improper = 0
    for i in range(n_img):
        c2w = ext[i]
        if c2w.shape == (3, 4):
            M = np.eye(4, dtype=np.float64)
            M[:3, :4] = c2w
        else:
            M = np.asarray(c2w, dtype=np.float64)
        poses[i] = M.astype(np.float32)
        if float(np.linalg.det(M[:3, :3])) < 0.0:
            improper += 1
    np.save(out_dir / "poses_c2w.npy", poses)

    # cameras.txt — single PINHOLE
    (sparse / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n",
        encoding="utf-8",
    )

    # images.txt — COLMAP stores WORLD-TO-CAMERA quaternion + translation.
    # When poses are improper (X-mirrored LingBot), write identity placeholders and
    # rely on poses_c2w.npy for training (see dataset_meta.poses_source).
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
        "# NOTE: If poses are improper (det=-1), quats here are NOT authoritative — use ../poses_c2w.npy",
    ]
    for i in range(n_img):
        M = poses[i].astype(np.float64)
        w2c = np.linalg.inv(M)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        src = frames[i]
        name = f"{i:06d}{src.suffix.lower()}"
        dest = images_out / name
        if not dest.exists():
            shutil.copy2(src, dest)
        if float(np.linalg.det(R)) < 0.0:
            # Placeholder: keep translation, identity rotation (trainer ignores this).
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            qw, qx, qy, qz = _rotmat_to_quat(R)
            n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
            qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        lines.append(
            f"{i + 1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
            f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {name}"
        )
        lines.append("")  # empty POINTS2D line
    (sparse / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # points3D.txt — subsample for COLMAP size
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(points_rgb, dtype=np.uint8).reshape(-1, 3)
    max_pts = 200_000
    if pts.shape[0] > max_pts:
        stride = max(1, pts.shape[0] // max_pts)
        pts = pts[::stride]
        rgb = rgb[::stride]
    p_lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for i, (p, c) in enumerate(zip(pts, rgb), start=1):
        p_lines.append(
            f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 0"
        )
    (sparse / "points3D.txt").write_text("\n".join(p_lines) + "\n", encoding="utf-8")

    meta = {
        "format": "colmap_txt",
        "num_images": n_img,
        "num_points": int(pts.shape[0]),
        "image_size": [w, h],
        "source": "lingbot_map",
        "poses_source": "poses_c2w.npy",
        "improper_rotations": int(improper),
        "colmap_images_txt_authoritative": improper == 0,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if improper:
        logger.warning(
            "Exported %d/%d improper c2w (det=-1). Phase B must use poses_c2w.npy — "
            "COLMAP images.txt quats are placeholders.",
            improper,
            n_img,
        )
    return sparse


def _rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix → (qw, qx, qy, qz). Raises if ``det(R) < 0`` (reflection)."""
    m = np.asarray(R, dtype=np.float64)
    det = float(np.linalg.det(m))
    if det < 0.0:
        raise ValueError(
            f"Cannot convert improper rotation (det={det:.6f}) to quaternion. "
            "LingBot X-mirror yields det=-1; Phase B must load poses_c2w.npy / "
            "cameras_aligned.npz matrices instead of COLMAP images.txt quats."
        )
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def refine_point_cloud_world_to_gaussian(
    world_dir: Path,
    *,
    frames_dir: Optional[Path] = None,
    cameras_npz: Optional[Path] = None,
    export_colmap: bool = True,
) -> Dict[str, Any]:
    """
    Phase A: convert ``environment.ply`` (XYZRGB) → Gaussian ``environment.ply``,
    keep XYZRGB backup, update manifest to ``gaussian_splat``.
    """
    from core.utils.lingbot_map_pipeline import _read_ply_xyzrgb_numpy

    world_dir = Path(world_dir)
    env_ply = world_dir / "environment.ply"
    if not env_ply.is_file():
        raise FileNotFoundError(f"Missing environment.ply under {world_dir}")

    verts, colors = _read_ply_xyzrgb_numpy(env_ply)
    backup = world_dir / "environment.points.ply"
    if not backup.exists():
        shutil.copy2(env_ply, backup)

    man_path = world_dir / "world.manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    env = dict(manifest.get("environment") or {})
    transform = dict(env.get("transform") or {})
    sx, sy, sz = _as_scale_xyz_tuple(transform.get("scale", 1.0))
    metric_baked = False
    # Non-uniform Three.js scale smears Spark covariances (soft blob). Bake the
    # door metric into Gaussian centers and leave identity scale for the viewport.
    if abs(sx - 1.0) > 1e-4 or abs(sy - 1.0) > 1e-4 or abs(sz - 1.0) > 1e-4:
        verts = np.asarray(verts, dtype=np.float64).copy()
        verts[:, 0] *= sx
        verts[:, 1] *= sy
        verts[:, 2] *= sz
        verts = verts.astype(np.float32)
        transform["scale"] = [1.0, 1.0, 1.0]
        metric_baked = True

    n = point_cloud_to_gaussian_ply(verts, colors, env_ply)
    env["type"] = "gaussian_splat"
    env["format"] = "ply"
    env["renderer"] = "spark"
    env["url"] = "environment.ply"
    env["transform"] = transform
    manifest["environment"] = env
    meta = dict(manifest.get("metadata") or {})
    meta["source_geometry"] = "gaussian_from_point_cloud"
    meta["gaussian_phase"] = "A_isotropic_from_points"
    meta["gaussian_scale_mode"] = "adaptive_knn"
    meta["gaussian_count"] = n
    meta["point_cloud_backup"] = "environment.points.ply"
    if metric_baked:
        meta["metric_baked_into_assets"] = True
        cal = dict(meta.get("metric_calibration") or {})
        cal["baked_into_assets"] = True
        cal["scale_xyz"] = [sx, sy, sz]
        cal["manifest_transform_scale"] = [1.0, 1.0, 1.0]
        meta["metric_calibration"] = cal
    manifest["metadata"] = meta
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    colmap_dir = None
    if export_colmap and cameras_npz and Path(cameras_npz).is_file() and frames_dir:
        try:
            data = np.load(cameras_npz, allow_pickle=True)
            sparse = export_colmap_text_from_lingbot(
                frames_dir=Path(frames_dir),
                extrinsic_c2w=data["extrinsic"],
                intrinsic=data["intrinsic"],
                points_xyz=verts,
                points_rgb=colors,
                out_dir=world_dir / "gs_dataset",
            )
            colmap_dir = str(sparse)
            meta["gs_dataset"] = "gs_dataset"
            meta["colmap_sparse"] = colmap_dir
            man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("COLMAP export for Phase B skipped: %s", exc)

    return {
        "gaussian_count": n,
        "environment_ply": str(env_ply),
        "backup_ply": str(backup),
        "colmap_sparse": colmap_dir,
        "phase": "A_isotropic_from_points",
        "next": (
            "Phase B: train gsplat on gs_dataset/ (COLMAP TXT + images), "
            "export optimized PLY, replace environment.ply — see "
            "docs/LINGBOT_MAP_ENVIRONMENT_SCAN.md"
        ),
    }
