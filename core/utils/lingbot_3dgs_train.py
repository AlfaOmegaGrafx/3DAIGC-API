"""
Phase B — gsplat train on LingBot env-scan ``gs_dataset/``.

Loads COLMAP TXT (no pycolmap), trains with ``gsplat.rasterization`` +
``DefaultStrategy``, exports Spark-compatible PLY, updates world manifest.

Preserves gravity-aligned world coordinates (no scene renormalization).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

_SH_C0 = 0.28209479177387814


@dataclass
class PhaseBTrainConfig:
    max_steps: int = 10_000
    data_factor: int = 2
    max_images: Optional[int] = None
    max_points: int = 800_000
    # Spark / TripoSplat only consume DC — train SH0 so photometric signal isn't
    # wasted on f_rest bands that get stripped on export.
    sh_degree: int = 0
    sh_degree_interval: int = 1_000
    ssim_lambda: float = 0.2
    init_opacity: float = 0.85
    means_lr: float = 4e-5
    scales_lr: float = 8e-3
    # Shrink Phase A blobs so optimization can fit sharper surfaces.
    init_scale_mult: float = 0.55
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Densify off by default — cloning with imperfect LingBot poses spawned
    # room-scale floaters (497k Gaussians, <0.01% inside Phase A AABB).
    refine_start_iter: int = 50_000
    refine_stop_iter: int = 50_000
    refine_every: int = 100
    # Opacity resets wipe density and left Phase B as a muddy translucent cloud.
    reset_every: int = 100_000
    packed: bool = False
    device: str = "cuda"
    num_workers: int = 2
    seed: int = 42
    # Prefer Phase A isotropic Gaussians as init when present next to gs_dataset.
    init_from_phase_a: bool = True
    phase_a_ply: Optional[str] = None
    # Soft pull toward init means (discourages drift out of the room).
    means_l2_lambda: float = 0.015
    means_l2_until: int = 2_000
    # Allow opting into densify for datasets with true COLMAP SfM poses.
    enable_densify: bool = False
    # Bake non-uniform metric into PLY is OFF by default — SVD bake produced
    # needle Gaussians (aniso >1000) that Spark draws as black spikes.
    bake_metric_scale: bool = False


@dataclass
class ColorRefineConfig:
    """Photometric color-only pass (freeze geometry)."""

    max_steps: int = 5_000
    data_factor: int = 2
    sh_lr: float = 2.5e-2
    ssim_lambda: float = 0.0  # L1-only — SSIM was unstable on LingBot poses
    device: str = "cuda"
    num_workers: int = 2
    seed: int = 42
    # Served LingBot worlds are already in Studio gravity frame — do not X-flip.
    studio_frame: bool = True


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / _SH_C0


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP (qw,qx,qy,qz) → 3x3 rotation (normalizes q)."""
    qw, qx, qy, qz = [float(x) for x in q]
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def _parse_colmap_cameras(path: Path) -> Dict[int, Dict[str, Any]]:
    cams: Dict[int, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        cam_id = int(toks[0])
        model = toks[1]
        width, height = int(toks[2]), int(toks[3])
        params = [float(x) for x in toks[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {model}")
        cams[cam_id] = {
            "model": model,
            "width": width,
            "height": height,
            "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
        }
    if not cams:
        raise RuntimeError(f"No cameras in {path}")
    return cams


def _parse_colmap_images(path: Path) -> List[Dict[str, Any]]:
    """
    Parse COLMAP images.txt.

    Each image is two lines (pose + POINTS2D). Empty POINTS2D lines may disappear
    when stripping blanks — accept any line with ≥10 pose tokens.
    """
    images: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 10:
            continue
        try:
            int(toks[0])
            float(toks[1])
        except ValueError:
            continue
        q = np.array([float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])])
        t = np.array([float(toks[5]), float(toks[6]), float(toks[7])], dtype=np.float64)
        cam_id = int(toks[8])
        name = toks[9]
        R = _quat_to_rotmat(q)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        images.append({"name": name, "camera_id": cam_id, "c2w": c2w.astype(np.float32)})
    if not images:
        raise RuntimeError(f"No images in {path}")
    images.sort(key=lambda x: x["name"])
    return images


def _parse_colmap_points3d(path: Path, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    pts: List[List[float]] = []
    cols: List[List[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 7:
            continue
        pts.append([float(toks[1]), float(toks[2]), float(toks[3])])
        cols.append([int(toks[4]), int(toks[5]), int(toks[6])])
    if not pts:
        raise RuntimeError(f"No points3D in {path}")
    xyz = np.asarray(pts, dtype=np.float32)
    rgb = np.asarray(cols, dtype=np.uint8)
    if xyz.shape[0] > max_points:
        stride = max(1, xyz.shape[0] // max_points)
        xyz = xyz[::stride][:max_points]
        rgb = rgb[::stride][:max_points]
    return xyz, rgb


def _knn_avg_dist(points: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Mean distance to k nearest neighbors (excluding self). Chunked for memory."""
    n = points.shape[0]
    if n <= 1:
        return torch.full((n,), 0.01, device=points.device, dtype=points.dtype)
    k = min(k + 1, n)
    out = torch.empty(n, device=points.device, dtype=points.dtype)
    chunk = 8192 if n > 20_000 else n
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist = torch.cdist(points[start:end], points)
        knn = torch.topk(dist, k=k, largest=False).values
        out[start:end] = knn[:, 1:].mean(dim=-1).clamp_min(1e-4)
    return out


def _as_c2w_4x4(ext: np.ndarray) -> np.ndarray:
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        M = np.eye(4, dtype=np.float64)
        M[:3, :4] = ext
        return M
    raise ValueError(f"Expected c2w 3x4 or 4x4, got {ext.shape}")


def _flip_x_c2w(c2w: np.ndarray) -> np.ndarray:
    """Undo/apply LingBot Studio X-mirror on a camera-to-world matrix."""
    M = _as_c2w_4x4(c2w).copy()
    S = np.diag([-1.0, 1.0, 1.0])
    M[:3, :3] = S @ M[:3, :3]
    M[:3, 3] = S @ M[:3, 3]
    return M


def _flip_x_points(xyz: np.ndarray) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float32).copy()
    out[:, 0] *= -1.0
    return out


def _quat_wxyz_to_rotmat_batch(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _flip_x_quats_wxyz(quats: np.ndarray) -> np.ndarray:
    """
    Apply Studio X-mirror to Gaussian orientations: R' = S R S with S=diag(-1,1,1).
    Keeps SO(3) (det=+1) so quaternions stay valid.
    """
    from scipy.spatial.transform import Rotation

    R = _quat_wxyz_to_rotmat_batch(quats)
    s = np.array([-1.0, 1.0, 1.0])
    R2 = R * s.reshape(1, 3, 1) * s.reshape(1, 1, 3)
    q_xyzw = Rotation.from_matrix(R2).as_quat()  # x,y,z,w
    return np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1).astype(np.float32)


def _load_matrix_poses(data_dir: Path, n_expected: int) -> Optional[np.ndarray]:
    """
    Load authoritative c2w poses (4x4). Prefer poses_c2w.npy, then world
    cameras_aligned.npz. Returns None if unavailable.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "poses_c2w.npy",
        data_dir.parent / "cameras_aligned.npz",
        data_dir.parent / "cameras.npz",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".npy":
            arr = np.load(path)
            poses = np.stack([_as_c2w_4x4(m) for m in arr], axis=0).astype(np.float32)
            logger.info("Phase B poses from %s (%d)", path, len(poses))
            return poses
        data = np.load(path, allow_pickle=True)
        key = "extrinsic" if "extrinsic" in data.files else (
            "extrinsic_raw" if "extrinsic_raw" in data.files else None
        )
        if key is None:
            continue
        ext = np.asarray(data[key])
        poses = np.stack([_as_c2w_4x4(m) for m in ext], axis=0).astype(np.float32)
        logger.info("Phase B poses from %s[%s] (%d)", path.name, key, len(poses))
        if key == "extrinsic_raw":
            logger.warning(
                "Using raw (pre-gravity) cameras — prefer cameras_aligned.npz"
            )
        return poses
    return None


def _read_phase_a_gaussian_init(
    ply_path: Path, max_points: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return means, rgb[0..1], opacity_logit, log_scales[N,3], quats_wxyz."""
    raw = Path(ply_path).read_bytes()
    sep = b"end_header\n"
    idx = raw.find(sep)
    if idx < 0:
        raise RuntimeError(f"Bad Phase A PLY: {ply_path}")
    header = raw[: idx + len(sep)].decode("ascii", errors="replace")
    props = [
        ln.split()[-1]
        for ln in header.splitlines()
        if ln.startswith("property float")
    ]
    n = int(
        next(ln.split()[-1] for ln in header.splitlines() if ln.startswith("element vertex"))
    )
    arr = np.frombuffer(raw[idx + len(sep) :], dtype="<f4")
    if arr.size != n * len(props):
        raise RuntimeError(f"Phase A PLY size mismatch: {ply_path}")
    verts = arr.reshape(n, len(props))
    pidx = {p: i for i, p in enumerate(props)}
    for req in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0"):
        if req not in pidx:
            raise RuntimeError(f"Phase A missing {req}")
    if n > max_points:
        sel = np.linspace(0, n - 1, max_points, dtype=int)
        verts = verts[sel]
    means = verts[:, [pidx["x"], pidx["y"], pidx["z"]]].astype(np.float32)
    sh = verts[:, [pidx["f_dc_0"], pidx["f_dc_1"], pidx["f_dc_2"]]].astype(np.float32)
    rgb = np.clip(0.5 + _SH_C0 * sh, 0.0, 1.0).astype(np.float32)
    opacity = verts[:, pidx["opacity"]].astype(np.float32)
    scales = verts[:, [pidx["scale_0"], pidx["scale_1"], pidx["scale_2"]]].astype(np.float32)
    quats = verts[:, [pidx["rot_0"], pidx["rot_1"], pidx["rot_2"], pidx["rot_3"]]].astype(
        np.float32
    )
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True).clip(min=1e-6)
    return means, rgb, opacity, scales, quats, sh


class ColmapTxtDataset(Dataset):
    """Train views from ``gs_dataset/`` (matrix poses + images)."""

    def __init__(
        self,
        data_dir: Path,
        *,
        data_factor: int = 4,
        max_images: Optional[int] = None,
        max_points: int = 200_000,
        studio_frame: bool = False,
    ):
        data_dir = Path(data_dir)
        sparse = data_dir / "sparse" / "0"
        if not sparse.is_dir():
            sparse = data_dir / "sparse"
        images_dir = data_dir / "images"
        if not sparse.is_dir() or not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected gs_dataset with sparse/ + images/ under {data_dir}"
            )

        cams = _parse_colmap_cameras(sparse / "cameras.txt")
        images = _parse_colmap_images(sparse / "images.txt")
        points, points_rgb = _parse_colmap_points3d(sparse / "points3D.txt", max_points)

        matrix_poses = _load_matrix_poses(data_dir, len(images))
        self.pose_source = "poses_c2w_matrix" if matrix_poses is not None else "colmap_images_txt"
        self.train_x_flipped = False
        self.studio_frame = studio_frame

        if max_images is not None and len(images) > max_images:
            # Even stride so coverage stays spatial.
            idx = np.linspace(0, len(images) - 1, max_images, dtype=int)
            images = [images[i] for i in idx]
            if matrix_poses is not None:
                matrix_poses = matrix_poses[idx]

        self.camtoworlds: List[np.ndarray] = []
        self.Ks: List[np.ndarray] = []
        self.image_paths: List[Path] = []
        self.heights: List[int] = []
        self.widths: List[int] = []

        for i, im in enumerate(images):
            path = images_dir / im["name"]
            if not path.is_file():
                logger.warning("Missing image %s — skip", path)
                continue
            img = imageio.imread(path)[..., :3]
            h0, w0 = img.shape[:2]
            cam = cams[im["camera_id"]]
            K = cam["K"].copy()
            # Scale COLMAP intrinsics from declared size → actual pixels.
            sx = w0 / float(cam["width"])
            sy = h0 / float(cam["height"])
            K[0, :] *= sx
            K[1, :] *= sy

            if data_factor > 1:
                w = max(1, int(round(w0 / data_factor)))
                h = max(1, int(round(h0 / data_factor)))
                from PIL import Image

                img = np.asarray(
                    Image.fromarray(img).resize((w, h), Image.BICUBIC),
                    dtype=np.uint8,
                )
                K[0, :] /= data_factor
                K[1, :] /= data_factor
            else:
                h, w = h0, w0

            if matrix_poses is not None:
                c2w = _as_c2w_4x4(matrix_poses[i]).astype(np.float32)
            else:
                c2w = np.asarray(im["c2w"], dtype=np.float32)
                if c2w.shape == (3, 4):
                    c2w = _as_c2w_4x4(c2w).astype(np.float32)
                logger.warning(
                    "Phase B falling back to COLMAP images.txt poses — "
                    "unsafe when LingBot X-mirror made det(R)=-1"
                )

            self.image_paths.append(path)
            self.camtoworlds.append(c2w)
            self.Ks.append(K.astype(np.float32))
            self.heights.append(int(img.shape[0]))
            self.widths.append(int(img.shape[1]))
            # Cache resized bytes on disk under gs_dataset/_cache_fxN for workers.
            cache_dir = data_dir / f"_cache_f{data_factor}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{path.stem}.png"
            if not cache_path.is_file():
                imageio.imwrite(cache_path, img)
            self.image_paths[-1] = cache_path

        if not self.image_paths:
            raise RuntimeError(f"No usable images under {images_dir}")

        # Right-handed train frame: undo Studio X-mirror when poses are improper.
        # Skip when refining a served Studio-frame PLY (viewer uses same frame).
        dets = [float(np.linalg.det(c[:3, :3])) for c in self.camtoworlds]
        if not studio_frame and sum(d < 0 for d in dets) > len(dets) // 2:
            self.camtoworlds = [_flip_x_c2w(c).astype(np.float32) for c in self.camtoworlds]
            points = _flip_x_points(points)
            self.train_x_flipped = True
            logger.info(
                "Phase B: undid X-mirror for right-handed gsplat train "
                "(%d improper → flipped)",
                sum(d < 0 for d in dets),
            )

        self.points = points
        self.points_rgb = points_rgb
        cam_locs = np.stack([c[:3, 3] for c in self.camtoworlds], axis=0)
        center = cam_locs.mean(axis=0)
        cam_scale = float(np.linalg.norm(cam_locs - center, axis=1).max())
        # Prefer camera baseline; fall back to point extent if poses are degenerate.
        pts_extent = float(np.linalg.norm(points.max(0) - points.min(0)))
        self.scene_scale = cam_scale if np.isfinite(cam_scale) and cam_scale > 1e-3 else pts_extent
        self.scene_scale = float(np.clip(max(self.scene_scale, 1e-2), 1e-2, 50.0))

        logger.info(
            "Phase B dataset: %d images, %d points, scene_scale=%.3f, factor=%d, "
            "poses=%s, train_x_flipped=%s",
            len(self.image_paths),
            self.points.shape[0],
            self.scene_scale,
            data_factor,
            self.pose_source,
            self.train_x_flipped,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = imageio.imread(self.image_paths[idx])[..., :3]
        return {
            "image": torch.from_numpy(image.copy()),
            "camtoworld": torch.from_numpy(self.camtoworlds[idx].copy()),
            "K": torch.from_numpy(self.Ks[idx].copy()),
            "image_id": torch.tensor(idx, dtype=torch.long),
        }


def _create_splats(
    points: np.ndarray,
    points_rgb: np.ndarray,
    *,
    scene_scale: float,
    sh_degree: int,
    init_opacity: float,
    means_lr: float,
    device: str,
    phase_a: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    train_x_flipped: bool = False,
    init_scale_mult: float = 1.0,
    scales_lr: float = 5e-3,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if phase_a is not None:
        means_np, rgb_np, opac_logit, log_scales, quats_np, _sh_dc = phase_a
        if train_x_flipped:
            means_np = _flip_x_points(means_np)
            quats_np = _flip_x_quats_wxyz(quats_np)
        pts = torch.from_numpy(means_np).float()
        rgbs = torch.from_numpy(rgb_np).float()
        scales = torch.from_numpy(log_scales).float()
        quats = torch.from_numpy(quats_np).float()
        opacities = torch.from_numpy(opac_logit).float()
        logger.info("Phase B init from Phase A: %d Gaussians", pts.shape[0])
    else:
        pts = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(points_rgb.astype(np.float32) / 255.0)
        dist = _knn_avg_dist(pts, k=4)
        scales = torch.log(dist).unsqueeze(-1).repeat(1, 3)
        n = pts.shape[0]
        quats = torch.rand(n, 4)
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        opacities = torch.logit(torch.full((n,), init_opacity))

    if init_scale_mult != 1.0 and init_scale_mult > 0:
        scales = scales + math.log(float(init_scale_mult))

    n = pts.shape[0]
    colors = torch.zeros(n, (sh_degree + 1) ** 2, 3)
    colors[:, 0, :] = _rgb_to_sh(rgbs)

    # Empty shN when sh_degree==0 — still register param for export_splats.
    shn = colors[:, 1:, :]
    params = [
        ("means", torch.nn.Parameter(pts), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), 5e-3),
        ("shN", torch.nn.Parameter(shn), 2.5e-3 / 20),
    ]
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for name, _, lr in params
        if splats[name].numel() > 0 or name in ("sh0", "means", "scales", "quats", "opacities")
    }
    # shN may be empty (0 bands) — Adam still needs a param; keep a tiny placeholder.
    if splats["shN"].numel() == 0:
        splats["shN"] = torch.nn.Parameter(torch.zeros(n, 0, 3, device=device))
        optimizers["shN"] = torch.optim.Adam(
            [{"params": splats["shN"], "lr": 1e-4, "name": "shN"}], eps=1e-15
        )
    return splats, optimizers


def _resolve_phase_a_ply(data_dir: Path, cfg: PhaseBTrainConfig) -> Optional[Path]:
    if not cfg.init_from_phase_a:
        return None
    if cfg.phase_a_ply:
        p = Path(cfg.phase_a_ply)
        return p if p.is_file() else None
    for cand in (
        data_dir.parent / "environment.phaseA.ply",
        data_dir.parent / "environment.ply",
    ):
        if not cand.is_file():
            continue
        head = cand.read_bytes()[:240].decode("ascii", errors="ignore")
        if "f_dc_0" in head:
            return cand
    return None


def _make_ssim(device: str):
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def _ssim_loss(ssim_metric, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred/gt: [B,H,W,3] in [0,1] → 1 - SSIM."""
    p = pred.permute(0, 3, 1, 2).clamp(0, 1)
    g = gt.permute(0, 3, 1, 2).clamp(0, 1)
    return 1.0 - ssim_metric(p, g)


def train_gsplat(
    data_dir: Path,
    result_dir: Path,
    cfg: Optional[PhaseBTrainConfig] = None,
) -> Dict[str, Any]:
    """
    Train Gaussians on ``gs_dataset`` and write ``point_cloud.ply`` under result_dir.
    """
    from gsplat import export_splats
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    cfg = cfg or PhaseBTrainConfig()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Phase B gsplat train requires CUDA")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = ColmapTxtDataset(
        data_dir,
        data_factor=cfg.data_factor,
        max_images=cfg.max_images,
        max_points=cfg.max_points,
    )
    scene_scale = dataset.scene_scale * 1.1

    phase_a_init = None
    phase_a_path = _resolve_phase_a_ply(Path(data_dir), cfg)
    if phase_a_path is not None:
        try:
            phase_a_init = _read_phase_a_gaussian_init(phase_a_path, cfg.max_points)
        except Exception as exc:
            logger.warning("Phase A init skipped (%s): %s", phase_a_path, exc)

    splats, optimizers = _create_splats(
        dataset.points,
        dataset.points_rgb,
        scene_scale=scene_scale,
        sh_degree=cfg.sh_degree,
        init_opacity=cfg.init_opacity,
        means_lr=cfg.means_lr,
        device=cfg.device,
        phase_a=phase_a_init,
        train_x_flipped=dataset.train_x_flipped,
        init_scale_mult=cfg.init_scale_mult,
        scales_lr=cfg.scales_lr,
    )

    strategy = DefaultStrategy(
        refine_start_iter=cfg.refine_start_iter if cfg.enable_densify else max(cfg.max_steps + 1, 50_000),
        refine_stop_iter=cfg.refine_stop_iter if cfg.enable_densify else max(cfg.max_steps + 1, 50_000),
        refine_every=cfg.refine_every,
        reset_every=cfg.reset_every,
        prune_opa=0.02,
        verbose=True,
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    means_anchor = splats["means"].detach().clone()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )
    loader_iter = iter(loader)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(cfg.max_steps, 1))
    )
    ssim_metric = _make_ssim(cfg.device)

    (result_dir / "cfg.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )

    t0 = time.time()
    last_loss = float("nan")
    pbar = tqdm(range(cfg.max_steps), desc="gsplat Phase B")
    for step in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        camtoworlds = batch["camtoworld"].to(cfg.device)
        Ks = batch["K"].to(cfg.device)
        pixels = batch["image"].to(cfg.device).float() / 255.0
        height, width = pixels.shape[1:3]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors_sh,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            sh_degree=sh_degree_to_use,
            packed=cfg.packed,
            absgrad=False,
            rasterize_mode="classic",
        )
        colors = renders[..., :3]

        strategy.step_pre_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        l1 = F.l1_loss(colors, pixels)
        try:
            ssim_l = _ssim_loss(ssim_metric, colors, pixels)
        except Exception:
            ssim_l = torch.tensor(0.0, device=cfg.device)
        loss = l1 * (1.0 - cfg.ssim_lambda) + ssim_l * cfg.ssim_lambda
        if cfg.means_l2_lambda > 0 and step < cfg.means_l2_until:
            # Keep Gaussians near Phase A / SfM init (room AABB).
            n_anchor = min(means_anchor.shape[0], splats["means"].shape[0])
            loss = loss + cfg.means_l2_lambda * F.mse_loss(
                splats["means"][:n_anchor], means_anchor[:n_anchor]
            )
        loss.backward()
        last_loss = float(loss.item())

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()

        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=cfg.packed,
        )

        if step % 50 == 0 or step == cfg.max_steps - 1:
            pbar.set_postfix(
                loss=f"{last_loss:.4f}",
                n=len(splats["means"]),
                sh=sh_degree_to_use,
            )

    # Remap back to Studio X-mirrored frame if we trained right-handed.
    means_out = splats["means"].detach()
    quats_out = splats["quats"].detach()
    if dataset.train_x_flipped:
        means_np = _flip_x_points(means_out.cpu().numpy())
        quats_np = _flip_x_quats_wxyz(quats_out.cpu().numpy())
        means_out = torch.from_numpy(means_np).to(cfg.device)
        quats_out = torch.from_numpy(quats_np).to(cfg.device)
        # Re-seat floor at Y=0 after mirror (mirror preserves Y).
        y_min = float(means_out[:, 1].min().item())
        if abs(y_min) > 1e-4:
            means_out = means_out.clone()
            means_out[:, 1] -= y_min

    ply_path = result_dir / "point_cloud.ply"
    export_splats(
        means=means_out,
        scales=splats["scales"].detach(),
        quats=quats_out,
        opacities=splats["opacities"].detach(),
        sh0=splats["sh0"].detach(),
        shN=splats["shN"].detach(),
        format="ply",
        save_to=str(ply_path),
    )

    elapsed = time.time() - t0
    with torch.no_grad():
        opac_sig = torch.sigmoid(splats["opacities"].detach())
        opac_stats = {
            "opacity_mean": float(opac_sig.mean().item()),
            "opacity_frac_gt_0_5": float((opac_sig > 0.5).float().mean().item()),
        }
    info = {
        "ply": str(ply_path),
        "gaussian_count": int(means_out.shape[0]),
        "max_steps": cfg.max_steps,
        "data_factor": cfg.data_factor,
        "num_images": len(dataset),
        "elapsed_sec": round(elapsed, 1),
        "final_loss": last_loss,
        "scene_scale": scene_scale,
        "pose_source": dataset.pose_source,
        "train_x_flipped": dataset.train_x_flipped,
        "init_from_phase_a": phase_a_path is not None and phase_a_init is not None,
        "sh_degree": cfg.sh_degree,
        "data_factor": cfg.data_factor,
        "init_scale_mult": cfg.init_scale_mult,
        **opac_stats,
    }
    (result_dir / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Phase B train done: %s", info)
    return info


def _splats_from_served_ply(
    ply_path: Path,
    *,
    device: str,
) -> torch.nn.ParameterDict:
    """Load a Spark-layout served PLY into gsplat ParameterDict (studio frame)."""
    means, rgb, opacity, log_scales, quats, sh_dc = _read_phase_a_gaussian_init(
        ply_path, max_points=2_000_000
    )
    n = means.shape[0]
    sh0 = torch.from_numpy(sh_dc).float().unsqueeze(1)  # [N,1,3]
    shN = torch.zeros(n, 0, 3)
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(means).float(), requires_grad=False),
            "scales": torch.nn.Parameter(torch.from_numpy(log_scales).float(), requires_grad=False),
            "quats": torch.nn.Parameter(torch.from_numpy(quats).float(), requires_grad=False),
            "opacities": torch.nn.Parameter(torch.from_numpy(opacity).float(), requires_grad=False),
            "sh0": torch.nn.Parameter(sh0.float()),
            "shN": torch.nn.Parameter(shN.float(), requires_grad=False),
        }
    ).to(device)
    return splats


def refine_gaussian_colors_only(
    data_dir: Path,
    served_ply: Path,
    result_dir: Path,
    cfg: Optional[ColorRefineConfig] = None,
) -> Dict[str, Any]:
    """
    Freeze geometry; optimize SH DC colors against training photos.

    Use on a sanitized served ``environment.ply`` when structure is good but
  textures look like averaged point-cloud colors.
    """
    from gsplat import export_splats
    from gsplat.rendering import rasterization

    cfg = cfg or ColorRefineConfig()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Color refine requires CUDA")

    torch.manual_seed(cfg.seed)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = ColmapTxtDataset(
        data_dir,
        data_factor=cfg.data_factor,
        studio_frame=cfg.studio_frame,
    )
    splats = _splats_from_served_ply(served_ply, device=cfg.device)
    optimizer = torch.optim.Adam([{"params": splats["sh0"], "lr": cfg.sh_lr, "name": "sh0"}])
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.01 ** (1.0 / max(cfg.max_steps, 1))
    )
    ssim_metric = _make_ssim(cfg.device)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )
    loader_iter = iter(loader)
    (result_dir / "color_refine_cfg.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )

    t0 = time.time()
    last_loss = float("nan")
    pbar = tqdm(range(cfg.max_steps), desc="gsplat color refine")
    for step in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        camtoworlds = batch["camtoworld"].to(cfg.device)
        Ks = batch["K"].to(cfg.device)
        pixels = batch["image"].to(cfg.device).float() / 255.0
        height, width = pixels.shape[1:3]

        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        with torch.set_grad_enabled(True):
            renders, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors_sh,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=1e10,
                sh_degree=0,
                packed=False,
                absgrad=False,
                rasterize_mode="classic",
            )
            colors = renders[..., :3]
            l1 = F.l1_loss(colors, pixels)
            try:
                ssim_l = _ssim_loss(ssim_metric, colors, pixels)
            except Exception:
                ssim_l = torch.tensor(0.0, device=cfg.device)
            loss = l1 * (1.0 - cfg.ssim_lambda) + ssim_l * cfg.ssim_lambda
            loss.backward()
            last_loss = float(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sched.step()

        if step % 50 == 0 or step == cfg.max_steps - 1:
            pbar.set_postfix(loss=f"{last_loss:.4f}")

    ply_path = result_dir / "point_cloud.color_refined.ply"
    export_splats(
        means=splats["means"].detach(),
        scales=splats["scales"].detach(),
        quats=splats["quats"].detach(),
        opacities=splats["opacities"].detach(),
        sh0=splats["sh0"].detach(),
        shN=splats["shN"].detach(),
        format="ply",
        save_to=str(ply_path),
    )
    elapsed = time.time() - t0
    with torch.no_grad():
        sh = splats["sh0"].detach().squeeze(1).cpu().numpy()
        rgb = np.clip(0.5 + _SH_C0 * sh, 0.0, 1.0)
    info = {
        "ply": str(ply_path),
        "gaussian_count": int(splats["means"].shape[0]),
        "max_steps": cfg.max_steps,
        "data_factor": cfg.data_factor,
        "elapsed_sec": round(elapsed, 1),
        "final_loss": last_loss,
        "rgb_mean": rgb.mean(axis=0).tolist(),
        "studio_frame": cfg.studio_frame,
        "phase": "B_color_refined",
    }
    (result_dir / "color_refine_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Color refine done: %s", info)
    return info


def refine_colors_and_apply(
    world_dir: Path,
    *,
    cfg: Optional[ColorRefineConfig] = None,
    gs_dataset: Optional[Path] = None,
) -> Dict[str, Any]:
    """Color-only photometric refine on served structure; re-export for Spark."""
    world_dir = Path(world_dir)
    data_dir = Path(gs_dataset) if gs_dataset else world_dir / "gs_dataset"
    served = world_dir / "gs_train" / "point_cloud.sanitized.ply"
    if not served.is_file():
        served = world_dir / "environment.ply"
    result_dir = world_dir / "gs_train"
    refine_info = refine_gaussian_colors_only(data_dir, served, result_dir, cfg=cfg)
    # Verify colors actually moved before re-exporting to Spark.
    import re as _re

    raw_before = served.read_bytes()
    raw_after = Path(refine_info["ply"]).read_bytes()
    sep = b"end_header\n"
    for raw in (raw_before, raw_after):
        pass
    i0, i1 = raw_before.find(sep), raw_after.find(sep)
    p0 = [ln.split()[-1] for ln in raw_before[:i0].decode().splitlines() if ln.startswith("property float")]
    p1 = [ln.split()[-1] for ln in raw_after[:i1].decode().splitlines() if ln.startswith("property float")]
    n0 = int(_re.search(r"element vertex (\d+)", raw_before[:i0].decode()).group(1))
    n1 = int(_re.search(r"element vertex (\d+)", raw_after[:i1].decode()).group(1))
    a0 = np.frombuffer(raw_before[i0 + len(sep) :], dtype="<f4").reshape(n0, len(p0))
    a1 = np.frombuffer(raw_after[i1 + len(sep) :], dtype="<f4").reshape(n1, len(p1))
    fdc_delta = float(
        np.abs(
            a0[:, [p0.index("f_dc_0"), p0.index("f_dc_1"), p0.index("f_dc_2")]]
            - a1[:, [p1.index("f_dc_0"), p1.index("f_dc_1"), p1.index("f_dc_2")]]
        ).max()
    )
    refine_info["fdc_max_delta"] = fdc_delta
    if fdc_delta < 1e-6:
        logger.warning("Color refine produced no f_dc change (delta=%.2e)", fdc_delta)
    train_meta = {**refine_info, "sh_degree": 0, "pose_source": "poses_c2w_matrix", "phase": "B_color_refined"}
    apply_info = apply_trained_ply_to_world(
        world_dir,
        Path(refine_info["ply"]),
        train_meta=train_meta,
    )
    out = {**refine_info, **apply_info}
    out["world_directory"] = str(world_dir)
    return out


def to_spark_compatible_gaussian_ply(
    src_ply: Path,
    dst_ply: Path,
) -> Dict[str, Any]:
    """
    Rewrite a gsplat/Inria PLY into the Spark / TripoSplat / Phase A layout.

    Spark's packed path expects DC-only + normals (17 float props). PLYs with
    ``f_rest_*`` SH bands mis-stride and render as a scattered blob.
    """
    src_ply = Path(src_ply)
    dst_ply = Path(dst_ply)
    raw = src_ply.read_bytes()
    sep = b"end_header\n"
    idx = raw.find(sep)
    if idx < 0:
        raise RuntimeError(f"Bad PLY header: {src_ply}")
    header_txt = raw[: idx + len(sep)].decode("ascii", errors="replace")
    body = raw[idx + len(sep) :]
    props = [
        ln.split()[-1]
        for ln in header_txt.splitlines()
        if ln.startswith("property float")
    ]
    n = int(
        next(
            ln.split()[-1]
            for ln in header_txt.splitlines()
            if ln.startswith("element vertex")
        )
    )
    if not props or props[0:3] != ["x", "y", "z"]:
        raise RuntimeError(f"Unexpected PLY props in {src_ply}: {props[:8]}")
    arr = np.frombuffer(body, dtype="<f4")
    if arr.size != n * len(props):
        raise RuntimeError(f"PLY size mismatch in {src_ply}")
    verts = arr.reshape(n, len(props))
    pidx = {p: i for i, p in enumerate(props)}
    for req in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
        if req not in pidx:
            raise RuntimeError(f"Missing {req} in {src_ply}")

    out = np.zeros((n, 17), dtype=np.float32)
    out[:, 0:3] = verts[:, 0:3]
    # nx,ny,nz left 0
    out[:, 6] = verts[:, pidx["f_dc_0"]]
    out[:, 7] = verts[:, pidx["f_dc_1"]]
    out[:, 8] = verts[:, pidx["f_dc_2"]]
    out[:, 9] = verts[:, pidx["opacity"]]
    out[:, 10] = verts[:, pidx["scale_0"]]
    out[:, 11] = verts[:, pidx["scale_1"]]
    out[:, 12] = verts[:, pidx["scale_2"]]
    out[:, 13] = verts[:, pidx["rot_0"]]
    out[:, 14] = verts[:, pidx["rot_1"]]
    out[:, 15] = verts[:, pidx["rot_2"]]
    out[:, 16] = verts[:, pidx["rot_3"]]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    ).encode("ascii")
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    dst_ply.write_bytes(header + out.tobytes())
    return {
        "vertices": n,
        "dropped_f_rest": sum(1 for p in props if p.startswith("f_rest")),
        "path": str(dst_ply),
    }


def sanitize_gaussian_scales(
    src_ply: Path,
    dst_ply: Path,
    *,
    max_scale: float = 0.06,
    max_aniso: float = 6.0,
    drop_aniso: float = 24.0,
) -> Dict[str, Any]:
    """
    Clamp runaway Gaussian scales that Spark renders as black spikes/needles.

    Training without densify can still stretch a few covariances to meters-long
    axes (aniso ≫ 1000). Drop those and clamp the rest.
    """
    src_ply = Path(src_ply)
    dst_ply = Path(dst_ply)
    raw = src_ply.read_bytes()
    sep = b"end_header\n"
    idx = raw.find(sep)
    if idx < 0:
        raise RuntimeError(f"Bad PLY: {src_ply}")
    header_txt = raw[: idx + len(sep)].decode("ascii", errors="replace")
    n = int(
        next(ln.split()[-1] for ln in header_txt.splitlines() if ln.startswith("element vertex"))
    )
    props = [
        ln.split()[-1]
        for ln in header_txt.splitlines()
        if ln.startswith("property float")
    ]
    verts = np.frombuffer(raw[idx + len(sep) :], dtype="<f4").reshape(n, len(props)).copy()
    pidx = {p: i for i, p in enumerate(props)}
    for req in ("scale_0", "scale_1", "scale_2"):
        if req not in pidx:
            raise RuntimeError(f"Missing {req} in {src_ply}")

    sc = np.exp(verts[:, [pidx["scale_0"], pidx["scale_1"], pidx["scale_2"]]].astype(np.float64))
    aniso = sc.max(axis=1) / (sc.min(axis=1) + 1e-8)
    keep = (sc.max(axis=1) <= max_scale * 3.0) & (aniso <= drop_aniso)
    verts = verts[keep]
    sc = sc[keep]
    sc = np.clip(sc, 1e-4, max_scale)
    longest = sc.max(axis=1, keepdims=True)
    shortest = sc.min(axis=1, keepdims=True)
    too = (longest / (shortest + 1e-8)) > max_aniso
    sc = np.where(too, np.minimum(sc, shortest * max_aniso), sc)
    sc = np.clip(sc, 1e-4, max_scale)
    verts[:, pidx["scale_0"]] = np.log(sc[:, 0]).astype(np.float32)
    verts[:, pidx["scale_1"]] = np.log(sc[:, 1]).astype(np.float32)
    verts[:, pidx["scale_2"]] = np.log(sc[:, 2]).astype(np.float32)

    new_header_lines = []
    for ln in header_txt.splitlines(True):
        if ln.startswith("element vertex"):
            new_header_lines.append(f"element vertex {verts.shape[0]}\n")
        else:
            new_header_lines.append(ln)
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    dst_ply.write_bytes("".join(new_header_lines).encode("ascii") + verts.astype("<f4").tobytes())
    return {
        "kept": int(verts.shape[0]),
        "dropped": int(n - int(keep.sum())),
        "max_scale": max_scale,
        "max_aniso": max_aniso,
        "path": str(dst_ply),
    }


def solidify_spark_gaussian_ply(
    src_ply: Path,
    points_ply: Path,
    dst_ply: Path,
    *,
    gamma: float = 0.85,
    exposure: float = 1.5,
    # Sharp default footprint; sparse regions get larger adaptive scales.
    base_sigma: float = 0.009,
    spacing_cover: float = 0.55,
    max_sigma: float = 0.028,
    color_smooth_k: int = 12,
    densify_gap: float = 0.028,
    densify_max: int = 80_000,
    min_opacity: float = 0.97,
    max_nn_dist: float = 0.08,
) -> Dict[str, Any]:
    """
    Sharp + solid Spark export from Phase A / PC colors.

    - Spatially smooth PC colors (k-NN) to kill splotchy walls
    - Adaptive isotropic scales from local spacing (small in dense, larger in gaps)
    - Midpoint densify where neighbor gaps would punch holes (corners/walls)
    - High opacity + mild brightness lift
    """
    from scipy.spatial import cKDTree

    info = recolor_gaussian_ply_from_points(
        src_ply, points_ply, dst_ply, max_nn_dist=max_nn_dist
    )
    raw = Path(dst_ply).read_bytes()
    sep = b"end_header\n"
    idx = raw.find(sep)
    header_txt = raw[: idx + len(sep)].decode("ascii", errors="replace")
    props = [
        ln.split()[-1]
        for ln in header_txt.splitlines()
        if ln.startswith("property float")
    ]
    n = int(
        next(
            ln.split()[-1]
            for ln in header_txt.splitlines()
            if ln.startswith("element vertex")
        )
    )
    verts = np.frombuffer(raw[idx + len(sep) :], dtype="<f4").reshape(n, len(props)).copy()
    pidx = {p: i for i, p in enumerate(props)}
    means = verts[:, [pidx["x"], pidx["y"], pidx["z"]]].astype(np.float32)
    sh = verts[:, [pidx["f_dc_0"], pidx["f_dc_1"], pidx["f_dc_2"]]]
    rgb = np.clip(0.5 + _SH_C0 * sh, 0.0, 1.0).astype(np.float32)

    tree = cKDTree(means)
    k_sp = min(9, max(3, n - 1))
    d_sp, _ = tree.query(means, k=k_sp, workers=-1)
    spacing = d_sp[:, -1] if d_sp.ndim == 2 else np.full(n, float(base_sigma), dtype=np.float64)

    # Color smooth: average self + neighbors (reduces wall splotches).
    k_col = min(int(color_smooth_k) + 1, n)
    _, nn = tree.query(means, k=k_col, workers=-1)
    if nn.ndim == 1:
        nn = nn[:, None]
    rgb = rgb[nn].mean(axis=1).astype(np.float32)
    rgb = np.clip(np.power(np.clip(rgb, 0.0, 1.0), float(gamma)) * float(exposure), 0.0, 1.0)

    # Adaptive scales: dense → sharp; sparse/corners → cover gaps.
    sigma = np.maximum(float(base_sigma), spacing.astype(np.float64) * float(spacing_cover))
    sigma = np.clip(sigma, float(base_sigma) * 0.75, float(max_sigma))
    # Extra inflate near AABB edges (wall/ceiling corners punch holes first).
    mins = means.min(axis=0)
    maxs = means.max(axis=0)
    edge = (
        (means[:, 0] < mins[0] + 0.18)
        | (means[:, 0] > maxs[0] - 0.18)
        | (means[:, 2] < mins[2] + 0.18)
        | (means[:, 2] > maxs[2] - 0.18)
        | (means[:, 1] > maxs[1] - 0.30)
    )
    sigma = np.where(edge, np.minimum(sigma * 1.35, float(max_sigma)), sigma)
    log_s = np.log(sigma.astype(np.float32))
    verts[:, pidx["scale_0"]] = log_s
    verts[:, pidx["scale_1"]] = log_s
    verts[:, pidx["scale_2"]] = log_s

    op = np.full(n, float(min_opacity), dtype=np.float64)
    op = np.clip(op, 1e-4, 0.995)
    verts[:, pidx["opacity"]] = np.log(op / (1.0 - op)).astype(np.float32)

    sh = ((rgb - 0.5) / _SH_C0).astype(np.float32)
    verts[:, pidx["f_dc_0"]] = sh[:, 0]
    verts[:, pidx["f_dc_1"]] = sh[:, 1]
    verts[:, pidx["f_dc_2"]] = sh[:, 2]

    # Midpoint densify for large nearest-neighbor gaps.
    d1, i1 = tree.query(means, k=2, workers=-1)
    nn_dist = d1[:, 1]
    nn_idx = i1[:, 1]
    gap_mask = nn_dist > float(densify_gap)
    gap_ids = np.where(gap_mask)[0]
    added = 0
    if gap_ids.size and densify_max > 0:
        # Unique undirected edges (i < j)
        pairs = []
        for i in gap_ids:
            j = int(nn_idx[i])
            a, b = (int(i), j) if i < j else (j, int(i))
            pairs.append((a, b))
        pairs = list(dict.fromkeys(pairs))[: int(densify_max)]
        extras = []
        for a, b in pairs:
            mid = 0.5 * (means[a] + means[b])
            row = verts[a].copy()
            row[pidx["x"]], row[pidx["y"]], row[pidx["z"]] = mid
            # Color blend + slightly larger splat to seal the seam
            rgb_m = 0.5 * (rgb[a] + rgb[b])
            sh_m = ((rgb_m - 0.5) / _SH_C0).astype(np.float32)
            row[pidx["f_dc_0"]], row[pidx["f_dc_1"]], row[pidx["f_dc_2"]] = sh_m
            sig_m = float(min(max(nn_dist[a] * 0.5, float(base_sigma)), float(max_sigma)))
            row[pidx["scale_0"]] = row[pidx["scale_1"]] = row[pidx["scale_2"]] = math.log(
                sig_m
            )
            extras.append(row)
        if extras:
            verts = np.vstack([verts, np.stack(extras, axis=0)])
            added = len(extras)
            n = verts.shape[0]
            header_txt = re.sub(
                r"element vertex \d+",
                f"element vertex {n}",
                header_txt,
            )

    Path(dst_ply).parent.mkdir(parents=True, exist_ok=True)
    Path(dst_ply).write_bytes(header_txt.encode("ascii") + verts.astype("<f4").tobytes())
    sc = np.exp(verts[:, [pidx["scale_0"], pidx["scale_1"], pidx["scale_2"]]])
    op_out = 1.0 / (1.0 + np.exp(-verts[:, pidx["opacity"]]))
    info.update(
        {
            "path": str(dst_ply),
            "gamma": float(gamma),
            "exposure": float(exposure),
            "base_sigma": float(base_sigma),
            "spacing_cover": float(spacing_cover),
            "max_sigma": float(max_sigma),
            "color_smooth_k": int(color_smooth_k),
            "densify_added": int(added),
            "min_opacity": float(min_opacity),
            "gaussian_count": int(n),
            "sigma_median": float(np.median(sc)),
            "sigma_p95": float(np.percentile(sc, 95)),
            "opacity_mean": float(op_out.mean()),
            "rgb_mean": rgb.mean(axis=0).tolist() if added == 0 else None,
            "rgb_std": float(rgb.std()) if added == 0 else None,
            "phase": "B_pc_sharp_solid_v2",
        }
    )
    return info


def solidify_world_for_spark(
    world_dir: Path,
    *,
    src_ply: Optional[Path] = None,
    points_ply: Optional[Path] = None,
    max_scale: float = 0.08,
) -> Dict[str, Any]:
    """Sharp adaptive Phase-A coverage + smoothed bright PC colors for Spark."""
    world_dir = Path(world_dir)
    src = Path(src_ply) if src_ply else world_dir / "environment.phaseA.ply"
    if not src.is_file():
        src = world_dir / "gs_train" / "point_cloud.sanitized.ply"
    pts = Path(points_ply) if points_ply else world_dir / "environment.points.ply"
    out = world_dir / "gs_train" / "point_cloud.solid_bright.ply"
    info = solidify_spark_gaussian_ply(src, pts, out)
    san = world_dir / "gs_train" / "point_cloud.solid_bright_san.ply"
    san_info = sanitize_gaussian_scales(
        out, san, max_scale=max_scale, max_aniso=6.0, drop_aniso=32.0
    )
    apply_info = apply_trained_ply_to_world(
        world_dir,
        san,
        train_meta={
            **{k: v for k, v in info.items() if v is not None},
            **san_info,
            "sh_degree": 0,
            "pose_source": "point_cloud_nn",
            "phase": info.get("phase", "B_pc_sharp_solid_v2"),
            "max_steps": 0,
            "data_factor": 0,
            "elapsed_sec": 0.0,
            "final_loss": None,
            "max_scale": max_scale,
        },
        ref_xyz=None,  # keep densified cloud — no floater prune
    )
    return {**info, **san_info, **apply_info, "world_directory": str(world_dir)}


def recolor_gaussian_ply_from_points(
    gaussian_ply: Path,
    points_ply: Path,
    dst_ply: Path,
    *,
    max_nn_dist: float = 0.08,
) -> Dict[str, Any]:
    """
    Replace Gaussian ``f_dc`` with nearest point-cloud RGB (Spark SH0 encoding).

    Use when photometric color refine overfits misaligned LingBot poses and free
    views look wrong, while the sibling point-cloud world colors look correct.
    Geometry (means / scales / quats / opacity) is left unchanged.
    """
    from scipy.spatial import cKDTree

    from core.utils.lingbot_map_pipeline import _read_ply_xyzrgb_numpy

    gaussian_ply = Path(gaussian_ply)
    points_ply = Path(points_ply)
    dst_ply = Path(dst_ply)

    raw = gaussian_ply.read_bytes()
    sep = b"end_header\n"
    idx = raw.find(sep)
    if idx < 0:
        raise RuntimeError(f"Bad Gaussian PLY: {gaussian_ply}")
    header_txt = raw[: idx + len(sep)].decode("ascii", errors="replace")
    props = [
        ln.split()[-1]
        for ln in header_txt.splitlines()
        if ln.startswith("property float")
    ]
    n = int(
        next(
            ln.split()[-1]
            for ln in header_txt.splitlines()
            if ln.startswith("element vertex")
        )
    )
    verts = np.frombuffer(raw[idx + len(sep) :], dtype="<f4").reshape(n, len(props)).copy()
    pidx = {p: i for i, p in enumerate(props)}
    for req in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
        if req not in pidx:
            raise RuntimeError(f"Missing {req} in {gaussian_ply}")

    xyz_pc, rgb_pc = _read_ply_xyzrgb_numpy(points_ply)
    tree = cKDTree(xyz_pc.astype(np.float32))
    means = verts[:, [pidx["x"], pidx["y"], pidx["z"]]].astype(np.float32)
    dists, nn = tree.query(means, k=1, workers=-1)
    rgb = (rgb_pc[nn].astype(np.float32) / 255.0).clip(0.0, 1.0)
    # Keep previous color when NN is too far (orphan floaters).
    far = dists > float(max_nn_dist)
    if far.any():
        old_sh = verts[far][:, [pidx["f_dc_0"], pidx["f_dc_1"], pidx["f_dc_2"]]]
        old_rgb = np.clip(0.5 + _SH_C0 * old_sh, 0.0, 1.0)
        rgb[far] = old_rgb
    sh = ((rgb - 0.5) / _SH_C0).astype(np.float32)
    verts[:, pidx["f_dc_0"]] = sh[:, 0]
    verts[:, pidx["f_dc_1"]] = sh[:, 1]
    verts[:, pidx["f_dc_2"]] = sh[:, 2]

    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    dst_ply.write_bytes(header_txt.encode("ascii") + verts.astype("<f4").tobytes())
    return {
        "path": str(dst_ply),
        "gaussian_count": int(n),
        "point_count": int(xyz_pc.shape[0]),
        "nn_dist_mean": float(np.mean(dists)),
        "nn_dist_p90": float(np.percentile(dists, 90)),
        "far_kept_old_color": int(far.sum()),
        "rgb_mean": rgb.mean(axis=0).tolist(),
        "phase": "B_pc_recolored",
    }


def project_image_colors_onto_gaussians(
    means: np.ndarray,
    rgb_base: np.ndarray,
    data_dir: Path,
    *,
    data_factor: int = 2,
    studio_frame: bool = True,
    min_views: int = 2,
    max_views: int = 80,
    blend: float = 0.65,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Multi-view color projection with point-cloud prior.

    Only views that see a meaningful fraction of the cloud contribute. Each
    Gaussian gets a confidence-weighted average of projected pixels, blended
    with ``rgb_base`` (typically point-cloud NN colors).
    """
    data_dir = Path(data_dir)
    ds = ColmapTxtDataset(data_dir, data_factor=data_factor, studio_frame=studio_frame)
    n = means.shape[0]
    acc = np.zeros((n, 3), dtype=np.float64)
    wsum = np.zeros((n,), dtype=np.float64)

    # Rank views by how many Gaussians they see (skip misaligned cameras).
    view_scores: List[Tuple[int, int]] = []
    xyz = means.astype(np.float64)
    if ds.train_x_flipped:
        xyz = xyz.copy()
        xyz[:, 0] *= -1.0
    sample_idx = np.linspace(0, len(ds) - 1, min(len(ds), max_views * 3), dtype=int)
    for i in sample_idx:
        item = ds[int(i)]
        c2w = np.asarray(item["camtoworld"], dtype=np.float64)
        K = np.asarray(item["K"], dtype=np.float64)
        img = np.asarray(item["image"])
        h, w = img.shape[:2]
        w2c = np.linalg.inv(c2w)
        cam = (w2c[:3, :3] @ xyz.T + w2c[:3, 3:4]).T
        z = cam[:, 2]
        u = K[0, 0] * cam[:, 0] / np.clip(z, 1e-4, None) + K[0, 2]
        v = K[1, 1] * cam[:, 1] / np.clip(z, 1e-4, None) + K[1, 2]
        m = (z > 0.15) & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        view_scores.append((int(m.sum()), int(i)))

    view_scores.sort(reverse=True)
    used = 0
    for hits, i in view_scores:
        if hits < max(500, int(0.01 * n)):
            continue
        if used >= max_views:
            break
        item = ds[int(i)]
        c2w = np.asarray(item["camtoworld"], dtype=np.float64)
        K = np.asarray(item["K"], dtype=np.float64)
        img = np.asarray(item["image"]).astype(np.float32) / 255.0
        h, w = img.shape[:2]
        w2c = np.linalg.inv(c2w)
        cam = (w2c[:3, :3] @ xyz.T + w2c[:3, 3:4]).T
        z = cam[:, 2]
        u = K[0, 0] * cam[:, 0] / np.clip(z, 1e-4, None) + K[0, 2]
        v = K[1, 1] * cam[:, 1] / np.clip(z, 1e-4, None) + K[1, 2]
        m = (z > 0.15) & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        if not np.any(m):
            continue
        ui = np.clip(np.rint(u[m]).astype(np.int32), 0, w - 1)
        vi = np.clip(np.rint(v[m]).astype(np.int32), 0, h - 1)
        # Prefer nearer samples (less occlusion risk) — weight ∝ 1/z.
        wt = (1.0 / np.clip(z[m], 0.2, 20.0)).astype(np.float64)
        cols = img[vi, ui]
        idx = np.where(m)[0]
        acc[idx] += cols * wt[:, None]
        wsum[idx] += wt
        used += 1

    conf = wsum
    has = conf >= float(min_views) * 0.15  # soft: enough weight ≈ a couple of near views
    rgb = rgb_base.astype(np.float64).copy()
    proj = np.zeros_like(rgb)
    proj[has] = acc[has] / conf[has, None]
    # Confidence 0..1 from view weight.
    c01 = np.clip(conf / (conf[has].mean() + 1e-6) if has.any() else conf, 0.0, 1.0)
    alpha = float(blend) * c01
    rgb[has] = (1.0 - alpha[has, None]) * rgb[has] + alpha[has, None] * proj[has]
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    meta = {
        "views_used": used,
        "gaussians_projected": int(has.sum()),
        "proj_frac": float(has.mean()),
        "blend": float(blend),
        "data_factor": int(data_factor),
    }
    return rgb, meta


def recolor_world_from_point_cloud(
    world_dir: Path,
    *,
    gaussian_ply: Optional[Path] = None,
    points_ply: Optional[Path] = None,
    project_images: bool = True,
) -> Dict[str, Any]:
    """Recolor served Gaussians from point cloud (+ optional gated image projection)."""
    world_dir = Path(world_dir)
    src = Path(gaussian_ply) if gaussian_ply else world_dir / "environment.ply"
    if gaussian_ply is None:
        for cand in (
            world_dir / "gs_train" / "point_cloud.sanitized.ply",
            world_dir / "environment.ply",
        ):
            if cand.is_file():
                src = cand
                break
    pts = Path(points_ply) if points_ply else world_dir / "environment.points.ply"
    if not pts.is_file():
        raise FileNotFoundError(pts)
    out_ply = world_dir / "gs_train" / "point_cloud.pc_recolored.ply"
    info = recolor_gaussian_ply_from_points(src, pts, out_ply)

    proj_meta = None
    if project_images and (world_dir / "gs_dataset").is_dir():
        try:
            # Reload just-written PLY for means + base RGB.
            raw = out_ply.read_bytes()
            sep = b"end_header\n"
            idx = raw.find(sep)
            header_txt = raw[: idx + len(sep)].decode("ascii", errors="replace")
            props = [
                ln.split()[-1]
                for ln in header_txt.splitlines()
                if ln.startswith("property float")
            ]
            n = int(
                next(
                    ln.split()[-1]
                    for ln in header_txt.splitlines()
                    if ln.startswith("element vertex")
                )
            )
            verts = np.frombuffer(raw[idx + len(sep) :], dtype="<f4").reshape(n, len(props)).copy()
            pidx = {p: i for i, p in enumerate(props)}
            means = verts[:, [pidx["x"], pidx["y"], pidx["z"]]].astype(np.float32)
            sh = verts[:, [pidx["f_dc_0"], pidx["f_dc_1"], pidx["f_dc_2"]]]
            rgb = np.clip(0.5 + _SH_C0 * sh, 0.0, 1.0).astype(np.float32)
            rgb, proj_meta = project_image_colors_onto_gaussians(
                means,
                rgb,
                world_dir / "gs_dataset",
                data_factor=2,
                studio_frame=True,
            )
            sh = ((rgb - 0.5) / _SH_C0).astype(np.float32)
            verts[:, pidx["f_dc_0"]] = sh[:, 0]
            verts[:, pidx["f_dc_1"]] = sh[:, 1]
            verts[:, pidx["f_dc_2"]] = sh[:, 2]
            out_ply.write_bytes(header_txt.encode("ascii") + verts.astype("<f4").tobytes())
            info["rgb_mean"] = rgb.mean(axis=0).tolist()
            info["phase"] = "B_pc_image_recolored"
            info["projection"] = proj_meta
        except Exception as exc:
            logger.warning("Image projection skipped: %s", exc)

    apply_info = apply_trained_ply_to_world(
        world_dir,
        out_ply,
        train_meta={
            **{k: v for k, v in info.items() if k != "projection"},
            "projection": proj_meta,
            "sh_degree": 0,
            "pose_source": "point_cloud_nn",
            "phase": info.get("phase", "B_pc_recolored"),
            "max_steps": 0,
            "data_factor": 0,
            "elapsed_sec": 0.0,
            "final_loss": None,
        },
    )
    return {**info, **apply_info, "world_directory": str(world_dir)}


def bake_metric_scale_into_spark_ply(
    src_ply: Path,
    dst_ply: Path,
    *,
    scale_xyz: Sequence[float],
) -> Dict[str, Any]:
    """
    Bake non-uniform world scale into Gaussian means + covariances, so Spark can
    use parent scale [1,1,1] (non-uniform parent scale smears ellipsoids).
    """
    sx, sy, sz = [float(x) for x in scale_xyz]
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6 and abs(sz - 1.0) < 1e-6:
        if Path(src_ply).resolve() != Path(dst_ply).resolve():
            shutil.copy2(src_ply, dst_ply)
        return {"baked": False, "scale_xyz": [sx, sy, sz], "path": str(dst_ply)}

    raw = Path(src_ply).read_bytes()
    sep = b"end_header\n"
    idx = raw.find(sep)
    if idx < 0:
        raise RuntimeError(f"Bad PLY: {src_ply}")
    header_txt = raw[: idx + len(sep)].decode("ascii", errors="replace")
    n = int(
        next(ln.split()[-1] for ln in header_txt.splitlines() if ln.startswith("element vertex"))
    )
    props = [
        ln.split()[-1]
        for ln in header_txt.splitlines()
        if ln.startswith("property float")
    ]
    verts = np.frombuffer(raw[idx + len(sep) :], dtype="<f4").reshape(n, len(props)).copy()
    pidx = {p: i for i, p in enumerate(props)}

    S = np.array([sx, sy, sz], dtype=np.float64)
    verts[:, 0] *= np.float32(sx)
    verts[:, 1] *= np.float32(sy)
    verts[:, 2] *= np.float32(sz)

    # Σ' = diag(S) Σ diag(S); re-factor via SVD of A' = diag(S) R diag(s).
    quats = verts[:, [pidx["rot_0"], pidx["rot_1"], pidx["rot_2"], pidx["rot_3"]]]
    log_s = verts[:, [pidx["scale_0"], pidx["scale_1"], pidx["scale_2"]]].astype(np.float64)
    scales = np.exp(log_s)
    R = _quat_wxyz_to_rotmat_batch(quats)
    # A = R @ diag(s)  →  A2 = diag(S) @ A
    A = R * scales[:, None, :]
    A2 = A * S.reshape(1, 3, 1)

    # Batch SVD
    from scipy.spatial.transform import Rotation

    chunk = 8192
    new_quats = np.empty((n, 4), dtype=np.float32)
    new_log_s = np.empty((n, 3), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        u, sig, vh = np.linalg.svd(A2[start:end])
        # R_new = U @ Vh; ensure det=+1
        rnew = u @ vh
        dets = np.linalg.det(rnew)
        flip = dets < 0
        if np.any(flip):
            u[flip, :, -1] *= -1
            rnew = u @ vh
            sig = sig.copy()
            sig[flip, -1] *= -1
        q_xyzw = Rotation.from_matrix(rnew).as_quat()
        new_quats[start:end] = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
        new_log_s[start:end] = np.log(np.clip(np.abs(sig), 1e-6, 1e3)).astype(np.float32)

    verts[:, pidx["rot_0"]] = new_quats[:, 0]
    verts[:, pidx["rot_1"]] = new_quats[:, 1]
    verts[:, pidx["rot_2"]] = new_quats[:, 2]
    verts[:, pidx["rot_3"]] = new_quats[:, 3]
    verts[:, pidx["scale_0"]] = new_log_s[:, 0]
    verts[:, pidx["scale_1"]] = new_log_s[:, 1]
    verts[:, pidx["scale_2"]] = new_log_s[:, 2]

    # Re-seat Y=0 after scale
    y_min = float(verts[:, 1].min())
    if abs(y_min) > 1e-4:
        verts[:, 1] -= y_min

    Path(dst_ply).parent.mkdir(parents=True, exist_ok=True)
    Path(dst_ply).write_bytes(raw[: idx + len(sep)] + verts.astype("<f4", copy=False).tobytes())
    return {
        "baked": True,
        "scale_xyz": [sx, sy, sz],
        "vertices": n,
        "path": str(dst_ply),
        "bbox_min": verts[:, 0:3].min(0).tolist(),
        "bbox_max": verts[:, 0:3].max(0).tolist(),
    }


def prune_gaussian_ply_to_ref_bounds(
    src_ply: Path,
    dst_ply: Path,
    *,
    ref_xyz: np.ndarray,
    margin: float = 0.35,
    min_opacity_logit: float = -5.0,
) -> Dict[str, Any]:
    """
    Drop densify floaters outside the gravity-aligned ref cloud AABB.

    Preserves locked LingBot seating (does not re-center). Spark-compatible binary PLY.
    """
    src_ply = Path(src_ply)
    dst_ply = Path(dst_ply)
    raw = src_ply.read_bytes()
    sep = b"end_header\n"
    idx = raw.find(sep)
    if idx < 0:
        raise RuntimeError(f"Bad PLY header: {src_ply}")
    header = raw[: idx + len(sep)]
    body = raw[idx + len(sep) :]
    header_txt = header.decode("ascii", errors="replace")
    props = [
        ln.split()[-1]
        for ln in header_txt.splitlines()
        if ln.startswith("property float")
    ]
    if len(props) < 7 or props[0:3] != ["x", "y", "z"]:
        raise RuntimeError(f"Unexpected Gaussian PLY layout in {src_ply}")
    n = int(
        next(ln.split()[-1] for ln in header_txt.splitlines() if ln.startswith("element vertex"))
    )
    arr = np.frombuffer(body, dtype="<f4")
    if arr.size != n * len(props):
        raise RuntimeError(
            f"PLY size mismatch: {arr.size} floats vs {n}*{len(props)} in {src_ply}"
        )
    verts = arr.reshape(n, len(props))
    ref = np.asarray(ref_xyz, dtype=np.float64).reshape(-1, 3)
    lo = ref.min(axis=0) - margin
    hi = ref.max(axis=0) + margin
    # Prefer Phase A floor: keep Y >= -margin around ref min Y
    lo[1] = min(lo[1], float(ref[:, 1].min()) - 0.05)

    prop_i = {p: i for i, p in enumerate(props)}
    opac_i = prop_i.get("opacity")
    mask = np.all((verts[:, 0:3] >= lo) & (verts[:, 0:3] <= hi), axis=1)
    if opac_i is not None:
        mask &= verts[:, opac_i] > min_opacity_logit
    kept = verts[mask]
    if kept.shape[0] < 1000:
        raise RuntimeError(
            f"Prune too aggressive: kept {kept.shape[0]} / {n} (margin={margin})"
        )

    # Re-seat on Y=0 (locked LingBot gravity pipeline ends with floor at Y=0).
    y_min = float(kept[:, 1].min())
    if abs(y_min) > 1e-4:
        kept = kept.copy()
        kept[:, 1] -= y_min

    # Rewrite header with new count
    new_header_lines = []
    for ln in header_txt.splitlines(True):
        if ln.startswith("element vertex"):
            new_header_lines.append(f"element vertex {kept.shape[0]}\n")
        else:
            new_header_lines.append(ln)
    new_header = "".join(new_header_lines).encode("ascii")
    if not new_header.endswith(b"\n"):
        new_header += b"\n"
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    dst_ply.write_bytes(new_header + kept.astype("<f4", copy=False).tobytes())
    y_ext = float(kept[:, 1].max() - kept[:, 1].min())
    return {
        "kept": int(kept.shape[0]),
        "dropped": int(n - kept.shape[0]),
        "y_extent": y_ext,
        "bbox_min": kept[:, 0:3].min(axis=0).tolist(),
        "bbox_max": kept[:, 0:3].max(axis=0).tolist(),
    }


def apply_trained_ply_to_world(
    world_dir: Path,
    trained_ply: Path,
    *,
    train_meta: Optional[Dict[str, Any]] = None,
    ref_xyz: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Replace ``environment.ply`` with trained splat; keep points backup + metric scale."""
    world_dir = Path(world_dir)
    trained_ply = Path(trained_ply)
    if not trained_ply.is_file():
        raise FileNotFoundError(trained_ply)

    env_ply = world_dir / "environment.ply"
    phase_a_backup = world_dir / "environment.phaseA.ply"
    if env_ply.is_file() and not phase_a_backup.exists():
        # Keep Phase A isotropic Gaussians for A/B comparison.
        shutil.copy2(env_ply, phase_a_backup)

    # Resolve ref cloud for floater prune (locked gravity AABB).
    if ref_xyz is None:
        points_backup = world_dir / "environment.points.ply"
        phase_a = world_dir / "environment.phaseA.ply"
        try:
            from core.utils.lingbot_map_pipeline import _read_ply_xyzrgb_numpy

            if points_backup.is_file():
                ref_xyz, _ = _read_ply_xyzrgb_numpy(points_backup)
            elif phase_a.is_file():
                # Phase A is Gaussian — use xyz only via binary parse fallback below
                ref_xyz = None
        except Exception:
            ref_xyz = None
        if ref_xyz is None and (world_dir / "gs_dataset" / "sparse" / "0" / "points3D.txt").is_file():
            ref_xyz, _ = _parse_colmap_points3d(
                world_dir / "gs_dataset" / "sparse" / "0" / "points3D.txt",
                500_000,
            )

    ply_to_apply = trained_ply
    prune_info = None
    spark_info = None

    # Always rewrite to Spark/TripoSplat DC+normals layout (f_rest → blob in Spark).
    spark_ply = world_dir / "gs_train" / "point_cloud.spark.ply"
    spark_info = to_spark_compatible_gaussian_ply(trained_ply, spark_ply)
    ply_to_apply = spark_ply
    logger.info("Phase B Spark rewrite: %s", spark_info)

    if ref_xyz is not None:
        pruned = world_dir / "gs_train" / "point_cloud.pruned.ply"
        prune_info = prune_gaussian_ply_to_ref_bounds(
            ply_to_apply,
            pruned,
            ref_xyz=ref_xyz,
            margin=0.4,
            min_opacity_logit=-2.2,  # ~0.1 sigmoid — drop translucent fog
        )
        ply_to_apply = pruned
        logger.info("Phase B floater prune: %s", prune_info)

    sanitize_info = sanitize_gaussian_scales(
        ply_to_apply, world_dir / "gs_train" / "point_cloud.sanitized.ply"
    )
    ply_to_apply = Path(sanitize_info["path"])
    logger.info("Phase B scale sanitize: %s", sanitize_info)

    bake_info = None
    man_path = world_dir / "world.manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    env_xform = dict((manifest.get("environment") or {}).get("transform") or {})
    scale = env_xform.get("scale")
    if scale is None and (world_dir / "metric_calibration.json").is_file():
        try:
            metric = json.loads((world_dir / "metric_calibration.json").read_text(encoding="utf-8"))
            scale = metric.get("scale_xyz") or metric.get("scale")
            if isinstance(scale, (int, float)):
                scale = [float(scale), 1.0, float(scale)]
        except Exception:
            scale = None
    # Metric bake is opt-in only — default off (SVD bake created Spark spikes).
    do_bake = bool((train_meta or {}).get("bake_metric_scale"))
    if do_bake and scale is not None and len(scale) >= 3:
        baked = world_dir / "gs_train" / "point_cloud.metric.ply"
        bake_info = bake_metric_scale_into_spark_ply(
            ply_to_apply, baked, scale_xyz=scale[:3]
        )
        if bake_info.get("baked"):
            ply_to_apply = baked
            logger.info("Phase B metric bake: %s", bake_info)

    shutil.copy2(ply_to_apply, env_ply)

    env = dict(manifest.get("environment") or {})
    env["type"] = "gaussian_splat"
    env["format"] = "ply"
    env["renderer"] = "spark"
    env["url"] = "environment.ply"
    if bake_info and bake_info.get("baked"):
        env["transform"] = {"scale": [1.0, 1.0, 1.0]}
    elif scale is not None and len(scale) >= 3:
        env["transform"] = {"scale": [float(scale[0]), float(scale[1]), float(scale[2])]}
    manifest["environment"] = env
    meta = dict(manifest.get("metadata") or {})
    meta["source_geometry"] = "gaussian_from_point_cloud"
    meta["gaussian_phase"] = str(
        (train_meta or {}).get("phase")
        or (train_meta or {}).get("gaussian_phase")
        or "B_gsplat_trained"
    )
    meta["gaussian_train_source"] = "gsplat"
    meta["gsplat_sanitize"] = sanitize_info
    meta.pop("metric_scale_baked_into_ply", None)
    meta.pop("metric_scale_xyz_baked", None)
    if train_meta:
        meta["gaussian_count"] = sanitize_info.get("kept") or (
            prune_info["kept"] if prune_info else train_meta.get("gaussian_count")
        )
        meta["gsplat_train"] = {
            k: train_meta[k]
            for k in (
                "max_steps",
                "data_factor",
                "num_images",
                "elapsed_sec",
                "final_loss",
                "pose_source",
                "train_x_flipped",
                "init_from_phase_a",
                "opacity_mean",
                "opacity_frac_gt_0_5",
                "sh_degree",
                "fdc_max_delta",
                "studio_frame",
            )
            if k in train_meta
        }
        if bake_info:
            meta["gsplat_metric_bake"] = bake_info
        else:
            meta.pop("gsplat_metric_bake", None)
        meta.pop("phase_b_status", None)
        meta.pop("phase_b_diagnosis", None)
        meta.pop("phase_b_note", None)
        if prune_info:
            meta["gsplat_prune"] = prune_info
        if spark_info:
            meta["gsplat_spark_export"] = {
                "dropped_f_rest": spark_info.get("dropped_f_rest"),
                "layout": "triposplat_dc_normals",
            }
    meta["gs_dataset"] = meta.get("gs_dataset") or "gs_dataset"
    meta["phase_a_backup"] = "environment.phaseA.ply"
    if (world_dir / "environment.points.ply").is_file():
        meta["point_cloud_backup"] = "environment.points.ply"
    manifest["metadata"] = meta
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "environment_ply": str(env_ply),
        "phase": meta["gaussian_phase"],
        "gaussian_count": meta.get("gaussian_count"),
        "prune": prune_info,
        "sanitize": sanitize_info,
        "metric_bake": bake_info,
        "manifest": str(man_path),
    }


def train_and_apply_phase_b(
    world_dir: Path,
    *,
    cfg: Optional[PhaseBTrainConfig] = None,
    gs_dataset: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full Phase B for a world package: train on ``gs_dataset/``, replace environment.ply.
    """
    world_dir = Path(world_dir)
    data_dir = Path(gs_dataset) if gs_dataset else world_dir / "gs_dataset"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {data_dir} — run Phase A with camera export first "
            "(refine_to_3dgs / refine_env_scan_to_3dgs.py)"
        )

    result_dir = world_dir / "gs_train"
    train_info = train_gsplat(data_dir, result_dir, cfg=cfg)
    apply_info = apply_trained_ply_to_world(
        world_dir,
        Path(train_info["ply"]),
        train_meta=train_info,
    )
    out = {**train_info, **apply_info}
    out["world_directory"] = str(world_dir)
    return out


def gsplat_available() -> bool:
    try:
        import gsplat  # noqa: F401
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
