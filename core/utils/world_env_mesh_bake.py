"""
Bake a world-package Gaussian environment into an OMB-ready GLB mesh.

Commercial-safe stack (DGX aarch64):
  - gsplat (Apache-2.0) depth/RGB render from posed ``gs_dataset/``
  - NumPy volumetric TSDF + scikit-image marching cubes (BSD) —
    Open3D wheels are unavailable on aarch64; algorithm matches Open3D TSDF fusion
  - trimesh quadric decimation (MIT) → ``environment_mesh.glb``

Requires multi-view cameras (LingBot Phase A ``gs_dataset/``). TripoSplat
image-to-world worlds have no cameras and must not call this path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Studio / OMB defaults. Photo quality uses denser TSDF + vertex COLOR_0
# (face-island atlas is OMB-safe but reads as flat low-poly in the viewport).
DEFAULT_TARGET_FACES = 150_000
DEFAULT_VOXEL_RES = 256
DEFAULT_MAX_VIEWS = 72

# Photo preset — denser geometry + more photo projection views.
PHOTO_TARGET_FACES = 300_000
PHOTO_VOXEL_RES = 320
PHOTO_MAX_VIEWS = 96
PHOTO_DATA_FACTOR = 2


@dataclass
class EnvMeshBakeConfig:
    target_face_count: int = DEFAULT_TARGET_FACES
    voxel_resolution: int = DEFAULT_VOXEL_RES
    max_views: int = DEFAULT_MAX_VIEWS
    data_factor: int = 2
    depth_trunc: Optional[float] = None  # auto from scene radius if None
    sdf_trunc_voxels: float = 4.0
    device: str = "cuda"
    write_collider: bool = True  # collider.glb = same mesh (walk / physics)
    # "vertex" = COLOR_0 (smooth studio viewport); "atlas" = OMB face-island PBR
    color_export: str = "vertex"
    # Soft gain so dark XR walk frames still texture the mesh.
    photo_exposure_normalize: bool = True
    quality: str = "balanced"  # draft | balanced | photo
    # auto: photo → camera_tsdf + gaussian_nn colors; draft/balanced → camera_tsdf + photo project
    geometry_mode: str = "auto"
    # photo_project = projective RGB (pose-sensitive); gaussian_nn = NN splat RGB (aligned);
    # tsdf = colors accumulated during fusion
    color_source: str = "auto"

    @classmethod
    def from_quality(cls, quality: str = "balanced", **overrides: Any) -> "EnvMeshBakeConfig":
        q = str(quality or "balanced").strip().lower()
        if q in ("photo", "high", "hq"):
            base = cls(
                target_face_count=PHOTO_TARGET_FACES,
                voxel_resolution=PHOTO_VOXEL_RES,
                max_views=PHOTO_MAX_VIEWS,
                data_factor=PHOTO_DATA_FACTOR,
                color_export="vertex",
                photo_exposure_normalize=True,
                quality="photo",
                # Thin crust around splat centers + NN colors (aligned).
                # Camera TSDF fills a solid blob when gsplat depths are thick.
                geometry_mode="means_mc",
                color_source="gaussian_nn",
            )
        elif q in ("draft", "fast", "low"):
            base = cls(
                target_face_count=80_000,
                voxel_resolution=160,
                max_views=32,
                data_factor=4,
                color_export="atlas",
                quality="draft",
                geometry_mode="camera_tsdf",
                color_source="photo_project",
            )
        else:
            base = cls(
                quality="balanced",
                geometry_mode="camera_tsdf",
                color_source="photo_project",
            )
        for key, val in overrides.items():
            if hasattr(base, key) and val is not None:
                setattr(base, key, val)
        return base


class WorldBakeError(ValueError):
    """Raised when a world cannot be baked (e.g. TripoSplat I2W without cameras)."""


def _normalize_photo_exposure(rgbs: List[np.ndarray], *, target_mean: float = 0.42) -> List[np.ndarray]:
    """Lift dark walk-scan frames toward a mid-grey mean without clipping highlights hard."""
    out: List[np.ndarray] = []
    for img in rgbs:
        arr = np.asarray(img, dtype=np.float32)
        mean = float(arr.mean())
        if mean < 1e-4:
            out.append(np.clip(arr, 0.0, 1.0))
            continue
        gain = min(3.0, max(1.0, target_mean / mean))
        if gain <= 1.05:
            out.append(np.clip(arr, 0.0, 1.0))
            continue
        lifted = arr * gain
        # Soft shoulder so bright lamps don't blow out.
        out.append(np.clip(lifted / (1.0 + 0.35 * np.maximum(lifted - 0.85, 0.0)), 0.0, 1.0))
    return out

def world_has_bake_cameras(world_dir: Path) -> bool:
    """True when multi-view poses + images exist for TSDF bake."""
    root = Path(world_dir)
    gs = root / "gs_dataset"
    if not gs.is_dir():
        return False
    images = gs / "images"
    if not images.is_dir():
        return False
    if not any(images.iterdir()):
        return False
    if (gs / "poses_c2w.npy").is_file():
        return True
    if (root / "cameras_aligned.npz").is_file() or (root / "cameras.npz").is_file():
        return True
    sparse = gs / "sparse" / "0"
    if not sparse.is_dir():
        sparse = gs / "sparse"
    return (sparse / "images.txt").is_file() and (sparse / "cameras.txt").is_file()


def resolve_environment_ply(world_dir: Path) -> Path:
    root = Path(world_dir)
    candidates = [
        root / "environment.ply",
        root / "gs_train" / "point_cloud.ply",
        root / "environment.points.ply",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 64:
            return p
    raise WorldBakeError(f"No environment Gaussian/point PLY under {root}")


def assert_bakeable_world(world_dir: Path) -> None:
    root = Path(world_dir)
    if not root.is_dir():
        raise WorldBakeError(f"World not found: {root}")
    manifest_path = root / "world.manifest.json"
    pipeline = ""
    if manifest_path.is_file():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            pipeline = str((meta.get("metadata") or {}).get("pipeline") or "")
        except Exception:
            pipeline = ""
    if "image-to-world" in pipeline.lower() and not world_has_bake_cameras(root):
        raise WorldBakeError(
            "Image-to-world (TripoSplat) has no multi-view cameras for TSDF bake. "
            "Publish TRELLIS props via World Library RP1, or run LingBot environment-scan "
            "/ a multi-view world path that exports gs_dataset/."
        )
    if not world_has_bake_cameras(root):
        raise WorldBakeError(
            "Missing gs_dataset/ cameras+images — run Phase A refine_to_3dgs "
            "(LingBot env-scan) before bake-env-mesh."
        )
    resolve_environment_ply(root)


def _load_gaussians_torch(ply_path: Path, device: str):
    """Load Spark/Phase-A layout PLY into gsplat tensors."""
    import torch
    import torch.nn.functional as F

    from core.utils.lingbot_3dgs_train import _read_phase_a_gaussian_init

    means_np, _rgb, opacity_logit, log_scales, quats, sh_dc = _read_phase_a_gaussian_init(
        ply_path, max_points=2_000_000
    )
    means = torch.from_numpy(means_np).float().to(device)
    quats_t = F.normalize(torch.from_numpy(quats).float().to(device), p=2, dim=-1)
    scales = torch.exp(torch.from_numpy(log_scales).float().to(device))
    opacities = torch.sigmoid(torch.from_numpy(opacity_logit).float().to(device))
    sh0 = torch.from_numpy(sh_dc).float().to(device).unsqueeze(1)  # [N,1,3]
    colors = sh0  # SH0 only
    return means, quats_t, scales, opacities, colors


def _mesh_from_gaussian_means(
    means_np: np.ndarray,
    rgb_np: np.ndarray,
    *,
    voxel_resolution: int = 256,
    pad: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mesh a thin crust around Gaussian *centers* (not camera TSDF).

    Camera-depth TSDF fills the room as a solid when gsplat depths are thick/
    noisy. A distance-to-means level set stays glued to the same points Spark
    shows; colors come from nearest Gaussian RGB.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    from skimage import measure

    xyz = np.asarray(means_np, dtype=np.float64)
    rgb = np.asarray(rgb_np, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[0] != xyz.shape[0]:
        raise WorldBakeError("rgb must match means for gaussian-means mesh")
    if rgb.max() > 1.5:
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = np.maximum(hi - lo, 1e-3)
    half = 0.5 * float(np.max(extent)) * float(pad)
    origin = center - half
    res = int(max(64, voxel_resolution))
    voxel = (2.0 * half) / float(res)

    # Query a coarse grid against means — thin shell, not filled solid.
    # Subsample means for speed if huge.
    if xyz.shape[0] > 400_000:
        rng = np.random.default_rng(0)
        xyz_q = xyz[rng.choice(xyz.shape[0], size=400_000, replace=False)]
    else:
        xyz_q = xyz
    tree = cKDTree(xyz_q)

    zs = origin[2] + (np.arange(res) + 0.5) * voxel
    ys = origin[1] + (np.arange(res) + 0.5) * voxel
    xs = origin[0] + (np.arange(res) + 0.5) * voxel
    # Build grid in (z,y,x) to match skimage marching_cubes axis order
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # Chunked KD query to limit peak RAM
    dist = np.empty(grid.shape[0], dtype=np.float64)
    chunk = 2_000_000
    for i0 in range(0, grid.shape[0], chunk):
        i1 = min(i0 + chunk, grid.shape[0])
        d, _ = tree.query(grid[i0:i1], k=1, workers=-1)
        dist[i0:i1] = d
    dist_vol = dist.reshape((res, res, res))
    # Soften slightly then extract isosurface at ~1.5 voxels from points
    dist_vol = gaussian_filter(dist_vol.astype(np.float32), sigma=0.6)
    level = float(1.5 * voxel)
    if not (np.any(dist_vol < level) and np.any(dist_vol > level)):
        raise WorldBakeError("gaussian-means distance field empty — cannot march cubes")

    verts, faces, _normals, _vals = measure.marching_cubes(
        dist_vol, level=level, spacing=(voxel, voxel, voxel)
    )
    vz, vy, vx = verts[:, 0], verts[:, 1], verts[:, 2]
    world = np.stack([vx + origin[0], vy + origin[1], vz + origin[2]], axis=1)

    # Drop tiny floater components
    try:
        import trimesh

        tmp = trimesh.Trimesh(vertices=world, faces=faces, process=False)
        comps = trimesh.graph.connected_components(tmp.face_adjacency, min_len=1)
        if comps:
            comps = sorted(comps, key=len, reverse=True)
            keep_n = max(1, sum(1 for c in comps if len(c) >= 0.02 * len(faces)))
            keep_faces = np.concatenate(
                [np.asarray(c, dtype=np.int64) for c in comps[:keep_n]]
            )
            tmp.update_faces(keep_faces)
            tmp.remove_unreferenced_vertices()
            world = np.asarray(tmp.vertices, dtype=np.float64)
            faces = np.asarray(tmp.faces, dtype=np.int64)
    except Exception as drop_exc:
        logger.debug("means_mc component prune skipped: %s", drop_exc)

    color_tree = cKDTree(xyz)
    _, nn = color_tree.query(world, k=1)
    colors = rgb[nn]
    return world.astype(np.float64), faces.astype(np.int64), colors.astype(np.float64)


def _colors_from_gaussian_nn(
    verts: np.ndarray,
    means_np: np.ndarray,
    rgb_np: np.ndarray,
) -> np.ndarray:
    """Nearest-Gaussian RGB for mesh vertices (pose-independent, matches Spark)."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(means_np, dtype=np.float64)
    rgb = np.asarray(rgb_np, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    tree = cKDTree(xyz)
    _, nn = tree.query(np.asarray(verts, dtype=np.float64), k=1)
    return rgb[nn].astype(np.float64)


def _prune_small_components(mesh: "trimesh.Trimesh", *, min_face_frac: float = 0.015) -> Dict[str, Any]:
    """
    Keep connected face components that are ≥ min_face_frac of total faces,
    always retaining at least the largest component. Mutates ``mesh``.
    """
    import trimesh

    n_before = int(len(mesh.faces))
    if n_before < 32:
        return {"before_faces": n_before, "after_faces": n_before, "components_kept": 1}
    try:
        comps = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1)
    except Exception:
        return {"before_faces": n_before, "after_faces": n_before, "components_kept": None}
    if not comps:
        return {"before_faces": n_before, "after_faces": n_before, "components_kept": 0}
    comps = sorted(comps, key=len, reverse=True)
    thresh = max(64, int(min_face_frac * n_before))
    keep = [c for c in comps if len(c) >= thresh]
    if not keep:
        keep = [comps[0]]
    keep_faces = np.concatenate([np.asarray(c, dtype=np.int64) for c in keep])
    mesh.update_faces(keep_faces)
    mesh.remove_unreferenced_vertices()
    return {
        "before_faces": n_before,
        "after_faces": int(len(mesh.faces)),
        "components_total": len(comps),
        "components_kept": len(keep),
        "min_faces": thresh,
    }


def _estimate_bounds(
    means_np: np.ndarray, camtoworlds: List[np.ndarray], percentile: float = 98.0
) -> Tuple[np.ndarray, float]:
    xyz = np.asarray(means_np, dtype=np.float64)
    center = np.median(xyz, axis=0)
    radius = float(np.percentile(np.linalg.norm(xyz - center, axis=-1), percentile))
    cams = np.stack([c[:3, 3] for c in camtoworlds], axis=0)
    cam_r = float(np.max(np.linalg.norm(cams - center, axis=-1)))
    radius = max(radius, cam_r * 0.35, 0.5) * 1.15
    return center.astype(np.float64), radius


def _tsdf_fuse(
    depths: List[np.ndarray],
    rgbs: List[np.ndarray],
    camtoworlds: List[np.ndarray],
    Ks: List[np.ndarray],
    *,
    center: np.ndarray,
    radius: float,
    voxel_resolution: int,
    depth_trunc: float,
    sdf_trunc: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Volumetric TSDF fusion (Open3D-style) without Open3D.

    Returns ``(verts, faces, vert_colors)`` in world coordinates.
    """
    from skimage import measure

    res = int(max(32, voxel_resolution))
    origin = np.asarray(center, dtype=np.float64).reshape(3) - float(radius)
    voxel_size = (2.0 * radius) / float(res)
    volume = np.ones((res, res, res), dtype=np.float32)  # SDF init far
    weight = np.zeros((res, res, res), dtype=np.float32)
    color_acc = np.zeros((res, res, res, 3), dtype=np.float32)

    # Voxel centers in world space (per-axis 1D grids — do not add origin(3,) to arange)
    axis = (np.arange(res, dtype=np.float64) + 0.5) * voxel_size
    xs = axis + origin[0]
    ys = axis + origin[1]
    zs = axis + origin[2]
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    # Note: meshgrid indexing ij → (z,y,x) axes of volume

    for depth, rgb, c2w, K in zip(depths, rgbs, camtoworlds, Ks):
        H, W = depth.shape[:2]
        c2w = np.asarray(c2w, dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        K = np.asarray(K, dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # World → camera for all voxels (subsample stride for speed on large grids)
        ones = np.ones_like(xx)
        pts = np.stack([xx, yy, zz, ones], axis=-1).reshape(-1, 4)
        cam = (pts @ w2c.T)[:, :3]
        z = cam[:, 2]
        valid_z = z > 1e-4
        u = fx * (cam[:, 0] / np.clip(z, 1e-4, None)) + cx
        v = fy * (cam[:, 1] / np.clip(z, 1e-4, None)) + cy
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        in_img = (
            valid_z
            & (ui >= 0)
            & (ui < W)
            & (vi >= 0)
            & (vi < H)
        )
        if not np.any(in_img):
            continue
        idx = np.flatnonzero(in_img)
        ui_v = ui[idx]
        vi_v = vi[idx]
        z_v = z[idx]
        d = depth[vi_v, ui_v]
        valid_d = (d > 1e-4) & (d < depth_trunc) & np.isfinite(d)
        if not np.any(valid_d):
            continue
        idx2 = idx[valid_d]
        ui2 = ui_v[valid_d]
        vi2 = vi_v[valid_d]
        z2 = z_v[valid_d]
        d2 = d[valid_d]
        sdf = d2 - z2
        mask = np.abs(sdf) < sdf_trunc
        if not np.any(mask):
            continue
        idx3 = idx2[mask]
        sdf3 = np.clip(sdf[mask] / sdf_trunc, -1.0, 1.0).astype(np.float32)
        # volume index from flat idx (z major via ij meshgrid: flat = z*res*res + y*res + x)
        zi = idx3 // (res * res)
        rem = idx3 % (res * res)
        yi = rem // res
        xi = rem % res
        w_old = weight[zi, yi, xi]
        w_new = w_old + 1.0
        volume[zi, yi, xi] = (volume[zi, yi, xi] * w_old + sdf3) / w_new
        cols = rgb[vi2[mask], ui2[mask]]
        color_acc[zi, yi, xi] = (
            color_acc[zi, yi, xi] * w_old[:, None] + cols
        ) / w_new[:, None]
        weight[zi, yi, xi] = w_new

    occupied = weight > 0
    if not np.any(occupied):
        raise WorldBakeError("TSDF volume empty — check cameras / Gaussian PLY alignment")

    # Marching cubes on SDF (zero crossing). Unobserved stays +1 (outside).
    try:
        verts_v, faces, *_ = measure.marching_cubes(
            volume, level=0.0, spacing=(voxel_size, voxel_size, voxel_size)
        )
    except Exception as e:
        raise WorldBakeError(f"Marching cubes failed: {e}") from e

    # skimage returns (z,y,x) order in verts relative to volume origin
    verts = verts_v[:, [2, 1, 0]] + origin.reshape(1, 3)

    # Sample vertex colors from color volume (trilinear-ish: nearest voxel)
    # verts_v from skimage are (z,y,x)*spacing; we converted to world xyz.
    idx_xyz = (verts - origin.reshape(1, 3)) / voxel_size
    vi = np.clip(np.round(idx_xyz).astype(np.int32), 0, res - 1)
    vert_colors = color_acc[vi[:, 2], vi[:, 1], vi[:, 0]].copy()
    w_at = weight[vi[:, 2], vi[:, 1], vi[:, 0]]
    missing = w_at < 1e-3
    if np.any(missing) and np.any(weight > 0):
        # Pull color from nearest occupied voxel (MC verts often sit on empty neighbors)
        occ = np.argwhere(weight > 0)  # (z,y,x)
        from scipy.spatial import cKDTree

        tree = cKDTree(occ.astype(np.float64))
        miss_ix = np.flatnonzero(missing)
        q = np.stack([vi[miss_ix, 2], vi[miss_ix, 1], vi[miss_ix, 0]], axis=1).astype(
            np.float64
        )
        _, nn = tree.query(q, k=1)
        oz, oy, ox = occ[nn, 0], occ[nn, 1], occ[nn, 2]
        vert_colors[miss_ix] = color_acc[oz, oy, ox]
        missing = np.zeros(len(vert_colors), dtype=bool)
    vert_colors[missing] = 0.55

    return (
        verts.astype(np.float64),
        faces.astype(np.int64),
        np.clip(vert_colors, 0, 1).astype(np.float64),
    )


def _project_vertex_colors(
    verts: np.ndarray,
    depthmaps: List[np.ndarray],
    rgbmaps: List[np.ndarray],
    camtoworlds: List[np.ndarray],
    Ks: List[np.ndarray],
    *,
    depth_eps: float = 0.05,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Photo-consistent vertex RGB by projecting into gsplat RGB+depth views.

    Prefer this over TSDF color volumes — those are muddy and read as noise.
    """
    n = len(verts)
    acc = np.zeros((n, 3), dtype=np.float64)
    wsum = np.zeros((n,), dtype=np.float64)
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.concatenate([verts.astype(np.float64), ones], axis=1)

    for depth, rgb, c2w, K in zip(depthmaps, rgbmaps, camtoworlds, Ks):
        H, W = depth.shape[:2]
        c2w = np.asarray(c2w, dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        K = np.asarray(K, dtype=np.float64)
        cam = (pts_h @ w2c.T)[:, :3]
        z = cam[:, 2]
        valid = z > 1e-4
        u = K[0, 0] * (cam[:, 0] / np.clip(z, 1e-4, None)) + K[0, 2]
        v = K[1, 1] * (cam[:, 1] / np.clip(z, 1e-4, None)) + K[1, 2]
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        in_img = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not np.any(in_img):
            continue
        idx = np.flatnonzero(in_img)
        ui_v = ui[idx]
        vi_v = vi[idx]
        z_v = z[idx]
        d = depth[vi_v, ui_v]
        ok = np.isfinite(d) & (d > 1e-4) & (np.abs(d - z_v) < depth_eps * np.maximum(d, 0.2))
        if not np.any(ok):
            continue
        idx2 = idx[ok]
        # Weight by front-facing proximity (smaller depth error → higher weight)
        err = np.abs(d[ok] - z_v[ok])
        w = 1.0 / (1e-3 + err)
        cols = rgb[vi_v[ok], ui_v[ok]].astype(np.float64)
        acc[idx2] += cols * w[:, None]
        wsum[idx2] += w

    out = np.zeros((n, 3), dtype=np.float64)
    hit = wsum > 1e-8
    out[hit] = acc[hit] / wsum[hit, None]
    if fallback is not None and np.any(~hit):
        fb = np.asarray(fallback, dtype=np.float64)
        if fb.shape == out.shape:
            out[~hit] = fb[~hit]
        else:
            out[~hit] = 0.55
    else:
        out[~hit] = 0.55
    return np.clip(out, 0.0, 1.0)


def _export_color_texture_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    OMB-safe textured GLB from vertex colors.

    ``ColorVisuals.to_texture()`` packs each *vertex* into a scattered atlas texel.
    Shared-triangle UVs then span distant atlas pixels → GPU interpolates a
    rainbow jumble (exactly the Scene Assembler screenshot failure mode).

    Fix: unweld faces and give each face one solid atlas cell; all three
    corners share the same UV (cell center) so fragments never cross cells.
    """
    import trimesh
    from PIL import Image

    if getattr(mesh.visual, "kind", None) != "vertex":
        return mesh

    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors)
    if colors.shape[0] != len(verts):
        raise WorldBakeError("vertex color count mismatch for atlas export")

    n_faces = int(len(faces))
    if n_faces < 1:
        return mesh

    # Unweld so faces do not share UV-bearing vertices across atlas cells.
    new_verts = verts[faces.reshape(-1)]
    face_vert_colors = colors[faces]  # (F, 3, 4)
    face_rgb = face_vert_colors[..., :3].astype(np.float64).mean(axis=1)  # (F, 3)
    face_rgb_u8 = np.clip(face_rgb, 0, 255).astype(np.uint8)
    new_faces = np.arange(n_faces * 3, dtype=np.int64).reshape(n_faces, 3)

    cols = int(np.ceil(np.sqrt(n_faces)))
    rows = int(np.ceil(n_faces / float(cols)))
    cell = 4
    atlas_w = cols * cell
    atlas_h = rows * cell
    max_atlas = 2048
    if max(atlas_w, atlas_h) > max_atlas:
        scale = max_atlas / float(max(atlas_w, atlas_h))
        cell = max(2, int(cell * scale))
        atlas_w = cols * cell
        atlas_h = rows * cell

    atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    atlas[..., 3] = 255

    fi = np.arange(n_faces, dtype=np.int64)
    rr = fi // cols
    cc = fi % cols
    # Vectorized cell fill via repeat into blocks
    for dy in range(cell):
        for dx in range(cell):
            atlas[rr * cell + dy, cc * cell + dx, :3] = face_rgb_u8

    u = (cc * cell + 0.5 * cell) / float(atlas_w)
    v = 1.0 - (rr * cell + 0.5 * cell) / float(atlas_h)
    uvs = np.zeros((n_faces * 3, 2), dtype=np.float64)
    uvs[0::3, 0] = u
    uvs[1::3, 0] = u
    uvs[2::3, 0] = u
    uvs[0::3, 1] = v
    uvs[1::3, 1] = v
    uvs[2::3, 1] = v

    out = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(atlas, mode="RGBA"),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    return out


def bake_world_env_mesh(
    world_dir: Path,
    *,
    cfg: Optional[EnvMeshBakeConfig] = None,
) -> Dict[str, Any]:
    """
    Bake ``environment_mesh.glb`` (+ ``collider.glb``) and update world.manifest.json.
    """
    import torch
    from gsplat.rendering import rasterization  # noqa: F401 — ensure importable

    from core.utils.lingbot_3dgs_train import ColmapTxtDataset
    from utils.mesh_decimate_utils import decimate_mesh

    cfg = cfg or EnvMeshBakeConfig()
    root = Path(world_dir)
    assert_bakeable_world(root)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise WorldBakeError("bake-env-mesh requires CUDA + gsplat")

    ply = resolve_environment_ply(root)
    gs_dataset = root / "gs_dataset"

    geom_mode = str(cfg.geometry_mode or "auto").strip().lower()
    if geom_mode == "auto":
        geom_mode = "camera_tsdf"

    color_src = str(cfg.color_source or "auto").strip().lower()
    if color_src == "auto":
        color_src = "gaussian_nn" if cfg.quality == "photo" else "photo_project"

    # --- Optional: mesh Gaussian centers (no cameras / fallback) ---
    if geom_mode in ("means_mc", "gaussian_means", "points_mc"):
        from core.utils.lingbot_3dgs_train import _read_phase_a_gaussian_init

        means_np, rgb_np, _op, _ls, _q, _sh = _read_phase_a_gaussian_init(
            ply, max_points=2_000_000
        )
        logger.info(
            "Env mesh bake (means_mc): %d Gaussians, ply=%s, voxel=%d",
            means_np.shape[0],
            ply.name,
            cfg.voxel_resolution,
        )
        verts, faces, vert_colors = _mesh_from_gaussian_means(
            means_np,
            rgb_np,
            voxel_resolution=int(cfg.voxel_resolution),
        )
        depth_eps = None
        camtoworlds = []
        bake_geom = "gaussian_means_mc"
        color_src = "gaussian_nn"
    else:
        dataset = ColmapTxtDataset(
            gs_dataset,
            data_factor=max(1, int(cfg.data_factor)),
            max_images=int(cfg.max_views) if cfg.max_views else None,
            # Served environment.ply is Studio/LingBot frame — do not undo X-mirror.
            studio_frame=True,
        )
        if len(dataset) < 2:
            raise WorldBakeError(f"Need ≥2 views for TSDF bake, got {len(dataset)}")

        means, quats, scales, opacities, colors = _load_gaussians_torch(ply, cfg.device)

        camtoworlds = []
        Ks: List[np.ndarray] = []
        widths: List[int] = []
        heights: List[int] = []
        photo_rgbs: List[np.ndarray] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            c2w = sample["camtoworld"]
            if hasattr(c2w, "numpy"):
                c2w = c2w.numpy()
            K = sample["K"]
            if hasattr(K, "numpy"):
                K = K.numpy()
            img = sample["image"]
            if hasattr(img, "numpy"):
                img = img.numpy()
            img_np = np.asarray(img, dtype=np.float32)
            if img_np.max() > 1.5:
                img_np = img_np / 255.0
            h, w = img_np.shape[:2]
            camtoworlds.append(np.asarray(c2w, dtype=np.float32))
            Ks.append(np.asarray(K, dtype=np.float32))
            heights.append(int(h))
            widths.append(int(w))
            photo_rgbs.append(np.clip(img_np[..., :3], 0.0, 1.0))

        if cfg.photo_exposure_normalize and color_src == "photo_project":
            photo_rgbs = _normalize_photo_exposure(photo_rgbs)

        logger.info(
            "Env mesh bake: %d Gaussians, %d views, ply=%s, quality=%s, color=%s",
            means.shape[0],
            len(camtoworlds),
            ply.name,
            cfg.quality,
            color_src,
        )

        from gsplat.rendering import rasterization as gsplat_rasterization
        from tqdm import tqdm

        rgbmaps: List[np.ndarray] = []
        depthmaps: List[np.ndarray] = []
        for i in tqdm(range(len(camtoworlds)), desc="bake render"):
            c2w = torch.from_numpy(camtoworlds[i]).float().to(cfg.device)
            K = torch.from_numpy(Ks[i]).float().to(cfg.device)
            W, H = widths[i], heights[i]
            viewmat = torch.linalg.inv(c2w)
            with torch.no_grad():
                renders, _, _ = gsplat_rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors,
                    viewmat[None],
                    K[None],
                    W,
                    H,
                    sh_degree=0,
                    render_mode="RGB+ED",
                    near_plane=0.01,
                    far_plane=1e4,
                    radius_clip=3.0,
                )
            rgbmaps.append(renders[0, ..., :3].clamp(0, 1).cpu().numpy().astype(np.float32))
            depthmaps.append(renders[0, ..., 3].cpu().numpy().astype(np.float32))

        means_np = means.detach().cpu().numpy()
        center, radius = _estimate_bounds(means_np, camtoworlds)
        depth_trunc = float(cfg.depth_trunc) if cfg.depth_trunc else max(2.0 * radius, 1.0)
        voxel_size = (2.0 * radius) / float(max(32, cfg.voxel_resolution))
        sdf_trunc = float(cfg.sdf_trunc_voxels) * voxel_size

        verts, faces, vert_colors_tsdf = _tsdf_fuse(
            depthmaps,
            rgbmaps,
            camtoworlds,
            Ks,
            center=center,
            radius=radius,
            voxel_resolution=cfg.voxel_resolution,
            depth_trunc=depth_trunc,
            sdf_trunc=sdf_trunc,
        )
        depth_eps = 0.18 if cfg.quality == "photo" else 0.12
        if color_src in ("gaussian_nn", "splat_nn", "means_nn"):
            # Pose-independent colors glued to Spark centers.
            from core.utils.lingbot_3dgs_train import _read_phase_a_gaussian_init

            g_means, g_rgb, *_rest = _read_phase_a_gaussian_init(ply, max_points=2_000_000)
            vert_colors = _colors_from_gaussian_nn(verts, g_means, g_rgb)
            bake_geom = "camera_tsdf_gauss_color"
        elif color_src in ("tsdf", "fused"):
            vert_colors = vert_colors_tsdf
            bake_geom = "camera_tsdf"
        else:
            vert_colors = _project_vertex_colors(
                verts,
                depthmaps,
                photo_rgbs,
                camtoworlds,
                Ks,
                depth_eps=depth_eps,
                fallback=vert_colors_tsdf,
            )
            bake_geom = "camera_tsdf"

    import trimesh

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=(np.clip(vert_colors, 0, 1) * 255).astype(np.uint8),
        process=True,
    )
    mesh.remove_unreferenced_vertices()
    # Drop TSDF/MC foam floaters — keep the dominant room shell(s).
    floater_info = _prune_small_components(mesh, min_face_frac=0.015)
    smooth_iters = 2 if cfg.quality == "photo" else 2
    try:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)
    except Exception as smooth_exc:
        logger.debug("Laplacian smooth skipped: %s", smooth_exc)
    before_faces = len(mesh.faces)
    color_std = float(np.std(vert_colors, axis=0).mean()) if len(vert_colors) else 0.0
    logger.info(
        "Env mesh pre-decimate: %s faces, vertex color std=%.4f, floater_prune=%s",
        before_faces,
        color_std,
        floater_info,
    )
    mesh, dec_info = decimate_mesh(
        mesh, target_face_count=int(cfg.target_face_count)
    )
    # Quadric decimation can reintroduce tiny islands — prune again.
    floater_after = _prune_small_components(mesh, min_face_frac=0.02)
    dec_info["floater_prune_after_decimate"] = floater_after

    # Studio photo quality: keep COLOR_0 (smooth shading). Atlas is for OMB/RP1.
    export_mode = str(cfg.color_export or "vertex").strip().lower()
    if export_mode in ("atlas", "omb", "pbr"):
        try:
            mesh = _export_color_texture_mesh(mesh)
            dec_info["color_export"] = "pbr_face_island_atlas"
        except Exception as tex_exc:
            logger.warning("Texture atlas export failed (keeping COLOR_0): %s", tex_exc)
            dec_info["color_export"] = "vertex_COLOR_0"
    else:
        dec_info["color_export"] = "vertex_COLOR_0"

    out_mesh = root / "environment_mesh.glb"
    mesh.export(out_mesh, file_type="glb")

    collider_name = "environment_mesh.glb"
    if cfg.write_collider:
        collider_path = root / "collider.glb"
        # Untextured collision proxy (same geometry)
        coll = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
        coll.export(collider_path, file_type="glb")
        collider_name = "collider.glb"

    bake_meta = {
        "backend": (
            "gaussian_means+skimage_mc+trimesh"
            if bake_geom == "gaussian_means_mc"
            else "gsplat+numpy_tsdf+skimage_mc+trimesh"
        ),
        "license_stack": "gsplat Apache-2.0; scikit-image BSD; trimesh MIT; scipy BSD",
        "source_ply": str(ply.relative_to(root)) if ply.is_relative_to(root) else ply.name,
        "geometry_mode": bake_geom,
        "color_source": color_src,
        "views": len(camtoworlds),
        "voxel_resolution": int(cfg.voxel_resolution),
        "quality": cfg.quality,
        "color_export": dec_info.get("color_export"),
        "photo_exposure_normalize": bool(cfg.photo_exposure_normalize),
        "faces_before_decimate": before_faces,
        "floater_prune": floater_info,
        "decimate": dec_info,
        "mesh_url": "environment_mesh.glb",
        "collider_url": collider_name,
    }
    # Fill path-specific diagnostics without NameError on means_mc.
    try:
        bake_meta["gaussian_count"] = int(means_np.shape[0])
    except Exception:
        try:
            bake_meta["gaussian_count"] = int(means.shape[0])
        except Exception:
            bake_meta["gaussian_count"] = None
    if bake_geom.startswith("camera_tsdf"):
        bake_meta["depth_eps"] = depth_eps
        bake_meta["depth_trunc"] = depth_trunc
        bake_meta["sdf_trunc"] = sdf_trunc
        bake_meta["radius"] = radius
        bake_meta["center"] = center.tolist()
        if bake_geom == "camera_tsdf_gauss_color":
            bake_meta["note"] = (
                "Geometry from camera-depth TSDF (walls); colors NN from splat RGB "
                "— avoids projective texture drift when poses misalign"
            )
    else:
        bake_meta["depth_eps"] = None
        bake_meta["note"] = (
            "Geometry from Gaussian means (matches Spark splat centers); "
            "colors NN from splat RGB — avoids camera-pose texture drift"
        )

    _update_manifest(root, bake_meta)
    (root / "env_mesh_bake.json").write_text(json.dumps(bake_meta, indent=2), encoding="utf-8")
    logger.info(
        "Env mesh bake done: %s faces → %s (%s)",
        before_faces,
        dec_info.get("after_faces"),
        out_mesh,
    )
    return bake_meta


def _update_manifest(world_dir: Path, bake_meta: Dict[str, Any]) -> None:
    path = Path(world_dir) / "world.manifest.json"
    if not path.is_file():
        logger.warning("No world.manifest.json — bake assets written without manifest update")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    env = manifest.setdefault("environment", {})
    env["mesh_url"] = bake_meta["mesh_url"]
    env["collider_url"] = bake_meta["collider_url"]
    meta = manifest.setdefault("metadata", {})
    meta["env_mesh_bake"] = bake_meta
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
