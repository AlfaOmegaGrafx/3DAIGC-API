"""
LingBot-Map environment scan pipeline (optional dependency).

Walk-around RGB (Galaxy XR outward cameras / phone video / frame folder) →
poses + depth → colored point cloud + world.manifest.json with optional 1:1
metric scale.

Install (DGX):
  bash scripts/install_lingbot_map.sh

Without install, ``lingbot_map_available()`` is False and the adapter fails
with a clear message — default ``opennexus_image_to_world`` is unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.utils.metric_scale import apply_metric_scale_to_manifest, resolve_metric_calibration

logger = logging.getLogger(__name__)

LINGBOT_ROOT = Path(
    os.environ.get("LINGBOT_MAP_ROOT", "thirdparty/lingbot-map")
).resolve()
DEFAULT_WEIGHTS = Path(
    os.environ.get(
        "LINGBOT_MAP_WEIGHTS",
        str(LINGBOT_ROOT / "checkpoints" / "lingbot-map.pt"),
    )
)


def lingbot_map_available() -> bool:
    if not LINGBOT_ROOT.is_dir():
        return False
    # Package importable from that tree or site-packages
    try:
        import importlib.util

        if importlib.util.find_spec("lingbot_map") is not None:
            return True
    except Exception:
        pass
    return (LINGBOT_ROOT / "demo.py").is_file() or (LINGBOT_ROOT / "lingbot_map").is_dir()


def lingbot_map_status() -> Dict[str, Any]:
    return {
        "available": lingbot_map_available(),
        "root": str(LINGBOT_ROOT),
        "weights": str(DEFAULT_WEIGHTS),
        "weights_present": DEFAULT_WEIGHTS.is_file(),
        "install_hint": "bash scripts/install_lingbot_map.sh",
    }


# Product max remains 600. Peak UMA is bounded by keeping the image tensor on CPU
# and using windowed inference for long sequences (see `_run_lingbot_demo`).
DEFAULT_MAX_FRAMES = 600
DEFAULT_FRAME_STRIDE = 1
HARD_CAP_MAX_FRAMES = 600
# Switch to windowed LingBot path above this (keeps peak GPU ≈ one window).
WINDOWED_FRAME_THRESHOLD = 64
WINDOW_SIZE_KEYFRAMES = 32
WINDOW_OVERLAP_KEYFRAMES = 4
MAX_EXPORT_POINTS = 750_000


def clamp_env_scan_frame_budget(max_frames: int, stride: int) -> tuple:
    """Clamp env-scan sampling to the product hard cap."""
    mf = int(max_frames or DEFAULT_MAX_FRAMES)
    st = int(stride or DEFAULT_FRAME_STRIDE)
    if mf < 3:
        mf = 3
    if st < 1:
        st = 1
    if mf > HARD_CAP_MAX_FRAMES:
        logger.warning(
            "env-scan max_frames=%s exceeds hard cap %s — clamping",
            mf,
            HARD_CAP_MAX_FRAMES,
        )
        mf = HARD_CAP_MAX_FRAMES
    return mf, st


def _adaptive_keyframe_interval(num_frames: int) -> int:
    """Bound KV-cache growth for long walks (mirrors demo.py streaming heuristic)."""
    if num_frames <= 320:
        return 1
    # ceil(num_frames / 320)
    return max(1, (num_frames + 319) // 320)


def _plan_video_frame_indices(
    *,
    total_frames: int,
    native_fps: float,
    max_frames: int,
    stride: int,
) -> List[int]:
    """Pick frame indices that span the full clip (uniform), capped by ``max_frames``."""
    import numpy as np

    if total_frames < 3:
        return list(range(max(0, total_frames)))
    target_fps = 2.0 if stride <= 1 else float(max(1, 6 // stride))
    native_fps = float(native_fps) or 24.0
    duration_s = total_frames / native_fps
    expected = max(3, int(round(duration_s * target_fps)))
    count = min(int(max_frames), expected, total_frames)
    count = max(3, count)
    idx = np.unique(np.linspace(0, total_frames - 1, num=count, dtype=int))
    return [int(i) for i in idx]


def extract_frames_from_video(
    video_path: str,
    out_dir: Path,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    stride: int = DEFAULT_FRAME_STRIDE,
) -> List[Path]:
    """Extract RGB frames spanning the **full** video (temporally uniform, capped).

    ``max_frames`` is a sample budget across the whole clip — not “first N frames”.
    When ffmpeg is available we use an fps filter; otherwise OpenCV seeks evenly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    pattern = out_dir / "frame_%06d.jpg"
    ffmpeg = shutil.which("ffmpeg")
    # Target extract rate: stride 1 ≈ 2 fps across the clip; higher stride → lower fps.
    target_fps = 2.0 if stride <= 1 else float(max(1, 6 // stride))

    if ffmpeg:
        # fps filter samples the whole timeline; we then keep at most max_frames
        # evenly if the filter still produced more than the budget.
        vf = f"fps={target_fps:g}"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-vf",
            vf,
            "-q:v",
            "2",
            str(pattern),
        ]
        logger.info("Extracting frames with ffmpeg: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        frames = sorted(out_dir.glob("frame_*.jpg"))
        if len(frames) > max_frames:
            # Keep temporally uniform subset (first/last + evenly spaced).
            import numpy as np

            keep_idx = set(
                np.linspace(0, len(frames) - 1, num=max_frames, dtype=int).tolist()
            )
            for i, path in enumerate(frames):
                if i not in keep_idx:
                    path.unlink(missing_ok=True)
            frames = sorted(out_dir.glob("frame_*.jpg"))
    else:
        import cv2

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 3:
            # Some containers report 0 — fall back to sequential read count.
            probe = 0
            while True:
                ok, _frame = cap.read()
                if not ok:
                    break
                probe += 1
            total = probe
            cap.release()
            cap = cv2.VideoCapture(str(video))
        indices = _plan_video_frame_indices(
            total_frames=total,
            native_fps=native_fps,
            max_frames=max_frames,
            stride=stride,
        )
        duration_s = total / native_fps if native_fps > 0 else 0.0
        logger.info(
            "Extracting %d frames via OpenCV (no ffmpeg): native_fps=%.2f total=%s "
            "duration=%.1fs target_fps=%g → indices[%s…%s]",
            len(indices),
            native_fps,
            total,
            duration_s,
            target_fps,
            indices[0] if indices else None,
            indices[-1] if indices else None,
        )
        written = 0
        for want in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(want))
            ok, frame = cap.read()
            if not ok:
                continue
            path = out_dir / f"frame_{written:06d}.jpg"
            cv2.imwrite(str(path), frame)
            written += 1
        cap.release()
        frames = sorted(out_dir.glob("frame_*.jpg"))

    frames = frames[:max_frames]
    if len(frames) < 3:
        raise RuntimeError(
            f"Need at least 3 frames for environment scan (got {len(frames)} from {video})"
        )
    logger.info(
        "Extracted %d frames from %s (max_frames=%s, stride=%s, target_fps=%g)",
        len(frames),
        video.name,
        max_frames,
        stride,
        target_fps,
    )
    return frames


def collect_frame_paths(
    *,
    video_path: Optional[str] = None,
    frame_dir: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    work_dir: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    stride: int = DEFAULT_FRAME_STRIDE,
) -> List[Path]:
    max_frames, stride = clamp_env_scan_frame_budget(max_frames, stride)
    if frame_dir:
        d = Path(frame_dir)
        frames = sorted(
            [
                *d.glob("*.jpg"),
                *d.glob("*.jpeg"),
                *d.glob("*.png"),
                *d.glob("*.webp"),
            ]
        )[:max_frames]
        if len(frames) < 3:
            raise RuntimeError(f"frame_dir needs ≥3 images (got {len(frames)}): {frame_dir}")
        return frames

    if image_paths and len(image_paths) >= 3:
        return [Path(p) for p in image_paths[:max_frames]]

    if video_path:
        return extract_frames_from_video(
            video_path,
            work_dir / "frames",
            max_frames=max_frames,
            stride=stride,
        )

    raise ValueError(
        "Provide video_path, frame_dir, or ≥3 image_paths for LingBot-Map environment scan"
    )


def _rotation_aligning_a_to_b(a, b):
    """3x3 rotation mapping unit vector ``a`` onto unit vector ``b``."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=np.float64,
        )
        return (np.eye(3) + 2.0 * (K @ K)).astype(np.float64)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3, dtype=np.float64)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return (np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))).astype(np.float64)


def _estimate_up_from_camera_extrinsics(extrinsic) -> Optional[Any]:
    """
    Average camera 'up' from c2w extrinsics.

    LingBot / OpenCV cams store Y-down in camera space; world up is typically ``-Y_cam``.
    If the averaged up points mostly downward (common with bad windowed pose averages),
    flip it — otherwise gravity align maps the floor to the ceiling.
    """
    import numpy as np

    ext = np.asarray(extrinsic, dtype=np.float64)
    if ext.ndim == 2:
        ext = ext[None, ...]
    if ext.shape[-2:] != (3, 4) and not (ext.ndim == 3 and ext.shape[-1] >= 3):
        return None
    step = max(1, int(ext.shape[0]) // 48)
    ups = []
    for e in ext[::step]:
        R = np.asarray(e)[:3, :3]
        up = -R[:, 1]
        n = np.linalg.norm(up)
        if n > 1e-8:
            ups.append(up / n)
    if not ups:
        return None
    up = np.mean(np.stack(ups, axis=0), axis=0)
    n = np.linalg.norm(up)
    if n < 1e-8:
        return None
    up = up / n
    # Reject / flip inverted averages (Galaxy + windowed poses often land near -Y).
    if float(up[1]) < -0.35:
        up = -up
    return up


def _estimate_up_from_floor_ransac(verts, *, max_samples: int = 120_000):
    """Dominant plane normal (floor) via RANSAC — fallback when poses are missing."""
    import numpy as np

    pts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 200:
        # Degenerate — use PCA thinnest axis
        c = pts - pts.mean(axis=0)
        _, _, vh = np.linalg.svd(c, full_matrices=False)
        return vh[-1]

    rng = np.random.default_rng(0)
    if pts.shape[0] > max_samples:
        pts = pts[rng.choice(pts.shape[0], max_samples, replace=False)]

    # Seed with PCA thinnest axis; pick the denser end as the floor slab.
    c = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    rough_up = vh[-1]
    height = pts @ rough_up
    lo_thr = float(np.percentile(height, 12.0))
    hi_thr = float(np.percentile(height, 88.0))
    lo_count = int((height <= lo_thr).sum())
    hi_count = int((height >= hi_thr).sum())
    if hi_count > lo_count:
        rough_up = -rough_up
        height = -height
        lo_thr = float(np.percentile(height, 12.0))
    floor_pts = pts[height <= lo_thr]
    if floor_pts.shape[0] < 100:
        floor_pts = pts

    best_n = None
    best_count = 0
    thresh = max(0.02, float(np.std(height) * 0.05))
    for _ in range(96):
        i = rng.choice(floor_pts.shape[0], 3, replace=False)
        p0, p1, p2 = floor_pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        if ln < 1e-8:
            continue
        n = n / ln
        d = -float(n @ p0)
        inl = int((np.abs(floor_pts @ n + d) < thresh).sum())
        if inl > best_count:
            best_count = inl
            best_n = n

    up = best_n if best_n is not None else rough_up
    # Point "up" from floor slab toward the rest of the room.
    floor_h = float(np.mean(floor_pts @ up))
    cloud_h = float(np.mean(pts @ up))
    if cloud_h < floor_h:
        up = -up
    return up / (np.linalg.norm(up) + 1e-12)


def gravity_align_point_cloud(
    verts,
    colors,
    *,
    extrinsic=None,
    prefer_floor: bool = True,
):
    """
    Rotate XYZ(+RGB) so gravity is +Y (OpenNexus / Three.js), then seat on Y=0.

    Order (product convention after Office walk scans):
      1. Estimate up (floor RANSAC by default; camera poses only if prefer_floor=False)
      2. Align up → +Y
      3. If densest slab is at the top, Y-flip (ceiling/floor)
      4. Seat on Y=0
      5. X-mirror (OpenCV/LingBot left↔right vs Three.js)

    Default ``prefer_floor=True`` — floor RANSAC first. If camera extrinsics are
    available and strongly disagree with the floor normal (common on close-up
    wall-heavy scans where RANSAC locks onto a wall), fall back to camera-up.
    """
    import numpy as np

    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    method = "floor_ransac"
    up = None
    cam_up = None
    if extrinsic is not None:
        cam_up = _estimate_up_from_camera_extrinsics(extrinsic)
        if cam_up is not None and abs(float(cam_up[1])) < 0.45:
            # Horizontal-ish "up" is useless for gravity.
            cam_up = None
    if not prefer_floor and cam_up is not None:
        up = cam_up
        method = "camera_extrinsics"
    if up is None:
        up = _estimate_up_from_floor_ransac(verts)
        method = "floor_ransac"
        # Close-up / wall-dominated clouds: RANSAC may treat a wall as the floor
        # (Office Detail sideways). If camera-up is vertical and disagrees, trust camera.
        if cam_up is not None:
            floor_u = np.asarray(up, dtype=np.float64)
            floor_u = floor_u / (np.linalg.norm(floor_u) + 1e-12)
            cam_u = np.asarray(cam_up, dtype=np.float64)
            cam_u = cam_u / (np.linalg.norm(cam_u) + 1e-12)
            if float(np.dot(floor_u, cam_u)) < 0.5:
                up = cam_up
                method = "camera_extrinsics_vs_wall"
                logger.info(
                    "Floor RANSAC disagreed with camera-up (dot<%.2f) — using camera",
                    0.5,
                )

    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    R = _rotation_aligning_a_to_b(up, target)
    aligned = (verts.astype(np.float64) @ R.T).astype(np.float32)

    # Densest low slab should be the floor. If the low-Y band is emptier than the
    # high-Y band, we mapped ceiling→floor — flip Y.
    y = aligned[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    span = max(y1 - y0, 1e-6)
    low_mask = y <= y0 + 0.15 * span
    high_mask = y >= y1 - 0.15 * span
    y_flipped = False
    if int(high_mask.sum()) > int(low_mask.sum()) * 1.25:
        aligned[:, 1] *= -1.0
        method = f"{method}+y_flip"
        y_flipped = True

    y_seat_offset = float(aligned[:, 1].min())
    aligned[:, 1] -= y_seat_offset

    # OpenCV / LingBot world is mirrored vs Three.js Y-up (left↔right).
    # Reflect through the YZ plane after uprighting.
    aligned[:, 0] *= -1.0
    method = f"{method}+x_mirror"

    # Residual floor leveling — camera-up paths often leave a few degrees of tilt
    # ("sitting caddy-corner" / one corner high).
    aligned, R_level, level_deg = _residual_floor_level(aligned)
    if level_deg >= 0.5:
        method = f"{method}+level"

    # Yaw-align dominant horizontal axis to +X so walls sit square to the grid.
    aligned, R_yaw, yaw_deg = _yaw_align_to_world_x(aligned)
    if abs(yaw_deg) >= 1.0:
        method = f"{method}+yaw"

    # PCA slab flatten — residual tip after camera/floor up (Office Detail ~19°).
    # Only when the thin axis is already near +Y (room slab), not a vertical wall.
    aligned, R_pca, pca_deg = _pca_flatten_to_y(aligned)
    if pca_deg >= 1.0:
        method = f"{method}+pca_flat"

    y_seat_final = float(aligned[:, 1].min())
    aligned[:, 1] -= y_seat_final

    info = {
        "method": method,
        "up_vector": [float(x) for x in np.asarray(up).tolist()],
        "y_extent_m": float(aligned[:, 1].max() - aligned[:, 1].min()),
        "x_mirrored": True,
        # 3x3 applied as p' = p @ R.T then Y seat + X mirror — for camera export
        "rotation_3x3": R.tolist(),
        "y_flipped": y_flipped,
        "y_seat_offset": y_seat_offset,
        "y_seat_final": y_seat_final,
        "level_3x3": R_level.tolist(),
        "level_deg": float(level_deg),
        "yaw_3x3": (np.asarray(R_pca, dtype=np.float64) @ np.asarray(R_yaw, dtype=np.float64)).tolist()
        if pca_deg >= 1.0
        else R_yaw.tolist(),
        "yaw_deg": float(yaw_deg),
        "pca_flat_3x3": R_pca.tolist(),
        "pca_flat_deg": float(pca_deg),
    }
    logger.info(
        "Gravity-aligned point cloud via %s (Y extent %.2fm, level=%.1f°, yaw=%.1f°, pca=%.1f°)",
        method,
        info["y_extent_m"],
        level_deg,
        yaw_deg,
        pca_deg,
    )
    return aligned, colors, info


def _pca_flatten_to_y(verts):
    """
    Align the point cloud's minimum-variance PCA axis to +Y when it is already
    near vertical (room slab tipped a little). Skips wall-thin clouds.
    """
    import numpy as np

    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if len(v) < 100:
        return v.astype(np.float32), np.eye(3), 0.0
    c = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    n = vt[-1].astype(np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[1] < 0:
        n = -n
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    deg = float(np.degrees(np.arccos(np.clip(np.dot(n, target), -1.0, 1.0))))
    # Near-horizontal thin axis ⇒ wall-dominated cloud — do not flatten.
    if deg < 1.0 or deg > 35.0:
        return v.astype(np.float32), np.eye(3), 0.0 if deg > 35.0 else deg
    R = _rotation_aligning_a_to_b(n, target)
    out = (v @ R.T).astype(np.float32)
    return out, R, deg


def _residual_floor_level(verts):
    """Rotate so the low-point plane normal → +Y. Returns (verts, R, degrees)."""
    import numpy as np

    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if len(v) < 100:
        return v.astype(np.float32), np.eye(3), 0.0
    y = v[:, 1]
    low = v[y <= np.percentile(y, 12.0)]
    if len(low) < 50:
        low = v[y <= np.percentile(y, 20.0)]
    c = low.mean(axis=0)
    _, _, vt = np.linalg.svd(low - c, full_matrices=False)
    n = vt[-1].astype(np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[1] < 0:
        n = -n
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    deg = float(np.degrees(np.arccos(np.clip(np.dot(n, target), -1.0, 1.0))))
    # Skip large corrections — incomplete detail scans have messy "floors".
    if deg < 0.5 or deg > 12.0:
        return v.astype(np.float32), np.eye(3), 0.0 if deg > 12.0 else deg
    R = _rotation_aligning_a_to_b(n, target)
    out = (v @ R.T).astype(np.float32)
    return out, R, deg


def _yaw_align_to_world_x(verts):
    """
    Rotate around +Y so the dominant horizontal PCA axis aligns with world +X.
    """
    import numpy as np

    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if len(v) < 100:
        return v.astype(np.float32), np.eye(3), 0.0
    y0, y1 = np.percentile(v[:, 1], [5.0, 45.0])
    slab = v[(v[:, 1] >= y0) & (v[:, 1] <= y1)]
    if len(slab) < 50:
        slab = v
    xz = slab[:, [0, 2]]
    xz = xz - xz.mean(axis=0)
    cov = (xz.T @ xz) / max(len(xz), 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]  # [dx, dz] in XZ
    ang = float(np.arctan2(axis[1], axis[0]))  # radians from +X
    # Prefer the axis direction that needs ≤90° of yaw.
    if abs(ang) > 0.5 * np.pi:
        ang = ang - np.sign(ang) * np.pi
    # Rotate cloud by -ang: x' = c x + s z, z' = -s x + c z with c=cos(ang), s=sin(ang)
    deg = float(np.degrees(-ang))
    if abs(deg) < 1.0:
        return v.astype(np.float32), np.eye(3), deg
    c, s = float(np.cos(ang)), float(np.sin(ang))
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    out = (v @ R.T).astype(np.float32)
    return out, R, deg


def invert_gravity_aligned_points(
    verts,
    gravity_info: Dict[str, Any],
):
    """Undo gravity align (final seat → yaw → level → X-mirror → seat → flip → R)."""
    import numpy as np

    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3).copy()
    y_final = float(gravity_info.get("y_seat_final") or 0.0)
    v[:, 1] += y_final
    R_yaw = gravity_info.get("yaw_3x3")
    if R_yaw is not None:
        v = v @ np.asarray(R_yaw, dtype=np.float64)
    R_level = gravity_info.get("level_3x3")
    if R_level is not None:
        v = v @ np.asarray(R_level, dtype=np.float64)
    if gravity_info.get("x_mirrored"):
        v[:, 0] *= -1.0
    y_seat = float(gravity_info.get("y_seat_offset") or 0.0)
    v[:, 1] += y_seat
    if gravity_info.get("y_flipped"):
        v[:, 1] *= -1.0
    R = np.asarray(gravity_info["rotation_3x3"], dtype=np.float64)
    return (v @ R).astype(np.float32)


def _reexport_aligned_cameras(
    world_dir: Path,
    gravity_info: Dict[str, Any],
) -> Optional[Path]:
    """Rebuild ``cameras_aligned.npz`` after a gravity-align fix."""
    import numpy as np

    from core.utils.lingbot_3dgs_refine import apply_gravity_to_c2w

    world_dir = Path(world_dir)
    for cam_src in (
        world_dir / "cameras.npz",
        world_dir / "_work" / "lingbot_out" / "cameras.npz",
    ):
        if not cam_src.is_file():
            continue
        z = np.load(cam_src)
        ext_raw = z.get("extrinsic_raw")
        if ext_raw is None:
            ext_raw = z.get("extrinsic")
        intr = z.get("intrinsic")
        if ext_raw is None or intr is None:
            continue
        R = np.asarray(gravity_info.get("rotation_3x3"), dtype=np.float64)
        aligned_ext = apply_gravity_to_c2w(
            np.asarray(ext_raw),
            rotation_3x3=R,
            y_flipped=bool(gravity_info.get("y_flipped")),
            x_mirrored=bool(gravity_info.get("x_mirrored", True)),
            y_offset=float(gravity_info.get("y_seat_offset") or 0.0),
            level_3x3=gravity_info.get("level_3x3"),
            yaw_3x3=gravity_info.get("yaw_3x3"),
            y_seat_final=float(gravity_info.get("y_seat_final") or 0.0),
        )
        out = world_dir / "cameras_aligned.npz"
        np.savez_compressed(
            out,
            extrinsic=np.asarray(aligned_ext, dtype=np.float32),
            intrinsic=np.asarray(intr, dtype=np.float32),
        )
        ling_out = world_dir / "_work" / "lingbot_out"
        if ling_out.is_dir():
            np.savez_compressed(
                ling_out / "cameras_aligned.npz",
                extrinsic=np.asarray(aligned_ext, dtype=np.float32),
                intrinsic=np.asarray(intr, dtype=np.float32),
            )
        return out
    return None


def repair_world_gravity_alignment(
    world_dir: Path,
    *,
    metric_calibration: Optional[Dict[str, Any]] = None,
    rerun_phase_a: bool = True,
) -> Dict[str, Any]:
    """
    Re-seat a world that used unreliable camera-extrinsic gravity.

    Inverts the stored gravity transform, re-applies floor RANSAC (product default),
    refreshes PLYs + cameras, and optionally re-runs Phase A Gaussian export.
    """
    world_dir = Path(world_dir)
    man_path = world_dir / "world.manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(man_path)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    meta = dict(manifest.get("metadata") or {})
    old_grav = dict(meta.get("gravity_align") or {})
    if not old_grav.get("rotation_3x3"):
        raise RuntimeError(f"No gravity_align metadata in {man_path}")

    pts_path = world_dir / "environment.points.ply"
    if not pts_path.is_file():
        pts_path = world_dir / "environment.ply"
    verts, colors = _read_ply_xyzrgb_numpy(pts_path)
    raw = invert_gravity_aligned_points(verts, old_grav)

    extrinsic = None
    for cam_src in (
        world_dir / "cameras.npz",
        world_dir / "_work" / "lingbot_out" / "cameras.npz",
    ):
        if not cam_src.is_file():
            continue
        import numpy as np

        z = np.load(cam_src)
        extrinsic = z.get("extrinsic_raw")
        if extrinsic is None:
            extrinsic = z.get("extrinsic")
        break

    aligned, colors, new_grav = gravity_align_point_cloud(
        raw,
        colors,
        extrinsic=extrinsic,
        prefer_floor=True,
    )

    _write_ply_xyzrgb_numpy(aligned, colors, world_dir / "environment.points.ply")
    _write_ply_xyzrgb_numpy(aligned, colors, world_dir / "environment.ply")

    pred_npz = world_dir / "_work" / "lingbot_out" / "predictions.npz"
    if pred_npz.parent.is_dir():
        import numpy as np

        np.savez_compressed(
            pred_npz,
            points=aligned.astype(np.float32),
            colors=colors.astype(np.uint8),
        )
        pred_ply = pred_npz.parent / "predictions.ply"
        _write_ply_xyzrgb_numpy(aligned, colors, pred_ply)

    meta["gravity_align"] = new_grav
    manifest["metadata"] = meta
    if metric_calibration is not None:
        # apply_metric_scale multiplies into existing transform — reset to 1 first
        env = dict(manifest.get("environment") or {})
        xf = dict(env.get("transform") or {})
        xf["scale"] = [1.0, 1.0, 1.0]
        env["transform"] = xf
        manifest["environment"] = env
        manifest = apply_metric_scale_to_manifest(manifest, metric_calibration)
        cal_path = world_dir / "metric_calibration.json"
        cal_path.write_text(
            json.dumps(
                manifest.get("metadata", {}).get("metric_calibration") or {},
                indent=2,
            ),
            encoding="utf-8",
        )
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cam_path = _reexport_aligned_cameras(world_dir, new_grav)
    phase_a = None
    if rerun_phase_a:
        from core.utils.lingbot_3dgs_refine import refine_point_cloud_world_to_gaussian

        frames = world_dir / "_work" / "frames_flat"
        phase_a = refine_point_cloud_world_to_gaussian(
            world_dir,
            frames_dir=frames if frames.is_dir() else None,
            cameras_npz=cam_path,
            export_colmap=True,
        )

    return {
        "world_directory": str(world_dir),
        "old_gravity_method": old_grav.get("method"),
        "new_gravity_method": new_grav.get("method"),
        "y_extent_m": new_grav.get("y_extent_m"),
        "cameras_aligned": str(cam_path) if cam_path else None,
        "phase_a": phase_a,
    }


def _subsample_xyzrgb(verts, colors, *, max_points: int = MAX_EXPORT_POINTS):
    """Downsample XYZ+RGB arrays to at most ``max_points`` (shared stride)."""
    import numpy as np

    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = int(verts.shape[0])
    if n == 0:
        raise RuntimeError("Cannot subsample empty point cloud")
    stride = max(1, (n + max_points - 1) // max_points) if n > max_points else 1
    if stride > 1:
        verts = verts[::stride]
        colors = colors[::stride]
    return verts, colors


def _write_ply_xyzrgb_numpy(
    verts,
    colors,
    path: Path,
    *,
    max_points: int = MAX_EXPORT_POINTS,
) -> int:
    """Write XYZRGB PLY from numpy arrays (no giant Python point lists)."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    verts, colors = _subsample_xyzrgb(verts, colors, max_points=max_points)
    n = int(verts.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    packed = np.empty(n, dtype=dtype)
    packed["x"] = verts[:, 0]
    packed["y"] = verts[:, 1]
    packed["z"] = verts[:, 2]
    packed["r"] = colors[:, 0]
    packed["g"] = colors[:, 1]
    packed["b"] = colors[:, 2]
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(packed.tobytes())
    return n


def _read_ply_xyzrgb_numpy(path: Path):
    """Read binary_little_endian XYZRGB PLY written by ``_write_ply_xyzrgb_numpy``."""
    import numpy as np

    data = path.read_bytes()
    marker = b"end_header\n"
    end = data.find(marker)
    if end < 0:
        marker = b"end_header\r\n"
        end = data.find(marker)
    if end < 0:
        raise RuntimeError(f"PLY missing end_header: {path}")
    header = data[:end].decode("ascii", errors="ignore")
    body = data[end + len(marker) :]
    if "binary_little_endian" not in header:
        raise RuntimeError(f"Only binary_little_endian XYZRGB PLY supported: {path}")
    n = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[2])
            break
    if n < 1:
        raise RuntimeError(f"PLY has no vertices: {path}")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    packed = np.frombuffer(body, dtype=dtype, count=n)
    verts = np.stack([packed["x"], packed["y"], packed["z"]], axis=1)
    colors = np.stack([packed["r"], packed["g"], packed["b"]], axis=1)
    return verts, colors


def _write_ply_xyzrgb(points: Sequence[Sequence[float]], path: Path, *, max_points: int = MAX_EXPORT_POINTS) -> None:
    """Write colored point cloud PLY (binary LE) for browser-friendly loads."""
    import numpy as np

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Invalid points shape for PLY: {getattr(arr, 'shape', None)}")
    verts = arr[:, :3]
    if arr.shape[1] >= 6:
        colors = np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)
    else:
        colors = np.full((verts.shape[0], 3), 200, dtype=np.uint8)
    _write_ply_xyzrgb_numpy(verts, colors, path, max_points=max_points)


def _write_ascii_ply_xyzrgb(points: Sequence[Sequence[float]], path: Path) -> None:
    """Backward-compatible alias — writes binary XYZRGB (not ASCII)."""
    _write_ply_xyzrgb(points, path)


def _run_lingbot_demo(frames_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Run LingBot-Map inference in-process and write ``predictions.ply``.

    Memory-safe for up to ``HARD_CAP_MAX_FRAMES`` on GB10:
    - Keep the full image tensor on **CPU**; LingBot slices frames to GPU.
    - Use **windowed** inference for long walks (peak GPU ≈ one window).
    - Export PLY via numpy (no multi‑million Python point lists).
    """
    if not lingbot_map_available():
        raise RuntimeError(
            "LingBot-Map is not installed. Run: bash scripts/install_lingbot_map.sh "
            f"(expected root {LINGBOT_ROOT})"
        )
    if not DEFAULT_WEIGHTS.is_file():
        raise RuntimeError(
            f"LingBot-Map weights missing at {DEFAULT_WEIGHTS}. "
            "Run: bash scripts/install_lingbot_map.sh"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    import types

    import numpy as np
    import torch

    if str(LINGBOT_ROOT) not in sys.path:
        sys.path.insert(0, str(LINGBOT_ROOT))

    import demo as lingbot_demo  # type: ignore  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Images stay on CPU — inference_* moves slices to GPU (do NOT .to(device) the batch).
    images, _paths, _folder = lingbot_demo.load_images(
        image_folder=str(frames_dir),
        video_path=None,
        fps=10,
        first_k=None,
        stride=1,
        image_size=518,
        patch_size=14,
        rotate_clockwise_90=False,
    )
    if images.device.type != "cpu":
        images = images.cpu()

    num_frames = int(images.shape[0])
    if num_frames < 3:
        raise RuntimeError(f"Need ≥3 frames for LingBot-Map (got {num_frames})")

    scale_frames = min(8, max(1, num_frames - 1))
    keyframe_interval = _adaptive_keyframe_interval(num_frames)
    # Windowed path for long walks — peak GPU ≈ one window, not the full sequence.
    mode = "windowed" if num_frames >= WINDOWED_FRAME_THRESHOLD else "streaming"

    args = types.SimpleNamespace(
        mode=mode,
        model_path=str(DEFAULT_WEIGHTS),
        image_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=max(1024, num_frames),
        num_scale_frames=scale_frames,
        kv_cache_sliding_window=64,
        camera_num_iterations=1,
        use_sdpa=True,
        offload_to_cpu=True,
        keyframe_interval=keyframe_interval,
        window_size=WINDOW_SIZE_KEYFRAMES,
        overlap_size=None,
        overlap_keyframes=WINDOW_OVERLAP_KEYFRAMES,
    )

    logger.info(
        "LingBot-Map in-process: frames=%s mode=%s keyframe_interval=%s "
        "window_size=%s device=%s (images on CPU)",
        num_frames,
        mode,
        keyframe_interval,
        WINDOW_SIZE_KEYFRAMES if mode == "windowed" else None,
        device,
    )
    model = lingbot_demo.load_model(args, device)

    if torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        torch.cuda.reset_peak_memory_stats()
    else:
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    output_device = torch.device("cpu")

    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=dtype, enabled=device.type == "cuda"
    ):
        if mode == "windowed" and hasattr(model, "inference_windowed"):
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )

    del images
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info("LingBot-Map GPU peak during inference: %.2f GB", peak_gb)
        torch.cuda.empty_cache()

    images_for_post = predictions["images"]
    predictions, images_cpu = lingbot_demo.postprocess(predictions, images_for_post)
    vis = lingbot_demo.prepare_for_visualization(predictions, images_cpu)

    world_points = vis.get("world_points")
    conf = vis.get("world_points_conf")
    if world_points is None:
        depth = vis.get("depth")
        extrinsic = vis.get("extrinsic")
        intrinsic = vis.get("intrinsic")
        if depth is None or extrinsic is None or intrinsic is None:
            raise RuntimeError(
                "LingBot-Map produced neither world_points nor depth+poses "
                f"(keys={sorted(vis.keys())})"
            )
        from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

        world_points = unproject_depth_map_to_point_map(
            np.asarray(depth),
            np.asarray(extrinsic),
            np.asarray(intrinsic),
        )
        conf = vis.get("depth_conf")
        logger.info(
            "LingBot-Map: built point cloud from depth unprojection shape=%s",
            getattr(world_points, "shape", None),
        )

    world_points = np.asarray(world_points)
    if conf is None:
        conf = np.ones(world_points.shape[:-1], dtype=np.float32)
    else:
        conf = np.asarray(conf)

    imgs = vis.get("images")
    if imgs is None:
        raise RuntimeError("LingBot-Map produced no images for coloring")
    imgs = np.asarray(imgs)
    if imgs.ndim == 4 and imgs.shape[1] == 3:
        colors = np.transpose(imgs, (0, 2, 3, 1))
    else:
        colors = imgs
    colors = (colors.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    verts = world_points.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    thr = float(np.percentile(conf_flat, 50.0)) if conf_flat.size else 0.0
    mask = (conf_flat >= thr) & (conf_flat > 1e-5) & np.isfinite(verts).all(axis=1)
    verts = verts[mask]
    colors = colors[mask]
    if verts.shape[0] < 100:
        raise RuntimeError(
            f"LingBot-Map point cloud too small after confidence filter ({verts.shape[0]} pts)"
        )

    verts, colors, gravity_info = gravity_align_point_cloud(
        verts,
        colors,
        extrinsic=vis.get("extrinsic"),
        prefer_floor=True,
    )

    # Persist cameras for Phase B 3DGS training (poses aligned with the cloud).
    cameras_aligned_path = None
    try:
        ext = vis.get("extrinsic")
        intr = vis.get("intrinsic")
        if ext is not None and intr is not None:
            from core.utils.lingbot_3dgs_refine import apply_gravity_to_c2w

            ext_np = np.asarray(ext)
            R = np.asarray(gravity_info.get("rotation_3x3"), dtype=np.float64)
            y_flipped = bool(gravity_info.get("y_flipped"))
            y_seat = float(gravity_info.get("y_seat_offset") or 0.0)
            np.savez_compressed(
                output_dir / "cameras.npz",
                extrinsic_raw=ext_np.astype(np.float32),
                intrinsic=np.asarray(intr, dtype=np.float32),
                gravity_align=R.astype(np.float32),
                x_mirrored=np.array(True),
                y_flipped=np.array(y_flipped),
                y_seat_offset=np.array(y_seat, dtype=np.float32),
            )
            try:
                aligned_ext = apply_gravity_to_c2w(
                    ext_np,
                    rotation_3x3=R,
                    y_flipped=y_flipped,
                    x_mirrored=True,
                    y_offset=y_seat,
                    level_3x3=gravity_info.get("level_3x3"),
                    yaw_3x3=gravity_info.get("yaw_3x3"),
                    y_seat_final=float(gravity_info.get("y_seat_final") or 0.0),
                )
                cameras_aligned_path = output_dir / "cameras_aligned.npz"
                np.savez_compressed(
                    cameras_aligned_path,
                    extrinsic=np.asarray(aligned_ext, dtype=np.float32),
                    intrinsic=np.asarray(intr, dtype=np.float32),
                )
            except Exception as exc:
                logger.warning("Aligned camera export skipped: %s", exc)
    except Exception as exc:
        logger.warning("Camera export for 3DGS skipped: %s", exc)

    ply_path = output_dir / "predictions.ply"
    verts_out, colors_out = _subsample_xyzrgb(verts, colors)
    exported = _write_ply_xyzrgb_numpy(verts_out, colors_out, ply_path, max_points=verts_out.shape[0])
    # Persist the same capped cloud (never the raw tens-of-millions).
    np.savez_compressed(
        output_dir / "predictions.npz",
        points=verts_out.astype(np.float32),
        colors=colors_out.astype(np.uint8),
    )

    del model, predictions, vis
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "output_dir": str(output_dir),
        "ply_path": str(ply_path),
        "point_count": int(exported),
        "raw_point_count": int(verts.shape[0]),
        "num_frames": num_frames,
        "inference_mode": mode,
        "keyframe_interval": keyframe_interval,
        "gravity_align": gravity_info,
        "cameras_npz": str(cameras_aligned_path) if cameras_aligned_path else None,
        "frames_dir": str(frames_dir),
    }


def _load_points_arrays_from_predictions(pred_dir: Path):
    """Load XYZ+RGB as numpy arrays (prefers colored binary PLY, then NPZ)."""
    import numpy as np

    ply_candidates = sorted(pred_dir.rglob("*.ply"))
    for ply in ply_candidates:
        try:
            return _read_ply_xyzrgb_numpy(ply)
        except Exception as exc:
            logger.debug("PLY load skipped for %s: %s", ply, exc)

    for npz in sorted(pred_dir.rglob("*.npz")):
        try:
            data = np.load(npz, allow_pickle=True)
            if "points" not in data:
                continue
            verts = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
            if verts.shape[0] < 100:
                continue
            if "colors" in data:
                colors = np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
                if colors.shape[0] != verts.shape[0]:
                    colors = np.full((verts.shape[0], 3), 200, dtype=np.uint8)
            else:
                colors = np.full((verts.shape[0], 3), 200, dtype=np.uint8)
            return verts, colors
        except Exception as exc:
            logger.debug("NPZ load skipped for %s: %s", npz, exc)

    raise RuntimeError(
        f"No usable point cloud found under {pred_dir}. "
        "LingBot-Map may have failed or used an unexpected output layout."
    )


def _load_points_from_predictions(pred_dir: Path) -> List[List[float]]:
    """Best-effort load of xyz(+rgb) rows (capped) for legacy callers."""
    import numpy as np

    verts, colors = _load_points_arrays_from_predictions(pred_dir)
    verts, colors = _subsample_xyzrgb(verts, colors)
    merged = np.concatenate([verts, colors.astype(np.float32)], axis=1)
    return merged.tolist()


def build_environment_scan_world_package(
    *,
    work_dir: Path,
    world_id: str,
    world_name: str,
    points: Optional[Sequence[Sequence[float]]] = None,
    metric_calibration: Optional[Dict[str, Any]] = None,
    generation_info: Optional[Dict[str, Any]] = None,
    source_ply: Optional[Path] = None,
    point_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Write world package (PLY + manifest) with optional 1:1 metric scale."""
    world_dir = work_dir / "world"
    world_dir.mkdir(parents=True, exist_ok=True)
    ply_path = world_dir / "environment.ply"

    if source_ply is not None and Path(source_ply).is_file():
        shutil.copy2(source_ply, ply_path)
        count = int(point_count) if point_count is not None else None
        if count is None:
            try:
                verts, _colors = _read_ply_xyzrgb_numpy(ply_path)
                count = int(verts.shape[0])
            except Exception:
                count = 0
    else:
        if points is None:
            raise RuntimeError("build_environment_scan_world_package needs points or source_ply")
        _write_ascii_ply_xyzrgb(points, ply_path)
        count = len(points)

    manifest: Dict[str, Any] = {
        "id": world_id,
        "version": 1,
        "name": world_name,
        "spawn": {"position": [0, 0, 0], "rotation_y": 0, "player_height": 1.6},
        "environment": {
            "type": "point_cloud",
            "url": "environment.ply",
            "format": "ply",
            "renderer": "points",
            "transform": {"scale": 1.0},
        },
        "props": [],
        "coordinate_system": "y-up",
        "metadata": {
            "pipeline": "lingbot_map_environment_scan",
            "source_geometry": "point_cloud",
            "point_count": count,
            **(generation_info or {}),
        },
    }
    manifest = apply_metric_scale_to_manifest(manifest, metric_calibration)

    manifest_path = world_dir / "world.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cal_path = world_dir / "metric_calibration.json"
    cal_path.write_text(
        json.dumps(manifest.get("metadata", {}).get("metric_calibration") or {}, indent=2),
        encoding="utf-8",
    )

    return {
        "world_directory": str(world_dir),
        "world_manifest_path": str(manifest_path),
        "output_splat_path": str(ply_path),
        "output_mesh_path": str(ply_path),
        "manifest": manifest,
        "metric_calibration": manifest.get("metadata", {}).get("metric_calibration"),
    }


def run_environment_scan(
    *,
    job_id: str,
    video_path: Optional[str] = None,
    frame_dir: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    metric_calibration: Optional[Dict[str, Any]] = None,
    world_name: Optional[str] = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    stride: int = DEFAULT_FRAME_STRIDE,
    output_root: Optional[Path] = None,
    refine_to_3dgs: bool = False,
    train_3dgs: bool = False,
    train_3dgs_steps: int = 7000,
) -> Dict[str, Any]:
    """
    Full scan: frames → LingBot-Map → metric world package.

    If ``refine_to_3dgs`` is True, Phase A converts the colored point cloud into
    Spark-compatible isotropic Gaussians and exports COLMAP for Phase B training.

    If ``train_3dgs`` is True, Phase B gsplat train runs after Phase A (implies A).

    If LingBot is not installed, raises RuntimeError (caller surfaces as job failure).
    """
    max_frames, stride = clamp_env_scan_frame_budget(max_frames, stride)
    root = Path(output_root or Path("outputs") / "worlds" / job_id)
    root.mkdir(parents=True, exist_ok=True)
    work = root / "_work"
    work.mkdir(parents=True, exist_ok=True)

    frames = collect_frame_paths(
        video_path=video_path,
        frame_dir=frame_dir,
        image_paths=image_paths,
        work_dir=work,
        max_frames=max_frames,
        stride=stride,
    )
    frames_dir = work / "frames_flat"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(frames):
        dst = frames_dir / f"frame_{i:06d}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)

    pred_dir = work / "lingbot_out"
    demo_info = _run_lingbot_demo(frames_dir, pred_dir)
    src_ply = Path(demo_info["ply_path"])
    verts, _colors = _load_points_arrays_from_predictions(pred_dir)

    # Optional: if calibration only has true_meters, estimate recon_length from bbox diagonal
    cal = dict(metric_calibration or {}) if metric_calibration else None
    if cal and cal.get("true_meters") and cal.get("recon_length") is None and cal.get("mode") in (
        None,
        "reference_length",
        "auto_bbox",
    ):
        import numpy as np

        arr = np.asarray(verts, dtype=float)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        cal["mode"] = "reference_length"
        cal["recon_length"] = diag
        # Interpret true_meters as real-world length of the dominant room span
        # (user should prefer door/wall two-point calibration when possible).
        logger.warning(
            "metric_calibration used auto bbox diagonal=%.4g as recon_length; "
            "prefer two_points or an explicit measured recon_length for accurate 1:1",
            diag,
        )

    package = build_environment_scan_world_package(
        work_dir=root,
        world_id=job_id,
        world_name=world_name or f"scan-{job_id[:8]}",
        source_ply=src_ply,
        point_count=int(demo_info.get("point_count") or verts.shape[0]),
        metric_calibration=cal,
        generation_info={
            "frame_count": len(frames),
            "lingbot_map": lingbot_map_status(),
            "inference_mode": demo_info.get("inference_mode"),
            "raw_point_count": demo_info.get("raw_point_count"),
            "gravity_align": demo_info.get("gravity_align"),
        },
    )

    # Flatten: move world/* to job root for URL layout compatibility
    world_src = Path(package["world_directory"])
    for item in world_src.iterdir():
        dest = root / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))

    # Keep LingBot cameras next to the world package for offline 3DGS refine.
    for cam_name in ("cameras.npz", "cameras_aligned.npz"):
        cam_src = pred_dir / cam_name
        if cam_src.is_file():
            shutil.copy2(cam_src, root / cam_name)

    gaussian_info = None
    gaussian_train = None
    do_phase_a = bool(refine_to_3dgs or train_3dgs)
    if do_phase_a:
        from core.utils.lingbot_3dgs_refine import refine_point_cloud_world_to_gaussian

        cams = root / "cameras_aligned.npz"
        if not cams.is_file():
            cams = pred_dir / "cameras_aligned.npz"
        gaussian_info = refine_point_cloud_world_to_gaussian(
            root,
            frames_dir=frames_dir,
            cameras_npz=cams if cams.is_file() else None,
            export_colmap=True,
        )
        logger.info(
            "Phase A 3DGS refine: %s gaussians → %s",
            gaussian_info.get("gaussian_count"),
            gaussian_info.get("environment_ply"),
        )

    if train_3dgs:
        from core.utils.lingbot_3dgs_train import (
            PhaseBTrainConfig,
            gsplat_available,
            train_and_apply_phase_b,
        )

        if not gsplat_available():
            raise RuntimeError(
                "train_3dgs requested but gsplat/CUDA is unavailable on this host"
            )
        if not (root / "gs_dataset").is_dir():
            raise RuntimeError(
                "train_3dgs requires gs_dataset/ from Phase A camera export"
            )
        steps = max(100, int(train_3dgs_steps or 7000))
        gaussian_train = train_and_apply_phase_b(
            root,
            cfg=PhaseBTrainConfig(
                max_steps=steps,
                enable_densify=False,
            ),
        )
        logger.info(
            "Phase B gsplat train: %s gaussians loss=%s",
            gaussian_train.get("gaussian_count"),
            gaussian_train.get("final_loss"),
        )

    manifest_path = root / "world.manifest.json"
    ply_path = root / "environment.ply"
    package.update(
        {
            "world_directory": str(root),
            "world_manifest_path": str(manifest_path),
            "world_manifest_url": f"/api/v1/system/jobs/{job_id}/download?asset=manifest",
            "world_base_url": f"/api/v1/system/jobs/{job_id}/world/",
            "output_splat_path": str(ply_path),
            "success": True,
            "gaussian_refine": gaussian_info,
            "gaussian_train": gaussian_train,
        }
    )
    return package
