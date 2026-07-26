"""Unit tests for LingBot env-scan frame budget + PLY export helpers."""

from pathlib import Path

import numpy as np

from core.utils.lingbot_map_pipeline import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MAX_FRAMES,
    HARD_CAP_MAX_FRAMES,
    WINDOWED_FRAME_THRESHOLD,
    _adaptive_keyframe_interval,
    _plan_video_frame_indices,
    _read_ply_xyzrgb_numpy,
    _write_ply_xyzrgb_numpy,
    clamp_env_scan_frame_budget,
)


def test_product_max_is_600():
    assert HARD_CAP_MAX_FRAMES == 600
    assert DEFAULT_MAX_FRAMES == 600
    assert DEFAULT_FRAME_STRIDE == 1


def test_clamp_allows_600():
    mf, st = clamp_env_scan_frame_budget(600, 1)
    assert mf == 600
    assert st == 1


def test_clamp_caps_above_600():
    mf, st = clamp_env_scan_frame_budget(5000, 2)
    assert mf == 600
    assert st == 2


def test_adaptive_keyframe_interval():
    assert _adaptive_keyframe_interval(48) == 1
    assert _adaptive_keyframe_interval(320) == 1
    assert _adaptive_keyframe_interval(600) == 2
    assert WINDOWED_FRAME_THRESHOLD == 64


def test_write_ply_numpy_subsamples(tmp_path: Path):
    n = 2_000_000
    verts = np.random.randn(n, 3).astype(np.float32)
    colors = np.full((n, 3), 128, dtype=np.uint8)
    out = tmp_path / "cloud.ply"
    written = _write_ply_xyzrgb_numpy(verts, colors, out, max_points=750_000)
    assert out.is_file()
    assert written <= 750_000
    assert written > 100
    head = out.read_bytes()[:200].decode("ascii", errors="ignore")
    assert f"element vertex {written}" in head


def test_write_read_ply_roundtrip_keeps_colors(tmp_path: Path):
    verts = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    out = tmp_path / "c.ply"
    n = _write_ply_xyzrgb_numpy(verts, colors, out)
    assert n == 3
    v2, c2 = _read_ply_xyzrgb_numpy(out)
    assert v2.shape == (3, 3)
    assert np.allclose(v2, verts)
    assert np.array_equal(c2, colors)


def test_plan_video_frame_indices_spans_full_clip():
    # Office-like: 4377 frames @ 24fps (~182s), max 600, stride 1 → ~2fps ≈ 365 samples
    idx = _plan_video_frame_indices(
        total_frames=4377, native_fps=24.0, max_frames=600, stride=1
    )
    assert idx[0] == 0
    assert idx[-1] == 4376
    assert 300 <= len(idx) <= 400
    # Must NOT be the old bug (first 600 consecutive frames only)
    assert idx[100] > 100


def test_gravity_align_recovers_tilted_room():
    from core.utils.lingbot_map_pipeline import gravity_align_point_cloud

    xs, zs = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
    floor = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
    ceil = floor + np.array([0.0, 2.4, 0.0])
    pts = np.concatenate([floor, ceil]).astype(np.float32)
    cols = np.full((len(pts), 3), 100, dtype=np.uint8)
    th = np.deg2rad(40.0)
    R = np.array(
        [[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tilted = (pts.astype(np.float64) @ R.T).astype(np.float32)
    aligned, _, info = gravity_align_point_cloud(tilted, cols)
    assert "floor_ransac" in info["method"]
    assert info.get("x_mirrored") is True
    assert abs(float(aligned[:, 1].min())) < 0.05
    assert 2.0 < float(aligned[:, 1].max()) < 2.9


def test_gravity_rejects_inverted_camera_up():
    """Windowed LingBot poses can average to world -Y; must not leave the room on its head."""
    from core.utils.lingbot_map_pipeline import (
        _estimate_up_from_camera_extrinsics,
        gravity_align_point_cloud,
    )

    # c2w with camera Y already = world +Y ⇒ -Y_cam = world -Y (the bug case)
    ext = np.zeros((4, 3, 4), dtype=np.float64)
    for i in range(4):
        ext[i, :3, :3] = np.eye(3)
        ext[i, :3, 3] = [i * 0.1, 1.5, 0.0]
    up = _estimate_up_from_camera_extrinsics(ext)
    assert up is not None
    assert float(up[1]) > 0.5  # flipped to point upward

    xs, zs = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    floor = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
    # denser floor
    floor = np.concatenate([floor, floor + 0.01], axis=0)
    ceil = floor[:400] + np.array([0.0, 2.5, 0.0])
    pts = np.concatenate([floor, ceil]).astype(np.float32)
    cols = np.full((len(pts), 3), 80, dtype=np.uint8)
    aligned, _, info = gravity_align_point_cloud(pts, cols, extrinsic=ext, prefer_floor=True)
    assert "floor_ransac" in info["method"]
    assert info["x_mirrored"] is True
    assert info["y_extent_m"] < 3.2
    assert float(aligned[:, 1].min()) > -0.05
