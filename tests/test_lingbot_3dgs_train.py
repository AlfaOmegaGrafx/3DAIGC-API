"""Unit tests for Phase B COLMAP TXT parsing + world apply (no GPU train)."""

from pathlib import Path

import numpy as np
import pytest

from core.utils.lingbot_3dgs_refine import _rotmat_to_quat
from core.utils.lingbot_3dgs_train import (
    _flip_x_c2w,
    _flip_x_points,
    _flip_x_quats_wxyz,
    _load_matrix_poses,
    _parse_colmap_cameras,
    _parse_colmap_images,
    _parse_colmap_points3d,
    apply_trained_ply_to_world,
)


def _write_mini_gs_dataset(root: Path):
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 64 64 50 50 32 32\n", encoding="utf-8"
    )
    # Identity-ish pose
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 000000.png\n\n",
        encoding="utf-8",
    )
    pts = ["1 0 0 0 10 20 30 0", "2 0.1 0 0 40 50 60 0"]
    (sparse / "points3D.txt").write_text("\n".join(pts) + "\n", encoding="utf-8")
    # tiny png
    try:
        import imageio.v2 as imageio

        imageio.imwrite(images / "000000.png", np.zeros((64, 64, 3), dtype=np.uint8))
    except Exception:
        (images / "000000.png").write_bytes(b"")


def test_parse_colmap_txt(tmp_path: Path):
    ds = tmp_path / "gs_dataset"
    _write_mini_gs_dataset(ds)
    cams = _parse_colmap_cameras(ds / "sparse" / "0" / "cameras.txt")
    assert 1 in cams
    assert cams[1]["width"] == 64
    images = _parse_colmap_images(ds / "sparse" / "0" / "images.txt")
    assert len(images) == 1
    assert images[0]["name"] == "000000.png"
    xyz, rgb = _parse_colmap_points3d(ds / "sparse" / "0" / "points3D.txt", 10)
    assert xyz.shape == (2, 3)
    assert rgb.shape == (2, 3)


def test_apply_trained_ply_updates_manifest(tmp_path: Path):
    world = tmp_path / "world"
    world.mkdir()
    (world / "gs_train").mkdir()
    # Minimal Spark-layout Gaussian PLY (1 vertex, 17 floats).
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
        "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
        "property float rot_2\nproperty float rot_3\nend_header\n"
    ).encode("ascii")
    verts = np.zeros((1, 17), dtype=np.float32)
    verts[0, 13] = 1.0  # rot_0 = 1 (identity quat)
    trained = tmp_path / "trained.ply"
    trained.write_bytes(header + verts.tobytes())
    # Phase A placeholder + points ref so prune can run lightly
    (world / "environment.ply").write_bytes(header + verts.tobytes())
    (world / "world.manifest.json").write_text(
        '{"environment":{"type":"gaussian_splat","renderer":"spark"},'
        '"metadata":{"gaussian_phase":"A_isotropic_from_points"}}',
        encoding="utf-8",
    )
    info = apply_trained_ply_to_world(
        world,
        trained,
        train_meta={"gaussian_count": 1, "max_steps": 100, "final_loss": 0.1},
        # Skip prune (needs >=1000 kept); Spark rewrite still runs.
        ref_xyz=None,
    )
    assert info["phase"] == "B_gsplat_trained"
    assert (world / "environment.phaseA.ply").is_file()
    man = (world / "world.manifest.json").read_text(encoding="utf-8")
    assert "B_gsplat_trained" in man


def test_rotmat_to_quat_rejects_improper():
    R = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="improper"):
        _rotmat_to_quat(R)


def test_flip_x_roundtrip_points_and_c2w():
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 1.0]], dtype=np.float32)
    assert np.allclose(_flip_x_points(_flip_x_points(pts)), pts)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    c2w[:3, :3] = np.diag([-1.0, 1.0, 1.0])  # improper
    fixed = _flip_x_c2w(c2w)
    assert np.linalg.det(fixed[:3, :3]) > 0
    assert np.allclose(_flip_x_c2w(fixed), c2w)


def test_flip_x_quats_preserves_so3():
    quats = np.array([[1, 0, 0, 0], [0.7071, 0.7071, 0, 0]], dtype=np.float32)
    out = _flip_x_quats_wxyz(quats)
    assert out.shape == quats.shape
    # Double flip ≈ identity orientation (up to sign)
    back = _flip_x_quats_wxyz(out)
    for a, b in zip(quats, back):
        assert abs(abs(np.dot(a, b)) - 1.0) < 1e-4


def test_load_matrix_poses_prefers_npy(tmp_path: Path):
    ds = tmp_path / "gs_dataset"
    ds.mkdir()
    poses = np.stack([np.eye(4), np.eye(4)], axis=0).astype(np.float32)
    poses[1, 0, 3] = 2.0
    np.save(ds / "poses_c2w.npy", poses)
    loaded = _load_matrix_poses(ds, 2)
    assert loaded is not None
    assert loaded.shape == (2, 4, 4)
    assert float(loaded[1, 0, 3]) == 2.0


def test_recolor_gaussian_ply_from_points(tmp_path: Path):
    from core.utils.lingbot_3dgs_train import _SH_C0, recolor_gaussian_ply_from_points
    from core.utils.lingbot_map_pipeline import _write_ply_xyzrgb_numpy

    # Points: one red, one green
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    points_ply = tmp_path / "points.ply"
    _write_ply_xyzrgb_numpy(pts, cols, points_ply, max_points=10)

    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
        "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
        "property float rot_2\nproperty float rot_3\nend_header\n"
    ).encode("ascii")
    verts = np.zeros((2, 17), dtype=np.float32)
    verts[0, 0:3] = [0, 0, 0]
    verts[1, 0:3] = [1, 0, 0]
    verts[:, 13] = 1.0
    # start with wrong blue-ish SH
    verts[:, 6:9] = ((np.array([0.0, 0.0, 1.0]) - 0.5) / _SH_C0).astype(np.float32)
    g_ply = tmp_path / "g.ply"
    g_ply.write_bytes(header + verts.tobytes())
    out = tmp_path / "out.ply"
    info = recolor_gaussian_ply_from_points(g_ply, points_ply, out)
    assert info["gaussian_count"] == 2
    assert info["far_kept_old_color"] == 0
    body = out.read_bytes().split(b"end_header\n", 1)[1]
    arr = np.frombuffer(body, dtype="<f4").reshape(2, 17)
    rgb = np.clip(0.5 + _SH_C0 * arr[:, 6:9], 0, 1)
    assert np.allclose(rgb[0], [1, 0, 0], atol=1e-3)
    assert np.allclose(rgb[1], [0, 1, 0], atol=1e-3)
