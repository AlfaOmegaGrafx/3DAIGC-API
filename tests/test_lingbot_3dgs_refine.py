"""Unit tests for Phase A env-scan → isotropic Gaussian PLY."""

from pathlib import Path

import numpy as np

from core.utils.lingbot_3dgs_refine import (
    _SH_C0,
    point_cloud_to_gaussian_ply,
    refine_point_cloud_world_to_gaussian,
    rgb_to_sh_dc,
)
from core.utils.lingbot_map_pipeline import _write_ply_xyzrgb_numpy


def test_rgb_to_sh_dc_mid_grey():
    # 128/255 ≈ 0.502 → near zero SH DC
    sh = rgb_to_sh_dc(np.array([[128, 128, 128]], dtype=np.uint8))
    assert sh.shape == (1, 3)
    assert abs(float(sh[0, 0])) < 0.02


def test_rgb_to_sh_dc_black_white():
    sh = rgb_to_sh_dc(np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8))
    assert sh[0, 0] < -1.0 / _SH_C0 * 0.4
    assert sh[1, 0] > 1.0 / _SH_C0 * 0.4


def test_point_cloud_to_gaussian_ply_header(tmp_path: Path):
    verts = np.random.randn(500, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (500, 3), dtype=np.uint8)
    out = tmp_path / "g.ply"
    n = point_cloud_to_gaussian_ply(verts, colors, out)
    assert n == 500
    head = out.read_bytes()[:400].decode("ascii", errors="ignore")
    assert "element vertex 500" in head
    assert "f_dc_0" in head
    assert "opacity" in head
    assert "scale_0" in head
    assert "rot_0" in head
    assert "property uchar red" not in head


def test_refine_world_package_updates_manifest(tmp_path: Path):
    world = tmp_path / "world"
    world.mkdir()
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]] + [[i * 0.01, 0, 0] for i in range(20)], dtype=np.float32)
    colors = np.full((verts.shape[0], 3), [40, 80, 120], dtype=np.uint8)
    _write_ply_xyzrgb_numpy(verts, colors, world / "environment.ply")
    (world / "world.manifest.json").write_text(
        '{"environment":{"type":"point_cloud","url":"environment.ply","renderer":"points"},'
        '"metadata":{"source_geometry":"point_cloud"}}',
        encoding="utf-8",
    )
    info = refine_point_cloud_world_to_gaussian(world, export_colmap=False)
    assert info["gaussian_count"] == verts.shape[0]
    assert (world / "environment.points.ply").is_file()
    man = (world / "world.manifest.json").read_text(encoding="utf-8")
    assert "gaussian_splat" in man
    assert "spark" in man
    head = (world / "environment.ply").read_bytes()[:300].decode("ascii", errors="ignore")
    assert "f_dc_0" in head
