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
    n = point_cloud_to_gaussian_ply(verts, colors, out, scale=0.012)
    assert n == 500
    head = out.read_bytes()[:400].decode("ascii", errors="ignore")
    assert "element vertex 500" in head
    assert "f_dc_0" in head
    assert "opacity" in head
    assert "scale_0" in head
    assert "rot_0" in head
    assert "property uchar red" not in head


def test_point_cloud_to_gaussian_adaptive_scales_smaller_than_legacy(tmp_path: Path):
    rng = np.random.default_rng(0)
    # Dense local cluster — adaptive scales should be << legacy 0.012
    verts = (rng.normal(size=(2000, 3)) * 0.01).astype(np.float32)
    colors = np.full((2000, 3), 120, dtype=np.uint8)
    out = tmp_path / "adaptive.ply"
    point_cloud_to_gaussian_ply(verts, colors, out)  # scale=None → adaptive
    raw = out.read_bytes()
    # skip header
    payload = raw.split(b"end_header\n", 1)[1]
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
    arr = np.frombuffer(payload, dtype=dtype)
    lin = np.exp(arr["scale_0"])
    assert float(np.median(lin)) < 0.006
    assert float(np.median(lin)) > 1e-4


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
