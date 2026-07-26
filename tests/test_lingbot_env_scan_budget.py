"""Unit tests for LingBot env-scan frame budget + memory-safe export helpers."""

from pathlib import Path

import numpy as np

from core.utils.lingbot_map_pipeline import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MAX_FRAMES,
    HARD_CAP_MAX_FRAMES,
    WINDOWED_FRAME_THRESHOLD,
    _adaptive_keyframe_interval,
    _write_ply_xyzrgb_numpy,
    clamp_env_scan_frame_budget,
)


class TestEnvScanFrameBudget:
    def test_product_max_is_600(self):
        assert HARD_CAP_MAX_FRAMES == 600
        assert DEFAULT_MAX_FRAMES == 600
        assert DEFAULT_FRAME_STRIDE == 1

    def test_allows_full_600(self):
        assert clamp_env_scan_frame_budget(600, 1) == (600, 1)

    def test_clamps_above_hard_cap(self):
        mf, st = clamp_env_scan_frame_budget(5000, 2)
        assert mf == 600
        assert st == 2

    def test_keyframe_interval_scales_with_length(self):
        assert _adaptive_keyframe_interval(48) == 1
        assert _adaptive_keyframe_interval(320) == 1
        assert _adaptive_keyframe_interval(600) == 2
        assert _adaptive_keyframe_interval(960) == 3

    def test_windowed_threshold(self):
        assert WINDOWED_FRAME_THRESHOLD == 64


class TestPlyNumpyExport:
    def test_subsamples_without_python_point_lists(self, tmp_path: Path):
        n = 10_000
        verts = np.random.randn(n, 3).astype(np.float32)
        colors = (np.random.rand(n, 3) * 255).astype(np.uint8)
        out = tmp_path / "cloud.ply"
        written = _write_ply_xyzrgb_numpy(verts, colors, out, max_points=1_000)
        assert written == 1_000
        assert out.is_file()
        assert out.stat().st_size > 100
        raw = out.read_bytes()
        assert raw.startswith(b"ply\n")
        assert b"binary_little_endian" in raw.split(b"end_header\n", 1)[0]
