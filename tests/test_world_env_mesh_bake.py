"""Unit tests for env mesh bake guards (no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.utils.world_env_mesh_bake import (
    WorldBakeError,
    assert_bakeable_world,
    world_has_bake_cameras,
)


def test_world_has_bake_cameras_false_for_empty(tmp_path: Path):
    assert world_has_bake_cameras(tmp_path) is False


def test_world_has_bake_cameras_true_with_gs_dataset(tmp_path: Path):
    gs = tmp_path / "gs_dataset"
    (gs / "images").mkdir(parents=True)
    (gs / "images" / "000.png").write_bytes(b"x")
    (gs / "poses_c2w.npy").write_bytes(b"")  # existence only for this unit test
    assert world_has_bake_cameras(tmp_path) is True


def test_assert_bakeable_rejects_image_to_world_without_cameras(tmp_path: Path):
    (tmp_path / "environment.ply").write_bytes(b"ply\n" + b"\0" * 80)
    (tmp_path / "world.manifest.json").write_text(
        json.dumps(
            {
                "id": "w1",
                "environment": {"url": "environment.ply"},
                "metadata": {"pipeline": "image-to-world-v1"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorldBakeError, match="TripoSplat|image-to-world"):
        assert_bakeable_world(tmp_path)


def test_assert_bakeable_rejects_missing_gs_dataset(tmp_path: Path):
    (tmp_path / "environment.ply").write_bytes(b"ply\n" + b"\0" * 80)
    with pytest.raises(WorldBakeError, match="gs_dataset"):
        assert_bakeable_world(tmp_path)


def test_tsdf_origin_axis_grids_do_not_broadcast_error():
    """Regression: voxel_res=192 + origin(3,) used to raise broadcast (192,)/(3,)."""
    import numpy as np

    from core.utils.world_env_mesh_bake import _tsdf_fuse

    res = 32
    center = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    radius = 1.0
    # One synthetic depth/rgb view looking +Z at origin
    H = W = 64
    depth = np.full((H, W), 1.5, dtype=np.float32)
    rgb = np.full((H, W, 3), 0.5, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float64)
    c2w[2, 3] = -1.5
    K = np.array([[50.0, 0, 32.0], [0, 50.0, 32.0], [0, 0, 1.0]], dtype=np.float64)
    verts, faces, colors = _tsdf_fuse(
        [depth],
        [rgb],
        [c2w],
        [K],
        center=center,
        radius=radius,
        voxel_resolution=res,
        depth_trunc=4.0,
        sdf_trunc=0.2,
    )
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert colors.shape == (len(verts), 3)


def test_face_island_atlas_uvs_do_not_span_atlas():
    """Regression: scattered per-vertex UVs caused rainbow jumble in Scene Assembler."""
    import numpy as np
    import trimesh

    from core.utils.world_env_mesh_bake import _export_color_texture_mesh

    m = trimesh.creation.icosphere(subdivisions=2)
    c = (np.clip(m.vertices * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    m.visual.vertex_colors = np.concatenate(
        [c, np.full((len(c), 1), 255, np.uint8)], axis=1
    )
    out = _export_color_texture_mesh(m)
    uv = np.asarray(out.visual.uv)
    f = np.asarray(out.faces)
    # Within each face, all three UVs must be identical (solid cell).
    for tri in f[:200]:
        assert np.allclose(uv[tri[0]], uv[tri[1]])
        assert np.allclose(uv[tri[1]], uv[tri[2]])
    assert out.visual.kind == "texture"
