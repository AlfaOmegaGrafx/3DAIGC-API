"""Tests for utils3d.torch compatibility shims."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
utils3d = pytest.importorskip("utils3d")

from utils.utils3d_compat import ensure_utils3d_torch_compat


@pytest.fixture(autouse=True)
def _patch_utils3d():
    ensure_utils3d_torch_compat()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_perspective_from_fov_xy_matches_device_on_cuda():
    fov = torch.deg2rad(torch.tensor(40.0, device="cuda"))
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    assert projection.device.type == "cuda"
    assert projection.shape == (4, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_intrinsics_to_perspective_keeps_cuda_with_scalar_planes():
    intr = torch.eye(3, device="cuda")
    projection = utils3d.torch.intrinsics_to_perspective(intr, near=0.1, far=100.0)
    assert projection.device.type == "cuda"
    view = utils3d.torch.extrinsics_to_view(torch.eye(4, device="cuda"))
    verts = torch.randn(1, 8, 3, device="cuda")
    ctx = utils3d.torch.RastContext(backend="cuda")
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    out = utils3d.torch.rasterize_triangle_faces(
        ctx, verts, faces, 64, 64, view=view, projection=projection
    )
    assert out["mask"].device.type == "cuda"


def test_legacy_mesh_symbols_exist():
    ut = utils3d.torch
    for name in (
        "rasterize_triangle_faces",
        "compute_edges",
        "compute_connected_components",
        "compute_edge_connected_components",
        "compute_dual_graph",
        "remove_unreferenced_vertices",
    ):
        assert hasattr(ut, name), name
