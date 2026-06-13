"""
Minimal Open3D compatibility layer for aarch64 hosts without PyPI open3d wheels.

TRELLIS mesh painting uses open3d for voxelization and point-in-mesh tests. This shim
implements the small subset of the API those code paths need via trimesh/numpy.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Iterable, Optional

import numpy as np

_INSTALLED = False


class _VectorArray:
    def __init__(self, data: Iterable):
        self._data = np.asarray(data)

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


class _TriangleMesh:
    def __init__(
        self,
        vertices: Optional[_VectorArray] = None,
        triangles: Optional[_VectorArray] = None,
    ):
        self._vertices = (
            np.asarray(vertices._data) if vertices is not None else np.zeros((0, 3))
        )
        self._triangles = (
            np.asarray(triangles._data)
            if triangles is not None
            else np.zeros((0, 3), dtype=np.int32)
        )

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        if isinstance(value, _VectorArray):
            self._vertices = np.asarray(value._data)
        else:
            self._vertices = np.asarray(value)

    @property
    def triangles(self):
        return self._triangles

    @triangles.setter
    def triangles(self, value):
        if isinstance(value, _VectorArray):
            self._triangles = np.asarray(value._data)
        else:
            self._triangles = np.asarray(value)

    @property
    def faces(self):
        return self.triangles

    def is_watertight(self) -> bool:
        import trimesh

        if len(self.vertices) == 0 or len(self.triangles) == 0:
            return False
        return trimesh.Trimesh(
            vertices=self.vertices, faces=self.triangles, process=False
        ).is_watertight


class _PointCloud:
    def __init__(self):
        self.points = np.zeros((0, 3))
        self.colors = np.zeros((0, 3))

    def paint_uniform_color(self, color):
        self.colors = np.tile(np.asarray(color, dtype=np.float64), (len(self.points), 1))


class _Voxel:
    def __init__(self, grid_index: np.ndarray):
        self.grid_index = grid_index


class _VoxelGrid:
    def __init__(self, voxels: list[_Voxel]):
        self._voxels = voxels

    def get_voxels(self):
        return self._voxels

    @staticmethod
    def create_from_triangle_mesh_within_bounds(
        mesh: _TriangleMesh,
        voxel_size: float,
        min_bound: tuple[float, float, float],
        max_bound: tuple[float, float, float],
    ) -> "_VoxelGrid":
        import trimesh

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        vox = tm.voxelized(pitch=float(voxel_size))
        origin = np.asarray(min_bound, dtype=np.float64)
        indices = []
        for point in vox.points:
            if np.all(point >= min_bound) and np.all(point <= max_bound):
                grid_index = np.floor((point - origin) / voxel_size).astype(np.int32)
                indices.append(_Voxel(grid_index))
        if not indices and vox.filled_count > 0:
            matrix = vox.matrix
            for idx in np.argwhere(matrix):
                point = vox.transform[:3, :3] @ idx + vox.transform[:3, 3]
                if np.all(point >= min_bound) and np.all(point <= max_bound):
                    grid_index = np.floor((point - origin) / voxel_size).astype(np.int32)
                    indices.append(_Voxel(grid_index))
        return _VoxelGrid(indices)


class _TensorWrapper:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=np.float32)

    def numpy(self):
        return self._data


class _RaycastingScene:
    def __init__(self):
        self._mesh = None

    def add_triangles(self, mesh_t: "_TensorTriangleMesh"):
        self._mesh = mesh_t._legacy

    def compute_signed_distance(self, query_points: _TensorWrapper) -> _TensorWrapper:
        import trimesh

        if self._mesh is None:
            raise RuntimeError("RaycastingScene has no mesh")
        tm = trimesh.Trimesh(
            vertices=np.asarray(self._mesh.vertices),
            faces=np.asarray(self._mesh.triangles),
            process=False,
        )
        pts = np.asarray(query_points._data)
        dist = trimesh.proximity.signed_distance(tm, pts)
        return _TensorWrapper(dist.astype(np.float32))


class _TensorTriangleMesh:
    def __init__(self, legacy: _TriangleMesh):
        self._legacy = legacy

    @staticmethod
    def from_legacy(mesh: _TriangleMesh) -> "_TensorTriangleMesh":
        return _TensorTriangleMesh(mesh)


def install_open3d_shim() -> None:
    """Register a minimal ``open3d`` module when the real package is unavailable."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import open3d  # noqa: F401

        _INSTALLED = True
        return
    except ImportError:
        pass

    geometry = types.ModuleType("open3d.geometry")
    geometry.TriangleMesh = _TriangleMesh
    geometry.PointCloud = _PointCloud
    geometry.VoxelGrid = _VoxelGrid

    utility = types.ModuleType("open3d.utility")
    utility.Vector3dVector = _VectorArray
    utility.Vector3iVector = _VectorArray

    io_mod = types.ModuleType("open3d.io")

    def _write_point_cloud(path: str, pcd: _PointCloud) -> None:
        import trimesh

        cloud = trimesh.PointCloud(
            vertices=np.asarray(pcd.points),
            colors=(np.asarray(pcd.colors) * 255).astype(np.uint8)
            if len(pcd.colors)
            else None,
        )
        cloud.export(path)

    io_mod.write_point_cloud = _write_point_cloud

    t_geometry = types.ModuleType("open3d.t.geometry")
    t_geometry.RaycastingScene = _RaycastingScene
    t_geometry.TriangleMesh = _TensorTriangleMesh

    core = types.ModuleType("open3d.core")
    core.Tensor = _TensorWrapper

    t_mod = types.ModuleType("open3d.t")
    t_mod.geometry = t_geometry
    t_mod.core = core

    o3d = types.ModuleType("open3d")
    o3d.geometry = geometry
    o3d.utility = utility
    o3d.io = io_mod
    o3d.t = t_mod
    o3d.core = core

    sys.modules["open3d"] = o3d
    sys.modules["open3d.geometry"] = geometry
    sys.modules["open3d.utility"] = utility
    sys.modules["open3d.io"] = io_mod
    sys.modules["open3d.t"] = t_mod
    sys.modules["open3d.t.geometry"] = t_geometry
    sys.modules["open3d.core"] = core
    _INSTALLED = True
