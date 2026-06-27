"""Shim legacy utils3d.torch names used by TRELLIS / VoxHammer thirdparty code."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch


def _coerce_scalar_tensor(
    value: float | int | torch.Tensor,
    *,
    ref: torch.Tensor,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def _matrix_to_device(
    value: torch.Tensor | None,
    *,
    ref: torch.Tensor,
) -> torch.Tensor | None:
    if value is None or not torch.is_tensor(value):
        return value
    return value.to(device=ref.device, dtype=torch.float32)


def _group(
    values: torch.Tensor,
    required_group_size: Optional[int] = None,
    return_values: bool = False,
) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
    sorted_values, indices = torch.sort(values)
    nondupe = torch.cat(
        [
            torch.tensor([True], dtype=torch.bool, device=values.device),
            sorted_values[1:] != sorted_values[:-1],
        ]
    )
    nondupe_indices = torch.cumsum(nondupe, dim=0) - 1
    counts = torch.bincount(nondupe_indices)
    if required_group_size is None:
        groups = torch.split(indices, counts.tolist())
        if return_values:
            group_values = sorted_values[nondupe]
            return groups, group_values
        return groups, None

    counts = counts[nondupe_indices]
    groups = indices[counts == required_group_size].reshape(-1, required_group_size)
    if return_values:
        group_values = sorted_values[nondupe][counts[nondupe] == required_group_size]
        return groups, group_values
    return groups, None


def _compute_edges(faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tri_count = faces.shape[0]
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        dim=0,
    )
    edges = torch.sort(edges, dim=1).values
    edges, inv_map, counts = torch.unique(
        edges, return_inverse=True, return_counts=True, dim=0
    )
    face2edge = inv_map.view(3, tri_count).T
    return edges, face2edge, counts


def _compute_connected_components(
    faces: torch.Tensor,
    edges: torch.Tensor | None = None,
    face2edge: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    tri_count = faces.shape[0]
    if edges is None or face2edge is None:
        edges, face2edge, _ = _compute_edges(faces)
    edge_count = edges.shape[0]

    labels = torch.arange(tri_count, dtype=torch.int32, device=faces.device)
    while True:
        edge_labels = torch.scatter_reduce(
            torch.zeros(edge_count, dtype=torch.int32, device=faces.device),
            0,
            face2edge.flatten().long(),
            labels.view(-1, 1).expand(-1, 3).flatten(),
            reduce="amin",
            include_self=False,
        )
        new_labels = torch.min(edge_labels[face2edge], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components, _ = _group(labels)
    return list(components)


def _compute_edge_connected_components(edges: torch.Tensor) -> List[torch.Tensor]:
    edge_count = edges.shape[0]
    verts, edges = torch.unique(edges.flatten(), return_inverse=True)
    edges = edges.view(-1, 2)
    vertex_count = verts.shape[0]

    labels = torch.arange(edge_count, dtype=torch.int32, device=edges.device)
    while True:
        vertex_labels = torch.scatter_reduce(
            torch.zeros(vertex_count, dtype=torch.int32, device=edges.device),
            0,
            edges.flatten().long(),
            labels.view(-1, 1).expand(-1, 2).flatten(),
            reduce="amin",
            include_self=False,
        )
        new_labels = torch.min(vertex_labels[edges], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components, _ = _group(labels)
    return list(components)


def _compute_dual_graph(
    face2edge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_edge_indices = face2edge.flatten()
    dual_edges, dual_edge2edge = _group(
        all_edge_indices, required_group_size=2, return_values=True
    )
    dual_edges = dual_edges // face2edge.shape[1]
    return dual_edges, dual_edge2edge


def ensure_utils3d_torch_compat() -> None:
    """Patch utils3d.torch with legacy names removed/renamed in utils3d 1.x."""
    try:
        import utils3d.torch as ut
        from utils3d.torch import mesh as mesh_mod
        from utils3d.torch import transforms as tx
        from utils3d.torch.rasterization import rasterize_triangles
    except ImportError:
        return

    if not hasattr(ut, "perspective_from_fov_xy") and hasattr(tx, "perspective_from_fov"):

        def perspective_from_fov_xy(fov_x, fov_y, near, far):
            ref = fov_x if torch.is_tensor(fov_x) else fov_y
            if torch.is_tensor(ref):
                near_t = _coerce_scalar_tensor(near, ref=ref)
                far_t = _coerce_scalar_tensor(far, ref=ref)
            else:
                near_t = near
                far_t = far
            return tx.perspective_from_fov(
                fov_x=fov_x, fov_y=fov_y, near=near_t, far=far_t
            )

        ut.perspective_from_fov_xy = perspective_from_fov_xy  # type: ignore[attr-defined]

    if not hasattr(ut, "intrinsics_from_fov_xy") and hasattr(tx, "intrinsics_from_fov"):

        def intrinsics_from_fov_xy(fov_x, fov_y):
            return tx.intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)

        ut.intrinsics_from_fov_xy = intrinsics_from_fov_xy  # type: ignore[attr-defined]

    # utils3d 1.x intrinsics_to_perspective builds CPU helper mats even for CUDA intrinsics.
    def intrinsics_to_perspective(intrinsics, near, far):
        device, dtype = intrinsics.device, intrinsics.dtype
        batch_shape = intrinsics.shape[:-2]
        near_t = _coerce_scalar_tensor(near, ref=intrinsics).to(dtype=dtype)
        far_t = _coerce_scalar_tensor(far, ref=intrinsics).to(dtype=dtype)
        flip = torch.tensor(
            [[2, 0, -1], [0, -2, 1], [0, 0, 1]], dtype=dtype, device=device
        )
        diag = torch.diag(torch.tensor([1, -1, -1], dtype=dtype, device=device))
        m = flip @ intrinsics @ diag
        ratio = near_t / far_t
        return torch.cat(
            [
                torch.cat(
                    [
                        m[..., :2, :],
                        torch.zeros((*batch_shape, 2, 1), dtype=dtype, device=device),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        torch.zeros((*batch_shape, 1, 2), dtype=dtype, device=device),
                        ((ratio + 1) / (ratio - 1))[..., None, None],
                        (2.0 * near_t / (ratio - 1))[..., None, None],
                    ],
                    dim=-1,
                ),
                torch.tensor([0.0, 0.0, -1.0, 0.0], dtype=dtype, device=device).expand(
                    *batch_shape, 1, 4
                ),
            ],
            dim=-2,
        )

    ut.intrinsics_to_perspective = intrinsics_to_perspective  # type: ignore[attr-defined]

    if not hasattr(ut, "rasterize_triangle_faces"):

        def rasterize_triangle_faces(
            ctx,
            vertices: torch.Tensor,
            faces: torch.Tensor,
            width: int,
            height: int,
            **kwargs,
        ):
            if "perspective" in kwargs and "projection" not in kwargs:
                kwargs["projection"] = kwargs.pop("perspective")
            for key in ("view", "projection", "model"):
                if key in kwargs:
                    kwargs[key] = _matrix_to_device(kwargs[key], ref=vertices)
            return rasterize_triangles(
                ctx,
                int(width),
                int(height),
                vertices=vertices,
                faces=faces,
                **kwargs,
            )

        ut.rasterize_triangle_faces = rasterize_triangle_faces  # type: ignore[attr-defined]

    if not hasattr(ut, "compute_edges"):
        ut.compute_edges = _compute_edges  # type: ignore[attr-defined]

    if not hasattr(ut, "compute_connected_components"):
        ut.compute_connected_components = _compute_connected_components  # type: ignore[attr-defined]

    if not hasattr(ut, "compute_edge_connected_components"):
        ut.compute_edge_connected_components = _compute_edge_connected_components  # type: ignore[attr-defined]

    if not hasattr(ut, "compute_dual_graph"):
        ut.compute_dual_graph = _compute_dual_graph  # type: ignore[attr-defined]

    if not hasattr(ut, "remove_unreferenced_vertices") and hasattr(
        mesh_mod, "remove_unused_vertices"
    ):
        ut.remove_unreferenced_vertices = mesh_mod.remove_unused_vertices  # type: ignore[attr-defined]
