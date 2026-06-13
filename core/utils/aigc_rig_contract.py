"""
Avatar rig export contract (API-side gate).

Spec: ``OpenNexus3DStudio/CharacterStudio/docs/API_AVATAR_RIG_CONTRACT.md``

Mirrors Character Studio ``aigcRigContract.js`` codes so remote logs can show
``[API-Contract] PASS|FAIL`` from the DGX export path before the client loads a GLB.

glTF / three.js conventions: Y-up, character forward ≈ -Z.
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Failure codes — keep in sync with Character Studio `aigcRigContract.js`
CHARACTER_UPSIDE_DOWN = "character_upside_down"
CHARACTER_FACING_BACKWARDS = "character_facing_backwards"
MESH_BONE_VERTICAL_MISMATCH = "mesh_bone_vertical_mismatch"
HIPS_NOT_AT_MESH_TORSO = "hips_not_at_mesh_torso"
MISSING_SKINNED_MESH = "missing_skinned_mesh"
INSUFFICIENT_JOINTS = "insufficient_joints"

# Legacy aliases (deprecated strings, same checks)
CHARACTER_FACES_WRONG_WAY = CHARACTER_FACING_BACKWARDS
HIPS_TORSO_MISMATCH = HIPS_NOT_AT_MESH_TORSO
MISSING_SKIN = MISSING_SKINNED_MESH

CRITICAL_CODES = frozenset(
    {
        CHARACTER_UPSIDE_DOWN,
        CHARACTER_FACING_BACKWARDS,
        MISSING_SKINNED_MESH,
        INSUFFICIENT_JOINTS,
    }
)
ADVISORY_CODES = frozenset(
    {
        MESH_BONE_VERTICAL_MISMATCH,
        HIPS_NOT_AT_MESH_TORSO,
    }
)


@dataclass
class RigContractResult:
    passed: bool
    codes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"Invalid GLB: {path}")
    magic, _version, total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB: {path}")
    offset = 12
    while offset < min(total_len, len(data)):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk)
    raise ValueError(f"GLB JSON chunk missing: {path}")


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    mat = np.eye(4)
    if "translation" in node:
        mat[:3, 3] = node["translation"]
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        mat[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
    if "scale" in node:
        sx, sy, sz = node["scale"]
        mat[:3, :3] = mat[:3, :3] @ np.diag([sx, sy, sz])
    return mat


def _world_matrices(gltf: dict) -> list[np.ndarray]:
    nodes = gltf.get("nodes") or []
    parents = [-1] * len(nodes)
    for pi, parent in enumerate(nodes):
        for ci in parent.get("children") or []:
            parents[ci] = pi
    worlds: list[Optional[np.ndarray]] = [None] * len(nodes)

    def world(i: int) -> np.ndarray:
        if worlds[i] is not None:
            return worlds[i]
        local = _node_matrix(nodes[i])
        p = parents[i]
        worlds[i] = world(p) @ local if p >= 0 else local
        return worlds[i]

    for i in range(len(nodes)):
        world(i)
    return worlds  # type: ignore[return-value]


def _mesh_bounds_y(gltf: dict) -> tuple[float, float, float, np.ndarray]:
    meshes = gltf.get("meshes") or []
    if not meshes:
        return 0.0, 0.0, 0.0, np.zeros(3)
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    accessors = gltf.get("accessors") or []
    for mesh in meshes:
        for prim in mesh.get("primitives") or []:
            pos_idx = prim.get("attributes", {}).get("POSITION")
            if pos_idx is None:
                continue
            acc = accessors[pos_idx]
            mins.append(np.array(acc["min"], dtype=float))
            maxs.append(np.array(acc["max"], dtype=float))
    if not mins:
        return 0.0, 0.0, 0.0, np.zeros(3)
    mn = np.min(np.stack(mins), axis=0)
    mx = np.max(np.stack(maxs), axis=0)
    center = (mn + mx) / 2.0
    height = float(mx[1] - mn[1])
    return float(mn[1]), float(mx[1]), max(height, 1e-6), center


def _named_joint_positions(gltf: dict, worlds: list[np.ndarray]) -> dict[str, np.ndarray]:
    nodes = gltf.get("nodes") or []
    out: dict[str, np.ndarray] = {}
    for i, node in enumerate(nodes):
        name = node.get("name")
        if name:
            out[name] = worlds[i][:3, 3]
    return out


def validate_aigc_rigged_glb(
    path: str | Path,
    *,
    min_joints: int = 40,
    mesh_bone_center_max_fraction: float = 0.35,
    hips_height_min: float = 0.25,
    hips_height_max: float = 0.70,
    forward_dot_threshold: float = -0.5,
) -> RigContractResult:
    """
    Validate exported rigged GLB against the API avatar rig contract.

    ``passed`` is False when any *critical* code is present. ``hips_torso_mismatch``
    is advisory (logged but does not fail export by default).
    """
    path = Path(path)
    gltf = _read_glb_json(path)
    worlds = _world_matrices(gltf)
    nodes = gltf.get("nodes") or []
    skins = gltf.get("skins") or []

    codes: list[str] = []
    y_min, y_max, height, mesh_center = _mesh_bounds_y(gltf)
    named = _named_joint_positions(gltf, worlds)

    joint_count = len(skins[0].get("joints") or []) if skins else 0
    if not skins:
        codes.append(MISSING_SKINNED_MESH)
    elif joint_count < min_joints:
        codes.append(INSUFFICIENT_JOINTS)

    head = named.get("Head")
    foot = named.get("LeftFoot")
    if foot is None:
        foot = named.get("RightFoot")
    hips = named.get("Hips")

    head_y = float(head[1]) if head is not None else None
    foot_y = float(foot[1]) if foot is not None else None
    hips_y = float(hips[1]) if hips is not None else None

    if head_y is not None and foot_y is not None and head_y <= foot_y:
        codes.append(CHARACTER_UPSIDE_DOWN)

    if foot_y is not None and abs(foot_y - y_min) / height > 0.2:
        codes.append(MESH_BONE_VERTICAL_MISMATCH)

    if hips is not None and head is not None:
        spine = named.get("Spine2")
        if spine is None:
            spine = named.get("Spine1")
        if spine is None:
            spine = named.get("Spine")
        left = named.get("LeftShoulder")
        if left is None:
            left = named.get("LeftArm")
        right = named.get("RightShoulder")
        if right is None:
            right = named.get("RightArm")
        if spine is not None and left is not None and right is not None:
            up = spine - hips
            right_vec = right - left
            rnorm = float(np.linalg.norm(right_vec))
            unorm = float(np.linalg.norm(up))
            if rnorm > 1e-6 and unorm > 1e-6:
                right_vec = right_vec / rnorm
                up = up / unorm
                forward = np.cross(right_vec, up)
                forward[1] = 0.0
                fnorm = float(np.linalg.norm(forward))
                if fnorm > 1e-6:
                    forward = forward / fnorm
                    if float(forward[2]) > -forward_dot_threshold:
                        codes.append(CHARACTER_FACING_BACKWARDS)
        else:
            spine_h = head - hips
            spine_h[1] = 0.0
            norm = float(np.linalg.norm(spine_h))
            if norm > 1e-6:
                forward = spine_h / norm
                if float(forward[2]) > -forward_dot_threshold:
                    codes.append(CHARACTER_FACING_BACKWARDS)

    joint_positions: list[np.ndarray] = []
    if skins:
        for ji in skins[0].get("joints") or []:
            joint_positions.append(worlds[ji][:3, 3])
    bone_center = (
        np.mean(np.stack(joint_positions), axis=0) if joint_positions else mesh_center
    )
    mesh_bone_delta_y = float(bone_center[1] - mesh_center[1])
    if abs(mesh_bone_delta_y) / height > mesh_bone_center_max_fraction:
        if MESH_BONE_VERTICAL_MISMATCH not in codes:
            codes.append(MESH_BONE_VERTICAL_MISMATCH)

    hips_offset_from_mesh_center_y = None
    hips_height_fraction = None
    if hips_y is not None:
        hips_offset_from_mesh_center_y = hips_y - float(mesh_center[1])
        hips_height_fraction = (hips_y - y_min) / height
        if not (hips_height_min <= hips_height_fraction <= hips_height_max):
            codes.append(HIPS_NOT_AT_MESH_TORSO)

    metrics = {
        "meshMinY": y_min,
        "meshMaxY": y_max,
        "meshHeightY": height,
        "meshCenterY": float(mesh_center[1]),
        "boneCenterY": float(bone_center[1]),
        "meshBoneCenterDeltaY": mesh_bone_delta_y,
        "hipsY": hips_y,
        "headY": head_y,
        "footY": foot_y,
        "hipsOffsetFromMeshCenterY": hips_offset_from_mesh_center_y,
        "hipsHeightFraction": hips_height_fraction,
        "jointCount": joint_count,
    }

    critical = [c for c in codes if c in CRITICAL_CODES]
    advisory = [c for c in codes if c in ADVISORY_CODES]
    return RigContractResult(
        passed=len(critical) == 0,
        codes=codes,
        metrics={**metrics, "advisoryCodes": advisory},
    )


def format_contract_log(result: RigContractResult, *, job_id: str = "") -> str:
    status = "PASS" if result.passed else "FAIL"
    suffix = f" job_id={job_id}" if job_id else ""
    codes = ",".join(result.codes) if result.codes else "none"
    return f"[API-Contract] {status}{suffix} codes={codes} metrics={result.metrics}"
