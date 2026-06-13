"""Parse and inspect VRM (glTF + VRM extension) files without Blender."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class VrmAnalysis:
    path: Path
    spec: str  # "0.x" or "1.0" or "unknown"
    mesh_count: int
    node_count: int
    skin_count: int
    morph_target_count: int
    blend_shape_group_count: int
    human_bone_count: int
    skin_joint_count: int
    total_vertices: int
    skin_joint_names: tuple[str, ...]
    blend_shape_presets: tuple[str, ...]
    has_vrm_humanoid: bool

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / (1024 * 1024)


def _read_gltf_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"Invalid GLB: {path}")
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB/VRM file: {path}")
    json_len = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + json_len])


def analyze_vrm(path: str | Path) -> VrmAnalysis:
    path = Path(path)
    js = _read_gltf_json(path)
    nodes = js.get("nodes") or []
    meshes = js.get("meshes") or []
    skins = js.get("skins") or []

    morphs = 0
    verts = 0
    for mesh in meshes:
        for prim in mesh.get("primitives") or []:
            morphs += len(prim.get("targets") or [])
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None:
                verts += js["accessors"][pos]["count"]

    ext = js.get("extensions") or {}
    vrm0 = ext.get("VRM")
    vrm1 = ext.get("VRMC_vrm")
    spec = "unknown"
    blend_groups = 0
    human_bones = 0
    presets: list[str] = []

    if vrm0:
        spec = "0.x"
        bsm = vrm0.get("blendShapeMaster") or {}
        groups = bsm.get("blendShapeGroups") or []
        blend_groups = len(groups)
        for g in groups:
            presets.append(g.get("presetName") or g.get("name") or "unknown")
        hb = (vrm0.get("humanoid") or {}).get("humanBones") or []
        human_bones = len(hb)
    elif vrm1:
        spec = "1.0"
        expressions = vrm1.get("expressions") or {}
        blend_groups = sum(
            1 for k in ("preset", "custom") for _ in (expressions.get(k) or {}).values()
        )
        presets = list((expressions.get("preset") or {}).keys()) + list(
            (expressions.get("custom") or {}).keys()
        )
        human_bones = len((vrm1.get("humanoid") or {}).get("humanBones") or {})

    joint_names: list[str] = []
    if skins:
        joint_names = [
            nodes[i].get("name") or f"node_{i}" for i in skins[0].get("joints") or []
        ]

    return VrmAnalysis(
        path=path,
        spec=spec,
        mesh_count=len(meshes),
        node_count=len(nodes),
        skin_count=len(skins),
        morph_target_count=morphs,
        blend_shape_group_count=blend_groups,
        human_bone_count=human_bones,
        skin_joint_count=len(joint_names),
        total_vertices=verts,
        skin_joint_names=tuple(joint_names),
        blend_shape_presets=tuple(sorted(set(presets))),
        has_vrm_humanoid=human_bones > 0,
    )
