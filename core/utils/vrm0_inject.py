"""Inject VRM 0.0 humanoid extensions into a Mixamo-skinned GLB for Appearance loadCustomTrait."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Mixamo / UniRig bone name → VRM 0 humanoid bone id
_MIXAMO_TO_VRM0: Dict[str, str] = {
    "mixamorig:hips": "hips",
    "hips": "hips",
    "j_bip_c_hips": "hips",
    "mixamorig:spine": "spine",
    "spine": "spine",
    "j_bip_c_spine": "spine",
    "mixamorig:spine1": "chest",
    "spine1": "chest",
    "j_bip_c_chest": "chest",
    "mixamorig:spine2": "upperChest",
    "spine2": "upperChest",
    "j_bip_c_upperchest": "upperChest",
    "mixamorig:neck": "neck",
    "neck": "neck",
    "j_bip_c_neck": "neck",
    "mixamorig:head": "head",
    "head": "head",
    "j_bip_c_head": "head",
    "mixamorig:leftupleg": "leftUpperLeg",
    "leftupleg": "leftUpperLeg",
    "j_bip_l_upperleg": "leftUpperLeg",
    "mixamorig:rightupleg": "rightUpperLeg",
    "rightupleg": "rightUpperLeg",
    "j_bip_r_upperleg": "rightUpperLeg",
    "mixamorig:leftleg": "leftLowerLeg",
    "leftleg": "leftLowerLeg",
    "j_bip_l_lowerleg": "leftLowerLeg",
    "mixamorig:rightleg": "rightLowerLeg",
    "rightleg": "rightLowerLeg",
    "j_bip_r_lowerleg": "rightLowerLeg",
    "mixamorig:leftfoot": "leftFoot",
    "leftfoot": "leftFoot",
    "j_bip_l_foot": "leftFoot",
    "mixamorig:rightfoot": "rightFoot",
    "rightfoot": "rightFoot",
    "j_bip_r_foot": "rightFoot",
    "mixamorig:lefttoebase": "leftToes",
    "lefttoebase": "leftToes",
    "j_bip_l_toebase": "leftToes",
    "mixamorig:righttoebase": "rightToes",
    "righttoebase": "rightToes",
    "j_bip_r_toebase": "rightToes",
    "mixamorig:leftshoulder": "leftShoulder",
    "leftshoulder": "leftShoulder",
    "j_bip_l_shoulder": "leftShoulder",
    "mixamorig:rightshoulder": "rightShoulder",
    "rightshoulder": "rightShoulder",
    "j_bip_r_shoulder": "rightShoulder",
    "mixamorig:leftarm": "leftUpperArm",
    "leftarm": "leftUpperArm",
    "j_bip_l_upperarm": "leftUpperArm",
    "mixamorig:rightarm": "rightUpperArm",
    "rightarm": "rightUpperArm",
    "j_bip_r_upperarm": "rightUpperArm",
    "mixamorig:leftforearm": "leftLowerArm",
    "leftforearm": "leftLowerArm",
    "j_bip_l_lowerarm": "leftLowerArm",
    "mixamorig:rightforearm": "rightLowerArm",
    "rightforearm": "rightLowerArm",
    "j_bip_r_lowerarm": "rightLowerArm",
    "mixamorig:lefthand": "leftHand",
    "lefthand": "leftHand",
    "j_bip_l_hand": "leftHand",
    "mixamorig:righthand": "rightHand",
    "righthand": "rightHand",
    "j_bip_r_hand": "rightHand",
}


def _read_glb_json(path: Path) -> Tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"Not a GLB: {path}")
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"GLB missing JSON chunk: {path}")
    json_bytes = data[20 : 20 + json_len]
    gltf = json.loads(json_bytes)
    rest = data[20 + json_len :]
    return gltf, rest


def _write_glb(path: Path, gltf: dict, rest_after_json: bytes) -> None:
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad
    total_len = 12 + 8 + len(json_bytes) + len(rest_after_json)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    json_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    path.write_bytes(header + json_header + json_bytes + rest_after_json)


def _normalize_bone_key(name: str) -> str:
    return str(name or "").strip().lower()


def inject_vrm0_humanoid_into_glb(
    input_glb: str,
    output_vrm: str,
    *,
    title: str = "Appearance Component",
    author: str = "OpenNexus3DStudio",
) -> str:
    """
    Write a ``.vrm`` that three-vrm can load by adding VRM 0.0 humanoid mapping
    onto an already Mixamo/J_Bip-skinned GLB.
    """
    src = Path(input_glb)
    dst = Path(output_vrm)
    gltf, rest = _read_glb_json(src)
    nodes: List[dict] = gltf.get("nodes") or []
    name_to_index = {
        _normalize_bone_key(n.get("name") or ""): i for i, n in enumerate(nodes)
    }

    human_bones = []
    used = set()
    for node_name, node_idx in name_to_index.items():
        vrm_bone = _MIXAMO_TO_VRM0.get(node_name)
        if not vrm_bone or vrm_bone in used:
            continue
        used.add(vrm_bone)
        human_bones.append(
            {
                "bone": vrm_bone,
                "node": node_idx,
                "useDefaultValues": True,
                "min": {"x": 0, "y": 0, "z": 0},
                "max": {"x": 0, "y": 0, "z": 0},
                "center": {"x": 0, "y": 0, "z": 0},
                "axisLength": 0,
            }
        )

    if "hips" not in used:
        raise RuntimeError(
            "Cannot inject VRM0: no hips bone found in skinned GLB "
            f"(nodes={list(name_to_index)[:12]})"
        )

    vrm_ext = {
        "exporterVersion": "OpenNexus appearance_component",
        "specVersion": "0.0",
        "meta": {
            "title": title,
            "version": "1.0",
            "author": author,
            "contactInformation": "",
            "reference": "",
            "allowedUserName": "Everyone",
            "violentUssageName": "Disallow",
            "sexualUssageName": "Disallow",
            "commercialUssageName": "Allow",
            "otherPermissionUrl": "",
            "licenseName": "Other",
        },
        "humanoid": {
            "humanBones": human_bones,
            "armStretch": 0.05,
            "legStretch": 0.05,
            "upperArmTwist": 0.5,
            "lowerArmTwist": 0.5,
            "upperLegTwist": 0.5,
            "lowerLegTwist": 0.5,
            "feetSpacing": 0,
            "hasTranslationDoF": False,
        },
        "firstPerson": {
            "firstPersonBone": next(
                (b["node"] for b in human_bones if b["bone"] == "head"),
                human_bones[0]["node"],
            ),
            "firstPersonBoneOffset": {"x": 0, "y": 0.06, "z": 0},
            "meshAnnotations": [],
            "lookAtTypeName": "Bone",
            "lookAtHorizontalInner": {"curve": [0, 0, 0, 1, 1, 1], "xRange": 90, "yRange": 10},
            "lookAtHorizontalOuter": {"curve": [0, 0, 0, 1, 1, 1], "xRange": 90, "yRange": 10},
            "lookAtVerticalDown": {"curve": [0, 0, 0, 1, 1, 1], "xRange": 90, "yRange": 10},
            "lookAtVerticalUp": {"curve": [0, 0, 0, 1, 1, 1], "xRange": 90, "yRange": 10},
        },
        "blendShapeMaster": {"blendShapeGroups": []},
        "secondaryAnimation": {"boneGroups": [], "colliderGroups": []},
        "materialProperties": [],
    }

    extensions = dict(gltf.get("extensions") or {})
    extensions["VRM"] = vrm_ext
    gltf["extensions"] = extensions
    used_ext = list(gltf.get("extensionsUsed") or [])
    if "VRM" not in used_ext:
        used_ext.append("VRM")
    gltf["extensionsUsed"] = used_ext

    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_glb(dst, gltf, rest)
    logger.info(
        "Injected VRM0 humanoid (%d bones) → %s",
        len(human_bones),
        dst,
    )
    return str(dst)
