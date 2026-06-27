"""
Convert Kimodo SOMA NPZ motion to OpenNexus3DStudio studio_motion.json for VRM playback.

studio_motion.json targets VRM humanoid bone names; the client resolves normalized
bone track names via resolveVrmBoneTrackName (see vrmMixamoPlaybackGuard).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SOMASkeleton77 bone order (kimodo/skeleton/definitions.py) → VRM humanoid names.
SOMA_NAME_TO_VRM_HUMANOID: Dict[str, str] = {
    "Hips": "hips",
    "Spine1": "spine",
    "Spine2": "chest",
    "Chest": "upperChest",
    "Neck1": "neck",
    "Neck2": "neck",
    "Head": "head",
    "LeftShoulder": "leftShoulder",
    "LeftArm": "leftUpperArm",
    "LeftForeArm": "leftLowerArm",
    "LeftHand": "leftHand",
    "RightShoulder": "rightShoulder",
    "RightArm": "rightUpperArm",
    "RightForeArm": "rightLowerArm",
    "RightHand": "rightHand",
    "LeftLeg": "leftUpperLeg",
    "LeftShin": "leftLowerLeg",
    "LeftFoot": "leftFoot",
    "LeftToeBase": "leftToes",
    "RightLeg": "rightUpperLeg",
    "RightShin": "rightLowerLeg",
    "RightFoot": "rightFoot",
    "RightToeBase": "rightToes",
    "LeftHandThumb1": "leftThumbMetacarpal",
    "LeftHandThumb2": "leftThumbProximal",
    "LeftHandThumb3": "leftThumbDistal",
    "LeftHandIndex1": "leftIndexProximal",
    "LeftHandIndex2": "leftIndexIntermediate",
    "LeftHandIndex3": "leftIndexDistal",
    "LeftHandMiddle1": "leftMiddleProximal",
    "LeftHandMiddle2": "leftMiddleIntermediate",
    "LeftHandMiddle3": "leftMiddleDistal",
    "LeftHandRing1": "leftRingProximal",
    "LeftHandRing2": "leftRingIntermediate",
    "LeftHandRing3": "leftRingDistal",
    "LeftHandPinky1": "leftLittleProximal",
    "LeftHandPinky2": "leftLittleIntermediate",
    "LeftHandPinky3": "leftLittleDistal",
    "RightHandThumb1": "rightThumbMetacarpal",
    "RightHandThumb2": "rightThumbProximal",
    "RightHandThumb3": "rightThumbDistal",
    "RightHandIndex1": "rightIndexProximal",
    "RightHandIndex2": "rightIndexIntermediate",
    "RightHandIndex3": "rightIndexDistal",
    "RightHandMiddle1": "rightMiddleProximal",
    "RightHandMiddle2": "rightMiddleIntermediate",
    "RightHandMiddle3": "rightMiddleDistal",
    "RightHandRing1": "rightRingProximal",
    "RightHandRing2": "rightRingIntermediate",
    "RightHandRing3": "rightRingDistal",
    "RightHandPinky1": "rightLittleProximal",
    "RightHandPinky2": "rightLittleIntermediate",
    "RightHandPinky3": "rightLittleDistal",
}

SOMA_BONE_ORDER: List[str] = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]


def _mat3_to_quat_xyzw(mat: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix 3x3 → quaternion (x, y, z, w)."""
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def soma_npz_to_studio_motion(
    npz_path: Path,
    out_json_path: Path,
    *,
    fps: float = 30.0,
    motion_name: str = "kimodo",
) -> Dict[str, Any]:
    """Write studio_motion.json from Kimodo NPZ local_rot_mats."""
    data = np.load(str(npz_path))
    if "local_rot_mats" not in data:
        raise ValueError(f"NPZ missing local_rot_mats: {npz_path}")

    local_rots = np.asarray(data["local_rot_mats"], dtype=np.float64)
    if local_rots.ndim != 4 or local_rots.shape[-2:] != (3, 3):
        raise ValueError(f"Unexpected local_rot_mats shape: {local_rots.shape}")

    # Kimodo batch dim: [samples, T, J, 3, 3] or [T, J, 3, 3]
    if local_rots.ndim == 5:
        local_rots = local_rots[0]
    t_frames, n_joints = local_rots.shape[0], local_rots.shape[1]

    soma_names = SOMA_BONE_ORDER
    if n_joints != len(soma_names):
        logger.warning(
            "Joint count %s != somaskel77 (%s); mapping by index where names exist",
            n_joints,
            len(soma_names),
        )

    times = [i / fps for i in range(t_frames)]
    tracks: List[Dict[str, Any]] = []
    seen_vrm_bones: set[str] = set()

    for j_idx in range(min(n_joints, len(soma_names))):
        soma_name = soma_names[j_idx]
        vrm_bone = SOMA_NAME_TO_VRM_HUMANOID.get(soma_name)
        if not vrm_bone or vrm_bone in seen_vrm_bones:
            continue
        seen_vrm_bones.add(vrm_bone)

        quats: List[float] = []
        for t in range(t_frames):
            x, y, z, w = _mat3_to_quat_xyzw(local_rots[t, j_idx])
            quats.extend([x, y, z, w])

        tracks.append({"bone": vrm_bone, "times": times, "quaternions": quats})

    if not tracks:
        raise ValueError("No SOMA→VRM tracks produced from NPZ")

    payload: Dict[str, Any] = {
        "version": 1,
        "format": "studio_motion",
        "name": motion_name,
        "fps": fps,
        "duration": t_frames / fps,
        "tracks": tracks,
        "source": "kimodo_soma_npz",
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload
