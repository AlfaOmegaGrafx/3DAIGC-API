"""Pure helpers for humanoid template armature ↔ AIGC mesh alignment (Y-up)."""
from __future__ import annotations

import math
from typing import Tuple

# Optional extra yaw (radians around Blender Z). Facing is corrected adaptively in the
# Blender script; default 0 avoids a fixed π flip that exported backwards on AIGC meshes.
DEFAULT_ARMATURE_YAW_RAD = 0.0


def compute_uniform_height_scale(
    template_height: float, target_height: float, *, min_height: float = 1e-6
) -> float:
    th = max(float(template_height), min_height)
    tg = max(float(target_height), min_height)
    return tg / th


def compute_armature_height_scale(
    armature_height: float, target_height: float, *, min_height: float = 1e-6
) -> float:
    """Scale armature bone span to match target mesh height (not template proxy mesh)."""
    ah = max(float(armature_height), min_height)
    tg = max(float(target_height), min_height)
    return tg / ah


def compute_vertical_shift(target_min_y: float, armature_min_y: float) -> float:
    """Shift armature so feet (lowest Y) sit on target ground (min Y)."""
    return float(target_min_y) - float(armature_min_y)


def compute_horizontal_shift(
    target_center_xz: Tuple[float, float], armature_center_xz: Tuple[float, float]
) -> Tuple[float, float]:
    tx, tz = target_center_xz
    ax, az = armature_center_xz
    return tx - ax, tz - az
