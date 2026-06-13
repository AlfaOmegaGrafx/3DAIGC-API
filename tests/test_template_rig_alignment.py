"""Unit tests for template rig alignment math."""
from __future__ import annotations

import math

from core.utils.template_rig_alignment import (
    DEFAULT_ARMATURE_YAW_RAD,
    compute_armature_height_scale,
    compute_horizontal_shift,
    compute_uniform_height_scale,
    compute_vertical_shift,
)


def test_height_scale():
    assert compute_uniform_height_scale(1.0, 2.0) == 2.0
    assert compute_uniform_height_scale(1.8, 1.8) == 1.0


def test_feet_alignment_not_center():
    # Feet use min Y; center alignment uses midpoint — differ when bounds are asymmetric.
    target_min, target_max = 0.0, 2.0
    arm_min, arm_max = 0.8, 2.0  # same top, higher floor → feet shift down
    center_shift = ((target_min + target_max) / 2) - ((arm_min + arm_max) / 2)
    feet_shift = compute_vertical_shift(target_min, arm_min)
    assert feet_shift == -0.8
    assert math.isclose(center_shift, -0.4)
    assert feet_shift != center_shift


def test_horizontal_shift():
    dx, dz = compute_horizontal_shift((1.0, 2.0), (0.25, 1.5))
    assert dx == 0.75
    assert dz == 0.5


def test_armature_height_scale():
    assert compute_armature_height_scale(2.0, 1.0) == 0.5
    assert compute_armature_height_scale(1.0, 1.0) == 1.0


def test_default_yaw_is_neutral():
    assert DEFAULT_ARMATURE_YAW_RAD == 0.0
