"""
Humanoid VRM template registry (master rig + blend shapes).

Product template id: ``humanoid`` — operator-local morph-head VRM + humanoid armature.
Resolution order (first existing file wins):

1. ``HUMANOID_TEMPLATE_VRM`` env (absolute path)
2. ``assets/example_autorig/humanoid_template.vrm`` (gitignored; usually a symlink)
3. Operator moat wrap ``assets/moat/ict/template_ict.vrm`` (gitignored local stack)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.utils.vrm_inspection import VrmAnalysis, analyze_vrm

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "assets" / "example_autorig"
REGRESSION_DIR = TEMPLATE_DIR / "regression"
MOAT_WRAP_VRM = REPO_ROOT / "assets" / "moat" / "ict" / "template_ict.vrm"
MOAT_SKELETON_FBX = REPO_ROOT / "assets" / "moat" / "ict" / "skeleton" / "template.fbx"

# Deprecated request ids → product default
_DEPRECATED_TEMPLATE_IDS = frozenset({"template", "sifr2", "ict"})

_DEFAULT_TEMPLATE_ID = "humanoid"


def _resolve_humanoid_vrm_path() -> Path:
    env = (os.environ.get("HUMANOID_TEMPLATE_VRM") or "").strip()
    if env:
        return Path(env)
    local = TEMPLATE_DIR / "humanoid_template.vrm"
    if local.is_file():
        return local
    if MOAT_WRAP_VRM.is_file():
        return MOAT_WRAP_VRM
    return local


def _resolve_humanoid_skeleton_fbx() -> Path:
    env = (os.environ.get("HUMANOID_TEMPLATE_SKELETON_FBX") or "").strip()
    if env:
        return Path(env)
    local = TEMPLATE_DIR / "skeleton" / "template.fbx"
    if local.is_file():
        return local
    if MOAT_SKELETON_FBX.is_file():
        return MOAT_SKELETON_FBX
    return local


@dataclass(frozen=True)
class HumanoidTemplateSpec:
    template_id: str
    vrm_path: Path
    skeleton_fbx_path: Path
    min_morph_targets: int = 50
    min_blend_shape_groups: int = 50
    min_skin_joints: int = 50
    min_human_bones: int = 40
    required_presets: tuple[str, ...] = ("blink", "neutral")


def _humanoid_spec() -> HumanoidTemplateSpec:
    return HumanoidTemplateSpec(
        template_id=_DEFAULT_TEMPLATE_ID,
        vrm_path=_resolve_humanoid_vrm_path(),
        skeleton_fbx_path=_resolve_humanoid_skeleton_fbx(),
        min_morph_targets=50,
        min_blend_shape_groups=40,
        min_skin_joints=40,
        min_human_bones=40,
        required_presets=("blink", "blink_l", "blink_r", "neutral"),
    )


def normalize_humanoid_template_id(template_id: str | None) -> str:
    """Map deprecated ids to product default ``humanoid``."""
    key = (template_id or _DEFAULT_TEMPLATE_ID).lower().strip()
    if key in _DEPRECATED_TEMPLATE_IDS:
        return _DEFAULT_TEMPLATE_ID
    return key or _DEFAULT_TEMPLATE_ID


def get_template(template_id: str = _DEFAULT_TEMPLATE_ID) -> HumanoidTemplateSpec:
    key = normalize_humanoid_template_id(template_id)
    if key != _DEFAULT_TEMPLATE_ID:
        raise KeyError(
            f"Unknown humanoid template '{template_id}'. "
            f"Available: [{_DEFAULT_TEMPLATE_ID!r}]"
        )
    return _humanoid_spec()


def template_paths_available(template_id: str = _DEFAULT_TEMPLATE_ID) -> bool:
    return get_template(template_id).vrm_path.is_file()


def resolve_default_humanoid_template_id() -> str:
    if template_paths_available(_DEFAULT_TEMPLATE_ID):
        return _DEFAULT_TEMPLATE_ID
    raise FileNotFoundError(
        "humanoid_template.vrm missing — set HUMANOID_TEMPLATE_VRM, place "
        "assets/example_autorig/humanoid_template.vrm, or build moat wrap "
        "assets/moat/ict/template_ict.vrm"
    )


def skeleton_reference_available(template_id: str = _DEFAULT_TEMPLATE_ID) -> bool:
    return get_template(template_id).skeleton_fbx_path.is_file()


def validate_humanoid_template(
    template_id: str = _DEFAULT_TEMPLATE_ID,
    analysis: Optional[VrmAnalysis] = None,
) -> list[str]:
    spec = get_template(template_id)
    errors: list[str] = []

    if not spec.vrm_path.is_file():
        errors.append(f"Template VRM missing: {spec.vrm_path}")
        return errors

    vrm = analysis or analyze_vrm(spec.vrm_path)

    if vrm.spec != "0.x":
        errors.append(f"[{template_id}] Expected VRM 0.x, got {vrm.spec}")
    if not vrm.has_vrm_humanoid:
        errors.append(f"[{template_id}] Missing VRM humanoid bone mapping")
    if vrm.morph_target_count < spec.min_morph_targets:
        errors.append(
            f"[{template_id}] morph_targets {vrm.morph_target_count} "
            f"< min {spec.min_morph_targets}"
        )
    if vrm.blend_shape_group_count < spec.min_blend_shape_groups:
        errors.append(
            f"[{template_id}] blendShapeGroups {vrm.blend_shape_group_count} "
            f"< min {spec.min_blend_shape_groups}"
        )
    if vrm.skin_joint_count < spec.min_skin_joints:
        errors.append(
            f"[{template_id}] skin joints {vrm.skin_joint_count} "
            f"< min {spec.min_skin_joints}"
        )
    if vrm.human_bone_count < spec.min_human_bones:
        errors.append(
            f"[{template_id}] humanBones {vrm.human_bone_count} "
            f"< min {spec.min_human_bones}"
        )
    preset_set = set(vrm.blend_shape_presets)
    for preset in spec.required_presets:
        if preset not in preset_set:
            errors.append(f"[{template_id}] Missing blend shape preset '{preset}'")

    return errors


def assert_humanoid_template(template_id: str = _DEFAULT_TEMPLATE_ID) -> VrmAnalysis:
    errors = validate_humanoid_template(template_id)
    if errors:
        raise ValueError("Humanoid template validation failed:\n  - " + "\n  - ".join(errors))
    return analyze_vrm(get_template(template_id).vrm_path)


def load_template_manifest(template_id: str = _DEFAULT_TEMPLATE_ID) -> dict:
    key = normalize_humanoid_template_id(template_id)
    candidates = [
        REGRESSION_DIR / f"{key}_template.json",
        REGRESSION_DIR / f"{key}.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}
