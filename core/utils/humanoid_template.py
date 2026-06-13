"""
Humanoid VRM template registry (master rig + blend shapes).

Reference template: ``template.vrm`` — VRM 0.x humanoid with facial tracking blend shapes.
Legacy id ``sifr2`` resolves to the same template for backward compatibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.utils.vrm_inspection import VrmAnalysis, analyze_vrm

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "assets" / "example_autorig"
REGRESSION_DIR = TEMPLATE_DIR / "regression"


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


_TEMPLATE_SPEC = HumanoidTemplateSpec(
    template_id="template",
    vrm_path=TEMPLATE_DIR / "template.vrm",
    skeleton_fbx_path=TEMPLATE_DIR / "skeleton" / "template.fbx",
    min_morph_targets=100,
    min_blend_shape_groups=100,
    min_skin_joints=60,
    min_human_bones=50,
    required_presets=("blink", "blink_l", "blink_r", "neutral"),
)

TEMPLATES: dict[str, HumanoidTemplateSpec] = {
    "template": _TEMPLATE_SPEC,
    "sifr2": _TEMPLATE_SPEC,  # deprecated alias
}


def get_template(template_id: str) -> HumanoidTemplateSpec:
    key = template_id.lower().strip()
    if key not in TEMPLATES:
        raise KeyError(
            f"Unknown humanoid template '{template_id}'. "
            f"Available: {sorted(set(TEMPLATES))}"
        )
    return TEMPLATES[key]


def template_paths_available(template_id: str = "template") -> bool:
    spec = get_template(template_id)
    return spec.vrm_path.is_file()


def skeleton_reference_available(template_id: str = "template") -> bool:
    spec = get_template(template_id)
    return spec.skeleton_fbx_path.is_file()


def validate_humanoid_template(
    template_id: str = "template",
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


def assert_humanoid_template(template_id: str = "template") -> VrmAnalysis:
    errors = validate_humanoid_template(template_id)
    if errors:
        raise ValueError("Humanoid template validation failed:\n  - " + "\n  - ".join(errors))
    return analyze_vrm(get_template(template_id).vrm_path)


def load_template_manifest(template_id: str = "template") -> dict:
    candidates = [
        REGRESSION_DIR / f"{template_id}_template.json",
        REGRESSION_DIR / "template.json",
        REGRESSION_DIR / f"{template_id}.json",
        REGRESSION_DIR / "sifr2_template.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}
