"""Tests for humanoid VRM template (template.vrm)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.utils.format_utils import (
    apply_humanoid_template_rig,
    extract_vrm_skeleton_fbx,
)
from core.utils.humanoid_template import (
    assert_humanoid_template,
    get_template,
    template_paths_available,
    validate_humanoid_template,
)
from core.utils.humanoid_template_checks import validate_template_rigged_glb
from core.utils.unirig_glb_checks import analyze_glb
from core.utils.vrm_inspection import analyze_vrm
from utils.blender_runtime import find_blender_binary

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_VRM = REPO / "assets" / "example_autorig" / "template.vrm"
LEGACY_VRM = REPO / "assets" / "example_autorig" / "sifr2.vrm"
BIRD_TEXTURED = (
    REPO / "assets" / "example_autorig" / "regression" / "bird_trellis_textured.glb"
)
BLENDER = find_blender_binary() is not None
SKIP_BLENDER = not BLENDER
SKIP_REASON = "Requires Blender with VRM addon"


def _resolve_template_vrm() -> Path:
    if TEMPLATE_VRM.is_file():
        return TEMPLATE_VRM
    if LEGACY_VRM.is_file():
        return LEGACY_VRM
    return TEMPLATE_VRM


@pytest.mark.unit
def test_template_vrm_present_or_documented():
    vrm = _resolve_template_vrm()
    if not vrm.is_file():
        pytest.skip(f"Template VRM not on disk (expected {TEMPLATE_VRM})")


@pytest.mark.unit
def test_template_validation():
    if not template_paths_available("template") and not LEGACY_VRM.is_file():
        pytest.skip("template.vrm not available")
    errors = validate_humanoid_template("template")
    assert errors == [], errors
    analysis = assert_humanoid_template("template")
    assert analysis.morph_target_count >= 100
    assert analysis.blend_shape_group_count >= 100
    assert "blink" in analysis.blend_shape_presets
    assert analysis.has_vrm_humanoid


@pytest.mark.unit
def test_legacy_sifr2_alias():
    spec_template = get_template("template")
    spec_legacy = get_template("sifr2")
    assert spec_template.vrm_path == spec_legacy.vrm_path


@pytest.mark.unit
def test_template_manifest_matches_analysis():
    import json

    manifest_path = REPO / "assets/example_autorig/regression/template.json"
    if not manifest_path.is_file():
        manifest_path = REPO / "assets/example_autorig/regression/sifr2_template.json"
    manifest = json.loads(manifest_path.read_text())
    exp = manifest["expected"]
    vrm_path = _resolve_template_vrm()
    if not vrm_path.is_file():
        pytest.skip("template.vrm not available")
    vrm = analyze_vrm(vrm_path)
    assert vrm.morph_target_count == exp["morph_target_count"]
    assert vrm.blend_shape_group_count == exp["blend_shape_group_count"]
    assert vrm.skin_joint_count == exp["skin_joint_count"]


@pytest.mark.integration
@pytest.mark.skipif(SKIP_BLENDER or not template_paths_available(), reason=SKIP_REASON)
def test_extract_template_skeleton_fbx():
    spec = get_template("template")
    with tempfile.TemporaryDirectory(prefix="template_skel_") as tmp:
        out = Path(tmp) / "template.fbx"
        extract_vrm_skeleton_fbx(str(spec.vrm_path), str(out))
        assert out.is_file()
        assert out.stat().st_size > 100_000
        spec.skeleton_fbx_path.parent.mkdir(parents=True, exist_ok=True)
        if not spec.skeleton_fbx_path.is_file():
            import shutil

            shutil.copy2(out, spec.skeleton_fbx_path)


@pytest.mark.integration
@pytest.mark.skipif(
    SKIP_BLENDER or not template_paths_available() or not BIRD_TEXTURED.is_file(),
    reason=SKIP_REASON,
)
def test_apply_template_to_textured_mesh():
    """Bones-only template rig on bird GLB — verifies skin + textures, not blend shapes."""
    spec = get_template("template")
    with tempfile.TemporaryDirectory(prefix="template_rig_") as tmp:
        out = Path(tmp) / "rigged.glb"
        apply_humanoid_template_rig(str(spec.vrm_path), str(BIRD_TEXTURED), str(out))
        errors = validate_template_rigged_glb(BIRD_TEXTURED, out, min_joints=40)
        assert errors == [], errors
        rigged = analyze_glb(out)
        assert rigged.has_skin
        assert rigged.joint_counts[0] >= 40
