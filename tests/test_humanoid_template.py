"""Tests for humanoid VRM template (template_ict.vrm)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.utils.format_utils import (
    apply_humanoid_template_rig,
    apply_humanoid_template_wrap,
    count_glb_morph_targets,
    extract_vrm_skeleton_fbx,
)
from core.utils.humanoid_template import (
    assert_humanoid_template,
    get_template,
    normalize_humanoid_template_id,
    template_paths_available,
    validate_humanoid_template,
)
from core.utils.humanoid_template_checks import validate_template_rigged_glb
from core.utils.unirig_glb_checks import analyze_glb
from core.utils.vrm_inspection import analyze_vrm
from utils.blender_runtime import find_blender_binary

REPO = Path(__file__).resolve().parents[1]
ICT_VRM = REPO / "assets" / "example_autorig" / "template_ict.vrm"
BIRD_TEXTURED = (
    REPO / "assets" / "example_autorig" / "regression" / "bird_trellis_textured.glb"
)
BLENDER = find_blender_binary() is not None
SKIP_BLENDER = not BLENDER
SKIP_REASON = "Requires Blender with VRM addon"


@pytest.mark.unit
def test_deprecated_template_ids_normalize_to_ict():
    assert normalize_humanoid_template_id("template") == "ict"
    assert normalize_humanoid_template_id("sifr2") == "ict"
    assert normalize_humanoid_template_id("ict") == "ict"
    assert get_template("template").template_id == "ict"


@pytest.mark.unit
def test_ict_template_validation():
    """MIT ICT-FaceKit morph head template for Body+Cloth wrap."""
    if not template_paths_available("ict"):
        pytest.skip(
            "template_ict.vrm not available — set HUMANOID_TEMPLATE_VRM or place assets/example_autorig/template_ict.vrm"
        )
    errors = validate_humanoid_template("ict")
    assert errors == [], errors
    analysis = assert_humanoid_template("ict")
    assert analysis.morph_target_count >= 50
    assert "blink" in analysis.blend_shape_presets
    assert analysis.has_vrm_humanoid


@pytest.mark.unit
def test_ict_manifest_matches_analysis():
    manifest_path = REPO / "assets/example_autorig/regression/ict_template.json"
    if not manifest_path.is_file():
        pytest.skip("ict_template.json missing")
    if not ICT_VRM.is_file():
        pytest.skip("template_ict.vrm not available")
    manifest = json.loads(manifest_path.read_text())
    exp = manifest["expected"]
    vrm = analyze_vrm(ICT_VRM)
    assert vrm.morph_target_count >= exp["morph_target_count"]
    assert vrm.blend_shape_group_count >= exp["blend_shape_group_count"]


@pytest.mark.integration
@pytest.mark.skipif(SKIP_BLENDER or not template_paths_available("ict"), reason=SKIP_REASON)
def test_extract_ict_skeleton_fbx():
    spec = get_template("ict")
    with tempfile.TemporaryDirectory(prefix="ict_skel_") as tmp:
        out = Path(tmp) / "template.fbx"
        extract_vrm_skeleton_fbx(str(spec.vrm_path), str(out))
        assert out.is_file()
        assert out.stat().st_size > 100_000


@pytest.mark.integration
@pytest.mark.skipif(
    SKIP_BLENDER or not template_paths_available("ict") or not BIRD_TEXTURED.is_file(),
    reason=SKIP_REASON,
)
def test_apply_ict_template_rig_to_textured_mesh():
    """Bones-only ICT template rig on bird GLB — verifies skin + textures."""
    spec = get_template("ict")
    with tempfile.TemporaryDirectory(prefix="ict_rig_") as tmp:
        out = Path(tmp) / "rigged.glb"
        apply_humanoid_template_rig(str(spec.vrm_path), str(BIRD_TEXTURED), str(out))
        errors = validate_template_rigged_glb(BIRD_TEXTURED, out, min_joints=40)
        assert errors == [], errors
        rigged = analyze_glb(out)
        assert rigged.has_skin
        assert rigged.joint_counts[0] >= 40


@pytest.mark.integration
@pytest.mark.skipif(
    SKIP_BLENDER or not template_paths_available("ict") or not BIRD_TEXTURED.is_file(),
    reason=SKIP_REASON,
)
def test_apply_ict_template_wrap_keeps_morphs():
    """ICT AvatarHead morphs survive wrap stitch onto stand-in body."""
    spec = get_template("ict")
    with tempfile.TemporaryDirectory(prefix="ict_wrap_") as tmp:
        out = Path(tmp) / "stitched.glb"
        path, validation = apply_humanoid_template_wrap(
            str(spec.vrm_path), str(BIRD_TEXTURED), str(out)
        )
        assert Path(path).is_file()
        assert validation.get("wrap_status") == "head_stitch"
        assert validation.get("blend_shapes_on_generated_mesh") is True
        morphs = validation.get("morph_target_count") or count_glb_morph_targets(path)
        assert morphs >= 8, f"expected ICT face morphs, got {morphs}"
        rigged = analyze_glb(out)
        assert rigged.has_skin
