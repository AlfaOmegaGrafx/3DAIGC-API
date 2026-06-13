"""Regression tests for UniRig textured rig merge (bird fixture)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.utils.unirig_glb_checks import (
    UnirigMergeExpectations,
    analyze_glb,
    assert_unirig_merged_glb,
    bird_regression_fixture_paths,
    fixture_paths_available,
    validate_unirig_merged_glb,
)
from core.utils.format_utils import fbx_to_glb, merge_rigged_fbx_with_source_mesh
from utils.blender_runtime import find_blender_binary

REPO_ROOT = Path(__file__).resolve().parents[1]
BIRD = bird_regression_fixture_paths(REPO_ROOT)
BLENDER_AVAILABLE = find_blender_binary() is not None
FIXTURES_AVAILABLE = fixture_paths_available(REPO_ROOT)
SKIP_MERGE = not (BLENDER_AVAILABLE and FIXTURES_AVAILABLE)
SKIP_REASON = "Requires Blender + assets/example_autorig/regression/bird_trellis_* fixtures"


@pytest.mark.unit
def test_analyze_bird_source_fixture():
    if not BIRD["source_glb"].is_file():
        pytest.skip("bird_trellis_textured.glb not found")
    analysis = analyze_glb(BIRD["source_glb"])
    assert analysis.primary_vert_count == 5829
    assert analysis.has_images
    assert analysis.albedo_mean is not None
    assert 60 < analysis.albedo_mean < 95


@pytest.mark.unit
def test_validate_rejects_source_topology_skinning():
    """Using upload vert count for skinned output must fail generic checks."""
    source = BIRD["source_glb"]
    if not source.is_file():
        pytest.skip("bird fixture missing")
    errors = validate_unirig_merged_glb(source, source)
    assert any("skinning the upload mesh" in e for e in errors)
    assert any("no skin" in e.lower() for e in errors)


@pytest.mark.integration
@pytest.mark.skipif(SKIP_MERGE, reason=SKIP_REASON)
def test_bird_trellis_merge_bones_and_textures():
    """Full merge must match bird regression baselines (proxy verts + PBR)."""
    with tempfile.TemporaryDirectory(prefix="unirig_regression_") as tmp:
        out = Path(tmp) / "bird_merged.glb"
        ref = Path(tmp) / "bird_fbx_only.glb"
        fbx_to_glb(str(BIRD["rig_fbx"]), str(ref))

        merge_rigged_fbx_with_source_mesh(
            str(BIRD["source_glb"]),
            str(BIRD["rig_fbx"]),
            str(out),
        )

        assert_unirig_merged_glb(
            BIRD["source_glb"],
            out,
            expectations=UnirigMergeExpectations(),
            reference_fbx_glb_path=ref,
        )
