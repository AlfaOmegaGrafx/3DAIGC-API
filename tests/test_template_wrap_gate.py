"""Phase 5 head-stitch helpers + humanoid wrap gate."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.routers.auto_rigging import AutoRigRequest, reject_non_humanoid_template_wrap
from core.utils.format_utils import count_glb_morph_targets


def _write_minimal_glb(path: Path, *, morph_targets: int = 0) -> None:
    """Write a tiny GLB with optional morph target stubs (no real buffers needed for count)."""
    mesh = {
        "name": "Head",
        "primitives": [
            {
                "attributes": {"POSITION": 0},
                "targets": [{"POSITION": 1} for _ in range(morph_targets)],
            }
        ],
    }
    if morph_targets:
        mesh["weights"] = [0.0] * morph_targets
    gltf = {
        "asset": {"version": "2.0"},
        "meshes": [mesh],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # Pad JSON chunk to 4-byte alignment with spaces
    pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * pad
    total = 12 + 8 + len(json_bytes)
    with path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)


def test_count_glb_morph_targets_reads_primitive_targets(tmp_path: Path):
    glb = tmp_path / "with_morphs.glb"
    _write_minimal_glb(glb, morph_targets=12)
    assert count_glb_morph_targets(str(glb)) == 12


def test_count_glb_morph_targets_zero_without_targets(tmp_path: Path):
    glb = tmp_path / "no_morphs.glb"
    _write_minimal_glb(glb, morph_targets=0)
    assert count_glb_morph_targets(str(glb)) == 0


def test_auto_rig_request_accepts_vrm_output_format():
    req = AutoRigRequest(
        mesh_path="assets/example_mesh/foo.glb",
        rig_mode="template",
        model_preference="unirig_auto_rig",
        output_format="vrm",
    )
    assert req.output_format == "vrm"


def test_auto_rig_request_accepts_template_wrap():
    req = AutoRigRequest(
        mesh_path="assets/example_mesh/foo.glb",
        rig_mode="template_wrap",
        model_preference="unirig_auto_rig",
        output_format="vrm",
    )
    assert req.rig_mode == "template_wrap"
    assert req.output_format == "vrm"


def test_auto_rig_request_accepts_likeness_image_file_id():
    req = AutoRigRequest(
        mesh_path="assets/example_mesh/foo.glb",
        rig_mode="template_wrap",
        model_preference="unirig_auto_rig",
        output_format="vrm",
        likeness_image_file_id="file_abc123",
        model_parameters={"face_likeness": True, "likeness_source": "selfie"},
    )
    assert req.likeness_image_file_id == "file_abc123"
    assert req.model_parameters["likeness_source"] == "selfie"


def test_template_wrap_humanoid_only_note_in_modes():
    with pytest.raises(ValidationError):
        AutoRigRequest(
            mesh_path="assets/example_mesh/foo.glb",
            rig_mode="template_wrap",
            output_format="obj",
        )


def test_reject_template_wrap_with_creature_template():
    err = reject_non_humanoid_template_wrap(
        "template_wrap", "creature_template_auto_rig"
    )
    assert err is not None
    assert "humanoid-only" in err
    assert "creatureFaceRetarget" in err


def test_reject_template_wrap_with_skintokens():
    err = reject_non_humanoid_template_wrap("template_wrap", "skintokens_auto_rig")
    assert err is not None
    assert "SkinTokens" in err


def test_allow_template_wrap_with_unirig():
    assert reject_non_humanoid_template_wrap("template_wrap", "unirig_auto_rig") is None
    assert reject_non_humanoid_template_wrap("template_wrap", None) is None
    assert reject_non_humanoid_template_wrap("template", "skintokens_auto_rig") is None


def test_auto_rig_request_accepts_mesh_job_id():
    req = AutoRigRequest(
        mesh_job_id="abc-123",
        rig_mode="appearance_component",
        output_format="glb",
    )
    assert req.mesh_job_id == "abc-123"
    assert req.mesh_path is None
    assert req.mesh_file_id is None


def test_auto_rig_request_rejects_mesh_job_id_with_mesh_path():
    with pytest.raises(ValidationError):
        AutoRigRequest(
            mesh_path="assets/example_mesh/foo.glb",
            mesh_job_id="abc-123",
            output_format="glb",
        )


def test_auto_rig_request_still_requires_a_mesh_source():
    with pytest.raises(ValidationError):
        AutoRigRequest(
            rig_mode="appearance_component",
            output_format="glb",
        )
