"""Tests for xr_voice_commands.yaml intent matching."""

from __future__ import annotations

from pathlib import Path

import pytest

from daigc_mcp.xr_voice_commands import load_voice_command_database

_YAML = Path(__file__).resolve().parents[1] / "yaml" / "xr_voice_commands.yaml"


@pytest.fixture
def db():
    return load_voice_command_database(_YAML)


@pytest.mark.parametrize(
    ("phrase", "intent_id"),
    [
        ("make a 3D model of this", "image_to_textured_mesh"),
        ("model this", "image_to_textured_mesh"),
        ("image to 3D please", "image_to_textured_mesh"),
        ("text to 3D a red sports car", "text_to_textured_mesh"),
        ("from my description build a chair", "text_to_textured_mesh"),
        ("make a 3D world from this", "image_to_world"),
        ("image to world", "image_to_world"),
        ("rig this", "auto_rig"),
        ("auto rig the model", "auto_rig"),
    ],
)
def test_example_phrases_match_intent(db, phrase: str, intent_id: str):
    intent = db.match(phrase)
    assert intent is not None, f"no match for {phrase!r}"
    assert intent.intent_id == intent_id


def test_world_does_not_steal_mesh_query(db):
    intent = db.match("make a 3D model of this")
    assert intent is not None
    assert intent.intent_id == "image_to_textured_mesh"


def test_routing_order_prefers_text_over_mesh(db):
    intent = db.match("text to 3D from my description of a cube")
    assert intent is not None
    assert intent.intent_id == "text_to_textured_mesh"


def test_database_has_all_job_intents(db):
    assert set(db.intents) == {
        "image_to_textured_mesh",
        "text_to_textured_mesh",
        "image_to_world",
        "auto_rig",
    }
