"""Basic unit tests for 3daigc-mcp (no live API required)."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from daigc_mcp.helpers import downscale_image_bytes, job_submit_envelope
from daigc_mcp.registry import all_tool_specs, tool_map


def test_v0_tool_count():
    specs = all_tool_specs()
    assert len(specs) == 12
    names = {s.tool.name for s in specs}
    assert names == {
        "health_check",
        "list_features",
        "list_models",
        "get_model_parameters",
        "upload_image",
        "upload_mesh",
        "text_to_textured_mesh",
        "image_to_textured_mesh",
        "generate_rig",
        "image_to_world",
        "get_job_status",
        "wait_for_job",
    }


def test_tool_map_unique():
    specs = tool_map()
    assert len(specs) == 12


def test_downscale_image_bytes():
    img = Image.new("RGB", (4096, 2048), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out, ctype = downscale_image_bytes(buf.getvalue(), 2048)
    assert ctype == "image/jpeg"
    with Image.open(io.BytesIO(out)) as scaled:
        assert max(scaled.size) <= 2048


def test_job_submit_envelope():
    payload = {"job_id": "abc", "status": "queued"}
    env = job_submit_envelope(payload, feature="image_to_textured_mesh")
    assert env["poll_tool"] == "get_job_status"
    assert env["wait_tool"] == "wait_for_job"
    assert env["feature"] == "image_to_textured_mesh"
