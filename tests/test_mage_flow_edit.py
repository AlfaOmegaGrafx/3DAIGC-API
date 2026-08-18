"""Unit tests for ImageEditModel validation and Mage adapter schema."""

from pathlib import Path

import pytest

from core.models.image_models import ImageEditModel
from adapters.mage_flow_edit_adapter import MageFlowEditTurboAdapter, TURBO_DEFAULTS


def test_image_edit_model_requires_image_and_prompt(tmp_path: Path):
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    model = ImageEditModel(
        model_id="test_edit",
        model_path=str(tmp_path),
        vram_requirement=1,
    )

    with pytest.raises(ValueError, match="image_path is required"):
        model._validate_edit_inputs({"text_prompt": "fix"})

    with pytest.raises(ValueError, match="text_prompt is required"):
        model._validate_edit_inputs({"image_path": str(img)})

    with pytest.raises(ValueError, match="cannot be empty"):
        model._validate_edit_inputs({"image_path": str(img), "text_prompt": "  "})

    path, prompt = model._validate_edit_inputs(
        {"image_path": str(img), "text_prompt": "make it red"}
    )
    assert path == str(img)
    assert prompt == "make it red"


def test_mage_flow_edit_turbo_schema_defaults():
    adapter = MageFlowEditTurboAdapter(model_path="/tmp/does-not-need-to-exist")
    schema = adapter.get_parameter_schema()["parameters"]
    assert schema["num_inference_steps"]["default"] == TURBO_DEFAULTS["num_inference_steps"]
    assert schema["guidance_scale"]["default"] == TURBO_DEFAULTS["guidance_scale"]
    assert adapter.MODEL_ID == "mage_flow_edit_turbo"
    assert adapter.FEATURE_TYPE == "image_edit"
