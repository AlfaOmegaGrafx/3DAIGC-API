"""Text-to-image generation models."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


class TextToImageModel(BaseModel):
    """Generate raster images from natural-language prompts."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "text_to_image",
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
        )
        self.supported_output_formats = supported_output_formats or ["png", "webp"]

    def _load_model(self):
        logger.info("Loading text-to-image model: %s", self.model_id)

    def _unload_model(self):
        logger.info("Unloading text-to-image model: %s", self.model_id)

    def _validate_text_inputs(self, inputs: Dict[str, Any]) -> str:
        if "text_prompt" not in inputs:
            raise ValueError("text_prompt is required for text-to-image generation")
        text_prompt = str(inputs["text_prompt"]).strip()
        if not text_prompt:
            raise ValueError("text_prompt cannot be empty")
        return text_prompt

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {"input": ["text"], "output": self.supported_output_formats}

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "width": {
                    "type": "integer",
                    "description": "Output width in pixels (multiple of 16)",
                    "default": 1024,
                    "minimum": 512,
                    "maximum": 2048,
                    "required": False,
                },
                "height": {
                    "type": "integer",
                    "description": "Output height in pixels (multiple of 16)",
                    "default": 1024,
                    "minimum": 512,
                    "maximum": 2048,
                    "required": False,
                },
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Denoising steps (Turbo: 8, Raw: 52)",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 100,
                    "required": False,
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "CFG scale (Turbo: 0.0, Raw: 3.5)",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "required": False,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed (optional)",
                    "minimum": 0,
                    "required": False,
                },
            }
        }
