"""Text-to-motion generation models (Kimodo and future motion backends)."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


class TextToMotionModel(BaseModel):
    """Generate skeletal motion clips from natural-language prompts."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "text_to_motion",
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
        )
        self.supported_output_formats = supported_output_formats or [
            "studio_motion",
            "npz",
            "bvh",
        ]

    def _load_model(self):
        logger.info("Loading text-to-motion model: %s", self.model_id)

    def _unload_model(self):
        logger.info("Unloading text-to-motion model: %s", self.model_id)

    def _validate_text_inputs(self, inputs: Dict[str, Any]) -> str:
        if "text_prompt" not in inputs:
            raise ValueError("text_prompt is required for text-to-motion generation")
        text_prompt = str(inputs["text_prompt"]).strip()
        if not text_prompt:
            raise ValueError("text_prompt cannot be empty")
        return text_prompt

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {"input": ["text"], "output": self.supported_output_formats}

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "duration": {
                    "type": "number",
                    "description": "Motion duration in seconds",
                    "default": 5.0,
                    "minimum": 1.0,
                    "maximum": 30.0,
                    "required": False,
                },
                "diffusion_steps": {
                    "type": "integer",
                    "description": "Kimodo denoising steps (quality vs speed)",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 200,
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
