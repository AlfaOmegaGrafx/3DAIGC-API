"""Text-to-image and image-edit generation models."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


class ImageEditModel(BaseModel):
    """Instruction-based image editing (image + text → edited image)."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "image_edit",
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
        logger.info("Loading image-edit model: %s", self.model_id)

    def _unload_model(self):
        logger.info("Unloading image-edit model: %s", self.model_id)

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _process_request")

    def _validate_edit_inputs(self, inputs: Dict[str, Any]) -> Tuple[str, str]:
        if "image_path" not in inputs:
            raise ValueError("image_path is required for image editing")
        image_path = str(inputs["image_path"]).strip()
        if not image_path:
            raise ValueError("image_path cannot be empty")
        if not Path(image_path).is_file():
            raise ValueError(f"image_path not found: {image_path}")

        if "text_prompt" not in inputs:
            raise ValueError("text_prompt is required for image editing")
        text_prompt = str(inputs["text_prompt"]).strip()
        if not text_prompt:
            raise ValueError("text_prompt cannot be empty")
        return image_path, text_prompt

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {"input": ["image", "text"], "output": self.supported_output_formats}

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Denoising steps (Turbo: 4)",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 50,
                    "required": False,
                },
                "guidance_scale": {
                    "type": "number",
                    "description": "CFG scale (Turbo: 1.0)",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "required": False,
                },
                "max_size": {
                    "type": "integer",
                    "description": "Longest output side (px); aspect follows reference",
                    "default": 1024,
                    "minimum": 512,
                    "maximum": 2048,
                    "required": False,
                },
                "width": {
                    "type": "integer",
                    "description": "Optional explicit output width (with height)",
                    "minimum": 512,
                    "maximum": 2048,
                    "required": False,
                },
                "height": {
                    "type": "integer",
                    "description": "Optional explicit output height (with width)",
                    "minimum": 512,
                    "maximum": 2048,
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
