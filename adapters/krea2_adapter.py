"""
Krea 2 text-to-image adapter (local open weights — no Krea API).

Uses Hugging Face diffusers Krea2Pipeline with weights from krea/Krea-2-Turbo (inference)
or krea/Krea-2-Raw (LoRA training base; not recommended for production inference).

Setup: bash scripts/setup_krea2.sh
License: Krea 2 Community License — see docs/MODEL_LICENSES.md (CONDITIONAL commercial).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.image_models import TextToImageModel
from core.utils.file_utils import OutputPathGenerator

logger = logging.getLogger(__name__)

DEFAULT_TURBO_HF_ID = "krea/Krea-2-Turbo"
DEFAULT_RAW_HF_ID = "krea/Krea-2-Raw"
DEFAULT_QWEN3_VL_TOKENIZER_ID = "Qwen/Qwen3-VL-4B-Instruct"

TURBO_DEFAULTS = {
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "width": 1024,
    "height": 1024,
}

RAW_DEFAULTS = {
    "num_inference_steps": 52,
    "guidance_scale": 3.5,
    "width": 1024,
    "height": 1024,
}


class Krea2TurboTextToImageAdapter(TextToImageModel):
    """Fast, high-quality text-to-image via Krea 2 Turbo (8-step distilled checkpoint)."""

    MODEL_ID = "krea2_turbo_text_to_image"
    FEATURE_TYPE = "text_to_image"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 32768,
        hf_model_id: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "pretrained", "krea", "Krea-2-Turbo")
            )
        super().__init__(
            model_id=self.MODEL_ID,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=self.FEATURE_TYPE,
            supported_output_formats=["png", "webp"],
        )
        self.hf_model_id = hf_model_id or DEFAULT_TURBO_HF_ID
        self.torch_dtype = torch_dtype
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        self._pipe = None

    def _resolve_pretrained_path(self) -> str:
        local = Path(self.model_path)
        if local.is_dir() and any(local.glob("*.json")):
            return str(local)
        return self.hf_model_id

    @staticmethod
    def _qwen3vl_config_for_transformers_457(pretrained_dir: Path):
        """Krea exports use transformers 5.x rope_parameters; 4.57.x expects rope_scaling."""
        import json

        from transformers import Qwen3VLConfig

        cfg_path = pretrained_dir / "text_encoder" / "config.json"
        if not cfg_path.is_file():
            return None
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        text_cfg = dict(cfg.get("text_config") or {})
        if text_cfg.get("rope_scaling") is None and text_cfg.get("rope_parameters"):
            rp = text_cfg["rope_parameters"]
            text_cfg["rope_scaling"] = {
                "mrope_interleaved": rp.get("mrope_interleaved", True),
                "mrope_section": rp.get("mrope_section", [24, 20, 20]),
                "rope_type": rp.get("rope_type", "default"),
            }
            cfg["text_config"] = text_cfg
        return Qwen3VLConfig.from_dict(cfg)

    def _load_text_encoder_and_tokenizer(self, pretrained_dir: Path, dtype):
        from transformers import AutoTokenizer, Qwen3VLModel

        config = self._qwen3vl_config_for_transformers_457(pretrained_dir)
        te_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if config is not None:
            te_kwargs["config"] = config
        text_encoder = Qwen3VLModel.from_pretrained(
            str(pretrained_dir / "text_encoder"),
            **te_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_QWEN3_VL_TOKENIZER_ID,
            use_fast=False,
            trust_remote_code=True,
        )
        text_encoder_device = os.environ.get("TEXT_ENCODER_DEVICE", "").strip().lower()
        if text_encoder_device in {"cpu", "cuda"}:
            text_encoder = text_encoder.to(text_encoder_device)
        return text_encoder, tokenizer

    def _load_model(self):
        import torch

        try:
            from diffusers import Krea2Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers Krea2Pipeline not installed. Run: bash scripts/setup_krea2.sh"
            ) from exc

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        pretrained = self._resolve_pretrained_path()
        pretrained_dir = Path(pretrained)
        logger.info("Loading Krea2Pipeline from %s (dtype=%s)", pretrained, dtype)

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if pretrained_dir.is_dir() and (pretrained_dir / "text_encoder").is_dir():
            text_encoder, tokenizer = self._load_text_encoder_and_tokenizer(
                pretrained_dir, dtype
            )
            load_kwargs["text_encoder"] = text_encoder
            load_kwargs["tokenizer"] = tokenizer

        pipe = Krea2Pipeline.from_pretrained(pretrained, **load_kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        self._pipe = pipe
        return pipe

    def _unload_model(self):
        self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text_prompt = self._validate_text_inputs(inputs)
        width = int(inputs.get("width") or TURBO_DEFAULTS["width"])
        height = int(inputs.get("height") or TURBO_DEFAULTS["height"])
        steps_raw = inputs.get("num_inference_steps")
        steps = int(steps_raw) if steps_raw is not None else TURBO_DEFAULTS["num_inference_steps"]
        guidance_raw = inputs.get("guidance_scale")
        guidance = (
            float(guidance_raw)
            if guidance_raw is not None
            else TURBO_DEFAULTS["guidance_scale"]
        )
        seed = inputs.get("seed")
        output_format = str(inputs.get("output_format", "png")).lower()
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        if self._pipe is None:
            self._load_model()

        import torch

        generator = None
        if seed is not None:
            device = self._pipe.device if hasattr(self._pipe, "device") else "cuda"
            generator = torch.Generator(device=device).manual_seed(int(seed))

        logger.info(
            "Krea2 Turbo generate: prompt=%r %sx%s steps=%s cfg=%s",
            text_prompt[:80],
            width,
            height,
            steps,
            guidance,
        )

        result = self._pipe(
            prompt=text_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        image = result.images[0]

        output_path = self.path_generator.generate_image_path(
            self.model_id,
            "image",
            output_format=output_format,
        )
        image.save(output_path)

        return {
            "output_image_path": str(output_path),
            "output_mesh_path": str(output_path),
            "success": True,
            "text_prompt": text_prompt,
            "generation_info": {
                "input_type": "text",
                "model": self.hf_model_id,
                "checkpoint_family": "turbo",
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "seed": seed,
                "output_format": output_format,
                "inference_mode": "local_open_weights",
                "success": True,
            },
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = super().get_parameter_schema()
        schema["parameters"]["num_inference_steps"]["default"] = TURBO_DEFAULTS[
            "num_inference_steps"
        ]
        schema["parameters"]["guidance_scale"]["default"] = TURBO_DEFAULTS["guidance_scale"]
        return schema


class Krea2RawTextToImageAdapter(Krea2TurboTextToImageAdapter):
    """Undistilled Krea 2 Raw — for research / LoRA training workflows, not default inference."""

    MODEL_ID = "krea2_raw_text_to_image"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 32768,
        hf_model_id: Optional[str] = None,
        **kwargs,
    ):
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "pretrained", "krea", "Krea-2-Raw")
            )
        super().__init__(
            model_path=model_path,
            vram_requirement=vram_requirement,
            hf_model_id=hf_model_id or DEFAULT_RAW_HF_ID,
            **kwargs,
        )
        self.model_id = self.MODEL_ID

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs.setdefault("num_inference_steps", RAW_DEFAULTS["num_inference_steps"])
        inputs.setdefault("guidance_scale", RAW_DEFAULTS["guidance_scale"])
        inputs.setdefault("width", RAW_DEFAULTS["width"])
        inputs.setdefault("height", RAW_DEFAULTS["height"])
        result = super()._process_request(inputs)
        result["generation_info"]["checkpoint_family"] = "raw"
        return result

    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = super().get_parameter_schema()
        schema["parameters"]["num_inference_steps"]["default"] = RAW_DEFAULTS[
            "num_inference_steps"
        ]
        schema["parameters"]["guidance_scale"]["default"] = RAW_DEFAULTS["guidance_scale"]
        return schema
