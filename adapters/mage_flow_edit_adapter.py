"""
Mage-Flow-Edit-Turbo adapter — instruction-based image editing (sidecar).

Uses an isolated venv (`.venv-mage-flow`) because Mage requires transformers 5.x
while the main API venv pins 4.57 for Krea/TRELLIS.

Weights: mage-flow-community/Mage-Flow-Edit-Turbo (pinned revision; official
microsoft/Mage-Flow-Edit-Turbo host withdrawn). Code: microsoft/Mage (MIT).

Setup: bash scripts/setup_mage_flow_edit.sh
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.image_models import ImageEditModel
from core.utils.file_utils import OutputPathGenerator
from utils.mage_flow_edit_helper import (
    DEFAULT_HF_ID,
    DEFAULT_HF_REVISION,
    DEFAULT_WEIGHTS_DIR,
    probe_mage_flow_edit_install,
    run_mage_flow_edit,
)

logger = logging.getLogger(__name__)

TURBO_DEFAULTS = {
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "max_size": 1024,
}


class MageFlowEditNotReadyError(RuntimeError):
    """Raised when isolated Mage venv / weights are incomplete."""


class MageFlowEditTurboAdapter(ImageEditModel):
    """4-step Mage-Flow-Edit-Turbo via isolated subprocess (unload = process exit)."""

    MODEL_ID = "mage_flow_edit_turbo"
    FEATURE_TYPE = "image_edit"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 20480,
        hf_model_id: Optional[str] = None,
        hf_revision: Optional[str] = None,
        **kwargs,
    ):
        if model_path is None:
            model_path = str(DEFAULT_WEIGHTS_DIR)
        super().__init__(
            model_id=self.MODEL_ID,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=self.FEATURE_TYPE,
            supported_output_formats=["png", "webp"],
        )
        self.hf_model_id = hf_model_id or os.environ.get(
            "MAGE_FLOW_EDIT_HF_ID", DEFAULT_HF_ID
        )
        self.hf_revision = hf_revision or os.environ.get(
            "MAGE_FLOW_EDIT_HF_REVISION", DEFAULT_HF_REVISION
        )
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        self._ready: Optional[Dict[str, Any]] = None

    def _load_model(self):
        status = probe_mage_flow_edit_install(weights_dir=Path(self.model_path))
        self._ready = status
        if not status.get("integrated"):
            raise MageFlowEditNotReadyError(
                "Mage-Flow-Edit not ready: "
                + "; ".join(status.get("blocking_reasons") or [])
            )
        return status

    def _unload_model(self):
        # Inference runs in a short-lived subprocess; nothing stays in this process.
        self._ready = None

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        image_path, text_prompt = self._validate_edit_inputs(inputs)
        image_path = str(Path(image_path).resolve())
        output_format = str(inputs.get("output_format", "png")).lower()
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        steps_raw = inputs.get("num_inference_steps")
        steps = (
            int(steps_raw)
            if steps_raw is not None
            else TURBO_DEFAULTS["num_inference_steps"]
        )
        cfg_raw = inputs.get("guidance_scale")
        cfg = (
            float(cfg_raw)
            if cfg_raw is not None
            else TURBO_DEFAULTS["guidance_scale"]
        )
        max_size = int(inputs.get("max_size") or TURBO_DEFAULTS["max_size"])
        seed = inputs.get("seed")
        width = inputs.get("width")
        height = inputs.get("height")

        if self._ready is None:
            self._load_model()

        output_path = self.path_generator.generate_image_path(
            self.model_id,
            "edit",
            output_format=output_format,
        )

        logger.info(
            "Mage-Flow-Edit: prompt=%r image=%s steps=%s cfg=%s",
            text_prompt[:80],
            image_path,
            steps,
            cfg,
        )

        result = run_mage_flow_edit(
            image_path=image_path,
            text_prompt=text_prompt,
            output_path=str(output_path),
            model_path=str(self.model_path),
            steps=steps,
            cfg=cfg,
            max_size=max_size,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            seed=int(seed) if seed is not None else 42,
        )

        return {
            "output_image_path": str(output_path),
            "output_mesh_path": str(output_path),
            "success": True,
            "text_prompt": text_prompt,
            "generation_info": {
                "input_type": "image_edit",
                "model": self.hf_model_id,
                "hf_revision": self.hf_revision,
                "checkpoint_family": "turbo",
                "image_path": image_path,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "max_size": max_size,
                "seed": seed,
                "output_format": output_format,
                "inference_mode": "isolated_venv_subprocess",
                "runner_result": result,
                "success": True,
            },
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = super().get_parameter_schema()
        schema["parameters"]["num_inference_steps"]["default"] = TURBO_DEFAULTS[
            "num_inference_steps"
        ]
        schema["parameters"]["guidance_scale"]["default"] = TURBO_DEFAULTS[
            "guidance_scale"
        ]
        return schema
