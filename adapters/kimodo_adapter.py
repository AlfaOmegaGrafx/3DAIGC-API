"""
Kimodo text-to-motion adapter (NVIDIA SOMA skeleton → studio_motion.json for VRM).

Requires Kimodo installed under thirdparty/kimodo (see scripts/setup_kimodo.sh).
Set TEXT_ENCODER_DEVICE=cpu in the environment when GPU VRAM is limited (~17GB).
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.motion_models import TextToMotionModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.soma_to_vrm_motion import soma_npz_to_studio_motion

logger = logging.getLogger(__name__)

DEFAULT_KIMODO_MODEL = "Kimodo-SOMA-RP-v1.1"


class KimodoTextToMotionAdapter(TextToMotionModel):
    """Generate human motion from text via Kimodo diffusion (SOMA skeleton)."""

    MODEL_ID = "kimodo_text_to_motion"
    FEATURE_TYPE = "text_to_motion"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 8192,
        kimodo_root: Optional[str] = None,
        hf_model_id: Optional[str] = None,
        **kwargs,
    ):
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "kimodo")
            )
        if kimodo_root is None:
            kimodo_root = model_path

        super().__init__(
            model_id=self.MODEL_ID,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=self.FEATURE_TYPE,
            supported_output_formats=["studio_motion", "npz", "bvh"],
        )

        self.kimodo_root = Path(kimodo_root)
        self.hf_model_id = hf_model_id or DEFAULT_KIMODO_MODEL
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        self._kimodo_model = None
        self._resolved_model_name: Optional[str] = None

    def _ensure_kimodo_importable(self) -> None:
        if not self.kimodo_root.is_dir():
            raise RuntimeError(
                f"Kimodo not found at {self.kimodo_root}. "
                "Run: bash scripts/setup_kimodo.sh"
            )
        root = str(self.kimodo_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    def _load_model(self):
        self._ensure_kimodo_importable()
        os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")

        import torch
        from kimodo import load_model

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Kimodo model %s on %s", self.hf_model_id, device)
        self._kimodo_model, self._resolved_model_name = load_model(
            self.hf_model_id,
            device=device,
            default_family="Kimodo",
            return_resolved_name=True,
        )
        return self._kimodo_model

    def _kimodo_postprocessing_enabled(self) -> bool:
        """MotionCorrection C++ ext is x86-only; skip on aarch64 (DGX Spark)."""
        if platform.machine().lower() in ("aarch64", "arm64"):
            return False
        try:
            import motion_correction  # noqa: F401
        except ImportError:
            return False
        return True

    def _unload_model(self):
        self._kimodo_model = None
        self._resolved_model_name = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        text_prompt = self._validate_text_inputs(inputs)
        duration = float(inputs.get("duration", 5.0))
        duration = max(1.0, min(30.0, duration))
        diffusion_steps = int(inputs.get("diffusion_steps", 100))
        seed = inputs.get("seed")
        output_format = inputs.get("output_format", "studio_motion")
        export_bvh = bool(inputs.get("export_bvh", False))

        if self._kimodo_model is None:
            self._load_model()

        model = self._kimodo_model
        fps = float(getattr(model, "fps", 30))
        num_frames = max(1, int(round(duration * fps)))

        if seed is not None:
            from kimodo.tools import seed_everything

            seed_everything(int(seed))

        logger.info(
            "Kimodo generate: prompt=%r frames=%s model=%s",
            text_prompt[:80],
            num_frames,
            self._resolved_model_name,
        )

        output = model(
            [text_prompt],
            [num_frames],
            constraint_lst=[],
            num_denoising_steps=diffusion_steps,
            num_samples=1,
            multi_prompt=True,
            post_processing=self._kimodo_postprocessing_enabled(),
            return_numpy=True,
        )

        stem = self.path_generator.generate_motion_path(
            self.model_id,
            "motion",
            output_format="npz",
        )
        npz_path = Path(stem)
        studio_path = npz_path.with_suffix(".studio_motion.json")
        bvh_path = npz_path.with_suffix(".bvh")

        from kimodo.exports.motion_io import save_kimodo_npz

        single = {
            k: (
                v[0]
                if hasattr(v, "shape")
                and len(v.shape) > 0
                and v.shape[0] == 1
                else v
            )
            for k, v in output.items()
        }
        save_kimodo_npz(str(npz_path), single)

        motion_meta = soma_npz_to_studio_motion(
            npz_path,
            studio_path,
            fps=fps,
            motion_name=text_prompt[:48] or "kimodo",
        )

        if export_bvh and "g1" not in (self._resolved_model_name or "").lower():
            try:
                import torch
                from kimodo.exports.bvh import save_motion_bvh

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                joints_pos = torch.from_numpy(output["posed_joints"][0]).to(device)
                joints_rot = torch.from_numpy(output["global_rot_mats"][0]).to(device)
                save_motion_bvh(
                    str(bvh_path),
                    model.skeleton,
                    joints_pos,
                    joints_rot,
                    fps=fps,
                    standard_tpose=False,
                )
            except Exception as exc:
                logger.warning("BVH export skipped: %s", exc)
                bvh_path = None
        else:
            bvh_path = None

        primary_path = studio_path
        if output_format == "npz":
            primary_path = npz_path
        elif output_format == "bvh" and bvh_path and bvh_path.exists():
            primary_path = bvh_path

        return {
            "output_motion_path": str(primary_path),
            "output_studio_motion_path": str(studio_path),
            "output_npz_path": str(npz_path),
            "output_bvh_path": str(bvh_path) if bvh_path and bvh_path.exists() else None,
            # Download endpoint compatibility
            "output_mesh_path": str(primary_path),
            "success": True,
            "text_prompt": text_prompt,
            "generation_info": {
                "input_type": "text",
                "duration_sec": duration,
                "num_frames": num_frames,
                "fps": fps,
                "kimodo_model": self._resolved_model_name,
                "track_count": len(motion_meta.get("tracks", [])),
                "output_format": output_format,
                "success": True,
            },
        }

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {"input": ["text"], "output": self.supported_output_formats}
