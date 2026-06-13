"""
TripoSplat inference helper (VAST-AI-Research/TripoSplat).

Wraps the upstream TripoSplatPipeline for headless API use.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIPOSPLAT_ROOT = REPO_ROOT / "thirdparty" / "TripoSplat"
DEFAULT_CKPT_ROOT = REPO_ROOT / "pretrained" / "TripoSplat" / "ckpts"


class TripoSplatPipelineHelper:
    """Lazy-loaded TripoSplat pipeline."""

    def __init__(
        self,
        triposplat_root: Optional[Path] = None,
        ckpt_root: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.triposplat_root = Path(triposplat_root or DEFAULT_TRIPOSPLAT_ROOT)
        self.ckpt_root = Path(ckpt_root or DEFAULT_CKPT_ROOT)
        self.device = device
        self._pipe = None

    def _ensure_import_path(self) -> None:
        root = str(self.triposplat_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    def _ckpt(self, *parts: str) -> str:
        return str(self.ckpt_root.joinpath(*parts))

    def load(self) -> None:
        if self._pipe is not None:
            return

        if not self.triposplat_root.is_dir():
            raise FileNotFoundError(
                f"TripoSplat source missing at {self.triposplat_root}. "
                "Clone https://github.com/VAST-AI-Research/TripoSplat into thirdparty/TripoSplat "
                "and download weights via scripts/download_models.sh triposplat"
            )

        required = [
            self.ckpt_root / "diffusion_models" / "triposplat_fp16.safetensors",
            self.ckpt_root / "vae" / "triposplat_vae_decoder_fp16.safetensors",
            self.ckpt_root / "clip_vision" / "dino_v3_vit_h.safetensors",
            self.ckpt_root / "vae" / "flux2-vae.safetensors",
            self.ckpt_root / "background_removal" / "birefnet.safetensors",
        ]
        missing = [p for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "TripoSplat checkpoints missing:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\nRun: ./scripts/download_models.sh triposplat"
            )

        self._ensure_import_path()
        from triposplat import TripoSplatPipeline  # noqa: WPS433

        logger.info("Loading TripoSplat pipeline on %s", self.device)
        self._pipe = TripoSplatPipeline(
            ckpt_path=self._ckpt("diffusion_models", "triposplat_fp16.safetensors"),
            decoder_path=self._ckpt("vae", "triposplat_vae_decoder_fp16.safetensors"),
            dinov3_path=self._ckpt("clip_vision", "dino_v3_vit_h.safetensors"),
            flux2_vae_encoder_path=self._ckpt("vae", "flux2-vae.safetensors"),
            rmbg_path=self._ckpt("background_removal", "birefnet.safetensors"),
            device=self.device,
        )

    def unload(self) -> None:
        self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def run(
        self,
        image_path: str,
        *,
        seed: int = 42,
        steps: int = 20,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        num_gaussians: int = 131072,
        erode_radius: int = 1,
    ) -> Any:
        """Run inference; returns upstream Gaussian object."""
        self.load()
        assert self._pipe is not None
        gaussian, _prepared = self._pipe.run(
            image_path,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            shift=shift,
            num_gaussians=num_gaussians,
            erode_radius=erode_radius,
            show_progress=False,
        )
        return gaussian
