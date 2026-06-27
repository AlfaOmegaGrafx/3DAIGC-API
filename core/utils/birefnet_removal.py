"""
Lazy BiRefNet background removal (TripoSplat weights or TRELLIS HF fallback).
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

_MODEL = None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRIPOSPLAT_ROOT = _REPO_ROOT / "thirdparty" / "TripoSplat"
_CKPT = _REPO_ROOT / "pretrained" / "TripoSplat" / "ckpts" / "background_removal" / "birefnet.safetensors"


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if _CKPT.is_file() and _TRIPOSPLAT_ROOT.is_dir():
        root = str(_TRIPOSPLAT_ROOT.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from triposplat import load_rmbg  # noqa: WPS433

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading BiRefNet from TripoSplat checkpoint (%s)", device)
        _MODEL = load_rmbg(str(_CKPT), device=device)
        return _MODEL

    trellis_rembg = _REPO_ROOT / "thirdparty" / "TRELLIS.2" / "trellis2" / "pipelines" / "rembg"
    if trellis_rembg.is_dir():
        trellis_root = str((_REPO_ROOT / "thirdparty" / "TRELLIS.2").resolve())
        if trellis_root not in sys.path:
            sys.path.insert(0, trellis_root)
        from trellis2.pipelines.rembg.BiRefNet import BiRefNet  # noqa: WPS433

        logger.info("Loading BiRefNet from HuggingFace (TRELLIS rembg)")
        model = BiRefNet()
        try:
            import torch

            if torch.cuda.is_available():
                model.cuda()
        except Exception:
            pass
        _MODEL = model
        return _MODEL

    raise FileNotFoundError(
        "BiRefNet weights not found. Run scripts/download_models.sh triposplat "
        "or ensure TRELLIS.2 rembg is available."
    )


def remove_background_from_bytes(
    image_bytes: bytes,
    *,
    erode_radius: int = 1,
) -> bytes:
    """Return PNG bytes (RGBA) with background removed."""
    model = _load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if hasattr(model, "remove_background"):
        rgba = model.remove_background(img)
    else:
        rgba = model(img.convert("RGB"))

    radius = max(0, min(int(erode_radius), 8))
    if radius > 0:
        rgba.putalpha(
            rgba.getchannel("A").filter(ImageFilter.MinFilter(2 * radius + 1))
        )

    out = io.BytesIO()
    rgba.save(out, format="PNG")
    return out.getvalue()
