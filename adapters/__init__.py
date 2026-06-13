"""
Model adapters for integrating specific AI models into the framework.

Imports are lazy so ``importlib.import_module("adapters.trellis_adapter")`` (used by
the scheduler) does not execute every adapter's dependency chain at import time.
"""

import importlib
from typing import Any

# Restore torch.load(weights_only=False) default for trusted local checkpoints
# (PyTorch 2.6+ defaults to weights_only=True, which breaks several model
# checkpoints). Imported here so it runs before any adapter loads a model.
from core.utils import torch_compat as _torch_compat  # noqa: F401

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Hunyuan3DV21ImageMeshPaintingAdapter": (
        ".hunyuan3d_adapter_v21",
        "Hunyuan3DV21ImageMeshPaintingAdapter",
    ),
    "Hunyuan3DV21ImageToRawMeshAdapter": (
        ".hunyuan3d_adapter_v21",
        "Hunyuan3DV21ImageToRawMeshAdapter",
    ),
    "Hunyuan3DV21ImageToTexturedMeshAdapter": (
        ".hunyuan3d_adapter_v21",
        "Hunyuan3DV21ImageToTexturedMeshAdapter",
    ),
    "PartFieldSegmentationAdapter": (
        ".partfield_adapter",
        "PartFieldSegmentationAdapter",
    ),
    "PartPackerImageToRawMeshAdapter": (
        ".partpacker_adapter",
        "PartPackerImageToRawMeshAdapter",
    ),
    "TrellisImageMeshPaintingAdapter": (
        ".trellis_adapter",
        "TrellisImageMeshPaintingAdapter",
    ),
    "TrellisImageToTexturedMeshAdapter": (
        ".trellis_adapter",
        "TrellisImageToTexturedMeshAdapter",
    ),
    "TrellisTextMeshPaintingAdapter": (
        ".trellis_adapter",
        "TrellisTextMeshPaintingAdapter",
    ),
    "TrellisTextToTexturedMeshAdapter": (
        ".trellis_adapter",
        "TrellisTextToTexturedMeshAdapter",
    ),
    "Trellis2ImageMeshPaintingAdapter": (
        ".trellis2_adapter",
        "Trellis2ImageMeshPaintingAdapter",
    ),
    "Trellis2ImageToTexturedMeshAdapter": (
        ".trellis2_adapter",
        "Trellis2ImageToTexturedMeshAdapter",
    ),
    "UniRigAdapter": (".unirig_adapter", "UniRigAdapter"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_rel, attr = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_rel, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
