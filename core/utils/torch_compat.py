"""
Torch compatibility shims.

PyTorch 2.6 flipped the default of ``torch.load(weights_only=...)`` from
``False`` to ``True``. Many model checkpoints shipped by the integrated
thirdparty projects (PartField, PartPacker, UltraShape, UniRig, Hunyuan3D,
FastMesh, …) are full pickles that contain non-tensor objects such as
``yacs.config.CfgNode`` or argparse namespaces, which the restricted
``weights_only=True`` unpickler refuses to load.

All of these checkpoints are trusted local files downloaded by
``scripts/download_models.sh``, so we restore the pre-2.6 behaviour by
defaulting ``weights_only=False`` whenever a caller did not specify it.

Import this module once, early, before any model is loaded.
"""

import torch

if not getattr(torch, "_3daigc_weights_only_patched", False):
    _orig_torch_load = torch.load

    def _torch_load_weights_only_false(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_weights_only_false
    torch._3daigc_weights_only_patched = True
