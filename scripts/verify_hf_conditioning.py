#!/usr/bin/env python3
"""
Preflight for HuggingFace-conditioned models (TRELLIS.2 BiRefNet + DINOv3).

Quick mode (default): API/layout checks, no GPU weight load.
GPU mode (--gpu): load BiRefNet + DINOv3 and verify dtype + feature path.

Used by verify_env_compat.py, install_local_venv_extras.sh, and post-pip drift checks.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRELLIS2_ROOT = ROOT / "thirdparty" / "TRELLIS.2"
EXPECTED_TRANSFORMERS = "4.57.3"
DINOV3_HF_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BIREFNET_HF_ID = "ZhengPeng7/BiRefNet"
# Manual layer loop vs last_hidden_state must differ (bad shortcut regresses here).
MIN_FEATURE_DIVERGENCE = 1.0


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> int:
    print(f"  FAIL {msg}")
    return 1


def warn(msg: str) -> None:
    print(f"  WARN {msg}")


def check_transformers_pin() -> int:
    import transformers

    ver = transformers.__version__
    if ver != EXPECTED_TRANSFORMERS:
        return fail(
            f"transformers=={EXPECTED_TRANSFORMERS} required (got {ver}); "
            "TRELLIS.2 DINOv3 layout is sensitive to this pin"
        )
    ok(f"transformers {ver}")
    return 0


def check_dinov3_layout() -> int:
    from transformers import DINOv3ViTModel

    if not hasattr(DINOv3ViTModel, "from_pretrained"):
        return fail("DINOv3ViTModel missing from transformers")
    # Structural check without loading weights.
    import inspect

    src = inspect.getsourcefile(DINOv3ViTModel) or ""
    ok(f"DINOv3ViTModel importable ({src})")
    return 0


def check_trellis2_extractor_source() -> int:
    path = TRELLIS2_ROOT / "trellis2" / "modules" / "image_feature_extractor.py"
    if not path.is_file():
        return fail(f"missing {path}")
    text = path.read_text(encoding="utf-8")
    if "class DinoV3FeatureExtractor" in text:
        return fail(
            "image_feature_extractor must re-export DinoV3FeatureExtractor "
            "(single implementation in image_conditioned.py)"
        )
    if "from ..trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor" not in text:
        return fail("image_feature_extractor must import DinoV3FeatureExtractor from image_conditioned")
    ok("DinoV3FeatureExtractor has one implementation (inference re-exports training)")
    return 0


def check_trellis2_extractor_runtime() -> int:
    """Mocked forward — catches wrong layer path without loading HF weights."""
    import torch
    from unittest.mock import MagicMock

    if str(TRELLIS2_ROOT) not in sys.path:
        sys.path.insert(0, str(TRELLIS2_ROOT))
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

    mock_vit = MagicMock()
    mock_vit.embeddings.patch_embeddings.weight.dtype = torch.float32
    mock_vit.embeddings.return_value = torch.randn(1, 100, 1024)
    mock_vit.rope_embeddings.return_value = (torch.randn(1, 100, 64), torch.randn(1, 100, 64))
    mock_layer = MagicMock(side_effect=lambda hidden_states, position_embeddings=None: hidden_states)
    mock_vit.layer = [mock_layer]

    ext = DinoV3FeatureExtractor.__new__(DinoV3FeatureExtractor)
    ext.model = mock_vit
    ext.transform = lambda x: x
    try:
        out = ext.extract_features(torch.randn(1, 3, 512, 512))
    except AttributeError as exc:
        return fail(f"DinoV3FeatureExtractor.extract_features runtime: {exc}")
    if out.shape != (1, 100, 1024):
        return fail(f"DinoV3FeatureExtractor unexpected output shape {tuple(out.shape)}")
    if not mock_layer.called:
        return fail("DinoV3FeatureExtractor did not iterate encoder layers")
    ok("DinoV3FeatureExtractor inference path executes layer loop")
    return 0


def check_birefnet_source() -> int:
    path = TRELLIS2_ROOT / "trellis2" / "pipelines" / "rembg" / "BiRefNet.py"
    if not path.is_file():
        return fail(f"missing {path}")
    text = path.read_text(encoding="utf-8")
    if "param.dtype" not in text:
        return fail("BiRefNet must cast inputs to model parameter dtype")
    ok("BiRefNet casts inputs to model dtype")
    return 0


def run_gpu_checks() -> int:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from transformers import DINOv3ViTModel

    if str(TRELLIS2_ROOT) not in sys.path:
        sys.path.insert(0, str(TRELLIS2_ROOT))
    from trellis2.modules.dinov3_encoder import get_dinov3_encoder_layers

    if not torch.cuda.is_available():
        warn("CUDA unavailable — skipping GPU HF conditioning checks")
        return 0

    errors = 0

    # --- DINOv3 feature path ---
    try:
        model = DINOv3ViTModel.from_pretrained(DINOV3_HF_ID).cuda().eval()
        layers = list(get_dinov3_encoder_layers(model))
        ok(f"DINOv3ViTModel layer stack ({len(layers)} blocks)")

        img = Image.new("RGB", (512, 512), color=(200, 100, 50))
        arr = np.array(img).astype(np.float32) / 255
        x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).cuda()
        x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)
        x = x.to(model.embeddings.patch_embeddings.weight.dtype)

        with torch.no_grad():
            out_fwd = model(pixel_values=x)
            hs = model.embeddings(x, bool_masked_pos=None)
            pe = model.rope_embeddings(x)
            for layer_module in layers:
                hs = layer_module(hs, position_embeddings=pe)
            manual = F.layer_norm(hs, hs.shape[-1:])
            shortcut = F.layer_norm(
                out_fwd.last_hidden_state, out_fwd.last_hidden_state.shape[-1:]
            )
            diff = (manual - shortcut).abs().max().item()

        if diff < MIN_FEATURE_DIVERGENCE:
            errors += fail(
                f"DINOv3 manual layer loop ≈ last_hidden_state (max diff {diff:.4f}); "
                "do not use last_hidden_state for TRELLIS conditioning"
            )
        else:
            ok(f"DINOv3 manual loop diverges from last_hidden_state (max diff {diff:.2f})")

        del model
        torch.cuda.empty_cache()

        # Inference module (same class pipelines use at runtime)
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        from PIL import Image

        ext = DinoV3FeatureExtractor(DINOV3_HF_ID, image_size=512)
        ext.cuda()
        img = Image.new("RGB", (512, 512), color=(200, 100, 50))
        feat = ext([img])
        if feat.ndim != 3 or feat.shape[0] != 1:
            errors += fail(f"DinoV3FeatureExtractor GPU output bad shape {tuple(feat.shape)}")
        else:
            ok(f"DinoV3FeatureExtractor inference forward {tuple(feat.shape)}")

        del ext
        torch.cuda.empty_cache()
    except Exception as exc:
        errors += fail(f"DINOv3 GPU check: {exc}")

    # --- BiRefNet dtype ---
    try:
        if str(TRELLIS2_ROOT) not in sys.path:
            sys.path.insert(0, str(TRELLIS2_ROOT))
        from trellis2.pipelines.rembg.BiRefNet import BiRefNet

        biref = BiRefNet(model_name=BIREFNET_HF_ID)
        biref.cuda()
        param = next(biref.model.parameters())
        ok(f"BiRefNet loaded dtype={param.dtype}")

        test_img = Image.new("RGB", (256, 256), color=(180, 90, 40))
        out = biref(test_img)
        if out.mode != "RGBA":
            errors += fail(f"BiRefNet output mode {out.mode}, expected RGBA")
        else:
            ok("BiRefNet forward pass (dtype-matched)")

        del biref
        torch.cuda.empty_cache()
    except Exception as exc:
        errors += fail(f"BiRefNet GPU check: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify HF conditioning stack for TRELLIS.2")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Load BiRefNet + DINOv3 on GPU (slower; run after pip install or weekly)",
    )
    args = parser.parse_args()

    print("=== HF / transformers conditioning (quick) ===")
    errors = 0
    for fn in (
        check_transformers_pin,
        check_dinov3_layout,
        check_trellis2_extractor_source,
        check_trellis2_extractor_runtime,
        check_birefnet_source,
    ):
        errors += fn()

    if args.gpu:
        print("\n=== HF conditioning (GPU load) ===")
        errors += run_gpu_checks()

    print("\n=== Summary ===")
    if errors:
        print(f"HF_VERIFY_FAIL ({errors} errors)")
        return 1
    print("HF_VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
