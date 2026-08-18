#!/usr/bin/env python3
"""Run Mage-Flow-Edit in the isolated .venv-mage-flow interpreter.

Usage (from adapter / helper):
  .venv-mage-flow/bin/python scripts/mage_flow_edit_runner.py '<json payload>'

Prints one JSON object on stdout (last line) with success + output_path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "missing JSON payload"}))
        return 2

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(json.dumps({"success": False, "error": f"bad JSON: {exc}"}))
        return 2

    try:
        from mage_flow import MageFlowPipeline
        from mage_flow.models.modules import _attn_backend
    except ImportError as exc:
        print(json.dumps({"success": False, "error": f"mage_flow import failed: {exc}"}))
        return 1

    attn = str(payload.get("attn_backend") or "sdpa")
    # Force SDPA for DiT + HF text encoder (flash-attn often unavailable on aarch64 Spark).
    os.environ["VF_HF_ATTN_IMPL"] = "sdpa" if attn == "sdpa" else os.environ.get(
        "VF_HF_ATTN_IMPL", ""
    )
    _orig_set = _attn_backend.set_attn_backend

    def _force_backend(_name: str):
        return _orig_set(attn)

    _attn_backend.set_attn_backend = _force_backend  # type: ignore[method-assign]
    _orig_set(attn)

    image_path = Path(payload["image_path"])
    text_prompt = str(payload["text_prompt"]).strip()
    output_path = Path(payload["output_path"])
    model_path = str(Path(payload.get("model_path") or "").expanduser().resolve())
    steps = int(payload.get("steps") or 4)
    cfg = float(payload.get("cfg") if payload.get("cfg") is not None else 1.0)
    max_size = int(payload.get("max_size") or 1024)
    seed = payload.get("seed")
    width = payload.get("width")
    height = payload.get("height")

    if not image_path.is_file():
        print(json.dumps({"success": False, "error": f"image not found: {image_path}"}))
        return 1
    if not text_prompt:
        print(json.dumps({"success": False, "error": "text_prompt is empty"}))
        return 1
    if not model_path or not (Path(model_path) / "model_index.json").is_file():
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"local Mage weights missing model_index.json: {model_path}",
                }
            )
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = MageFlowPipeline.from_pretrained(model_path, device="cuda")
    edit_kwargs = {
        "steps": steps,
        "cfg": cfg,
        "max_size": max_size,
    }
    if seed is not None:
        edit_kwargs["seeds"] = [int(seed)]
    if width is not None and height is not None:
        edit_kwargs["widths"] = [int(width)]
        edit_kwargs["heights"] = [int(height)]

    images = pipe.edit([text_prompt], [str(image_path)], **edit_kwargs)
    images[0].save(str(output_path))

    # Explicit unload before process exit (helps UMA reclaim sooner).
    del pipe
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(
        json.dumps(
            {
                "success": True,
                "output_path": str(output_path),
                "steps": steps,
                "cfg": cfg,
                "attn_backend": attn,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
