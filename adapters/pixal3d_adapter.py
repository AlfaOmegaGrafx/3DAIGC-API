"""
Pixal3D adapter — single image to textured GLB (PBR, pixel-aligned).

Upstream: https://github.com/TencentARC/Pixal3D (fork: AlfaOmegaGrafx/Pixal3D)
Weights: TencentARC/Pixal3D on Hugging Face
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.thumbnail_utils import generate_mesh_thumbnail

from huggingface_hub import get_token

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = _REPO_ROOT / "thirdparty" / "Pixal3D"


def pixal3d_repo_present() -> bool:
    return (PIXAL3D_ROOT / "inference.py").is_file()


def pixal3d_available() -> bool:
    if not pixal3d_repo_present():
        return False
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class Pixal3DImageToTexturedMeshAdapter(ImageToMeshModel):
    """SIGGRAPH 2026 pixel-aligned image → textured GLB."""

    FEATURE_TYPE = "image_to_textured_mesh"
    MODEL_ID = "pixal3d_image_to_textured_mesh"

    def __init__(self, **kwargs):
        super().__init__(
            model_id=self.MODEL_ID,
            model_path=str(PIXAL3D_ROOT),
            vram_requirement=int(kwargs.get("vram_requirement", 20480)),
            supported_output_formats=["glb"],
            feature_type=self.FEATURE_TYPE,
            max_images=1,
        )
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        self.model_hf_path = kwargs.get("model_hf_path", "TencentARC/Pixal3D")
        self.default_low_vram = bool(kwargs.get("default_low_vram", True))

    def _load_model(self):
        if not pixal3d_available():
            raise RuntimeError(
                "Pixal3D unavailable — clone thirdparty/Pixal3D and run "
                "scripts/install_pixal3d_deps.sh"
            )
        return {"repo": str(PIXAL3D_ROOT)}

    def _unload_model(self):
        pass

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        extra = [
            str(PIXAL3D_ROOT.resolve()),
            str(_REPO_ROOT),
        ]
        venv_site = (
            _REPO_ROOT
            / "venv"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if venv_site.is_dir():
            extra.append(str(venv_site))
        if env.get("PYTHONPATH"):
            extra.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(extra)
        env.setdefault("ATTN_BACKEND", os.environ.get("ATTN_BACKEND", "sdpa"))
        # pipeline.json defaults to gated briaai/RMBG-2.0; BiRefNet HF repo is open.
        env.setdefault("PIXAL3D_REMBG_MODEL", "ZhengPeng7/BiRefNet")
        hf_token = (
            env.get("HF_TOKEN")
            or env.get("HUGGINGFACE_TOKEN")
            or env.get("HUGGING_FACE_HUB_TOKEN")
            or get_token()
        )
        if hf_token:
            env.setdefault("HF_TOKEN", hf_token)
            env.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        return env

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" not in inputs:
            raise ValueError("image_path is required")

        image_path = Path(inputs["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        output_format = inputs.get("output_format", "glb")
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        params = inputs.get("model_parameters") or {}
        low_vram = bool(params.get("low_vram", self.default_low_vram))
        resolution = int(params.get("resolution", 1024 if low_vram else 1536))
        seed = int(params.get("seed", inputs.get("seed", 42)))
        manual_fov = float(params.get("fov", inputs.get("fov", -1.0)))

        output_path = self.path_generator.generate_mesh_path(
            self.model_id,
            image_path.stem,
            output_format="glb",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PIXAL3D_ROOT / "inference.py"),
            "--image",
            str(image_path.resolve()),
            "--output",
            str(output_path.resolve()),
            "--seed",
            str(seed),
            "--model_path",
            str(params.get("model_path", self.model_hf_path)),
            "--resolution",
            str(resolution),
        ]
        if low_vram:
            cmd.append("--low_vram")
        if manual_fov >= 0:
            cmd.extend(["--fov", str(manual_fov)])

        logger.info("Pixal3D inference: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=str(PIXAL3D_ROOT),
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stdout or "")[-4000:] + "\n" + (proc.stderr or "")[-4000:]
            raise RuntimeError(f"Pixal3D inference failed (exit {proc.returncode}):\n{tail}")

        if not output_path.is_file():
            raise FileNotFoundError(f"Pixal3D did not produce output: {output_path}")

        thumb_path = None
        try:
            thumb_path = generate_mesh_thumbnail(str(output_path))
        except Exception as exc:
            logger.warning("Thumbnail generation skipped: %s", exc)

        return {
            "output_mesh_path": str(output_path),
            "success": True,
            "thumbnail_path": thumb_path,
            "generation_info": {
                "model": self.model_id,
                "input_image": str(image_path),
                "output_format": output_format,
                "low_vram": low_vram,
                "resolution": resolution,
                "seed": seed,
            },
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "low_vram": {
                    "type": "boolean",
                    "description": "Load models on-demand (lower peak VRAM, slower)",
                    "default": True,
                    "required": False,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Pipeline resolution (1024 or 1536)",
                    "default": 1024,
                    "enum": [1024, 1536],
                    "required": False,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed",
                    "default": 42,
                    "required": False,
                },
                "fov": {
                    "type": "number",
                    "description": "Manual camera FOV radians (-1 = MoGe auto)",
                    "default": -1,
                    "required": False,
                },
            }
        }
