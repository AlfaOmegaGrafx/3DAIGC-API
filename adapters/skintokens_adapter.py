"""
SkinTokens / TokenRig model adapter for automatic rigging of 3D meshes.

Integration strategy:
- We call the upstream repo's CLI (`demo.py`) via subprocess to avoid importing a large
  dependency chain at module import time.
- We optionally download checkpoints using the upstream helper (`download.py --model`)
  if requested and if the expected `experiments/` layout is missing.

Upstream:
- GitHub: https://github.com/VAST-AI-Research/SkinTokens
- Project page: https://zjp-shadow.github.io/works/SkinTokens/
- HF weights: https://huggingface.co/VAST-AI/SkinTokens
"""

import logging
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.models.base import ModelStatus
from core.models.rig_models import AutoRigModel
from core.utils.file_utils import OutputPathGenerator

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.resolve()


class SkinTokensAdapter(AutoRigModel):
    """
    Adapter for SkinTokens / TokenRig.

    Notes:
    - Upstream CLI produces `.glb` outputs. This adapter currently supports `output_format="glb"`.
    - `rig_mode` in 3DAIGC-API supports multiple modes; TokenRig is effectively "full" rig.
      We accept "full" (preferred) and also allow "skeleton"/"skin" for compatibility,
      but they are mapped to full rigging.
    """

    def __init__(
        self,
        model_id: str = "skintokens_auto_rig",
        model_path: Optional[str] = None,
        vram_requirement: int = 15360,  # MB; upstream recommends >=14GB
        skintokens_root: Optional[str] = None,
        device: str = "cuda",
        auto_download_checkpoints: bool = False,
    ):
        if model_path is None:
            model_path = "thirdparty/SkinTokens"
        if skintokens_root is None:
            skintokens_root = model_path

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_input_formats=["glb", "gltf", "obj", "fbx", "ply", "stl"],
            supported_output_formats=["glb"],
        )

        self.device = device
        self.skintokens_root = _resolve_repo_path(skintokens_root)
        self.model_path = str(self.skintokens_root)
        self.auto_download_checkpoints = auto_download_checkpoints
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _repo_python(self) -> str:
        # Prefer the current interpreter (venv) for consistency inside workers.
        return sys.executable

    def _site_packages_paths(self) -> list[str]:
        paths: list[str] = []
        venv_root = os.environ.get("VIRTUAL_ENV")
        if venv_root:
            candidate = (
                Path(venv_root)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            if candidate.is_dir():
                paths.append(str(candidate))
        api_venv = (
            Path(__file__).resolve().parent.parent
            / "venv"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if api_venv.is_dir() and str(api_venv) not in paths:
            paths.append(str(api_venv))
        return paths

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parent.parent
        extra_paths = [
            str(self.skintokens_root.resolve()),
            str(repo_root),
            *self._site_packages_paths(),
        ]
        if env.get("PYTHONPATH"):
            extra_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)
        env.setdefault("BLENDER_BIN", "/usr/bin/blender")
        env.setdefault("DAIGC_ROOT", str(repo_root))
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        return env

    def _expected_qwen_dir(self) -> Path:
        return self.skintokens_root / "models" / "Qwen3-0.6B"

    def _qwen_weights_present(self) -> bool:
        qwen_dir = self._expected_qwen_dir()
        if not qwen_dir.is_dir():
            return False
        for pattern in ("*.safetensors", "*.bin"):
            if any(qwen_dir.glob(pattern)):
                return True
        return False

    def _glb_has_invalid_skin(self, mesh_path: Path) -> bool:
        if mesh_path.suffix.lower() not in {".glb", ".gltf"}:
            return False
        try:
            import json
            import struct

            data = mesh_path.read_bytes()
            if len(data) < 20 or data[:4] != b"glTF":
                return False
            off = 12
            chunk_len, chunk_type = struct.unpack("<I4s", data[off : off + 8])
            if chunk_type != b"JSON":
                return False
            gltf = json.loads(data[off + 8 : off + 8 + chunk_len])
            skins = gltf.get("skins") or []
            if not skins:
                return False
            for skin in skins:
                joints = skin.get("joints") or []
                if any(j is None or not isinstance(j, int) for j in joints):
                    return True
            return False
        except Exception:
            return False

    def _sanitize_mesh_input(self, mesh_path: Path) -> tuple[Path, Optional[Path]]:
        """
        SkinTokens/Blender cannot import GLBs with malformed skin joint indices.
        Strip rig metadata via trimesh while keeping mesh geometry.
        """
        needs_clean = self._glb_has_invalid_skin(mesh_path)
        if not needs_clean:
            return mesh_path, None

        try:
            import trimesh
        except ImportError as e:
            raise RuntimeError(
                "Input GLB contains invalid rig/skin data and trimesh is required to sanitize it."
            ) from e

        logger.warning(
            "Sanitizing input GLB for SkinTokens (invalid skin joints detected): %s",
            mesh_path,
        )
        loaded = trimesh.load(str(mesh_path))
        if isinstance(loaded, trimesh.Scene):
            export_mesh = loaded
        else:
            export_mesh = trimesh.Scene(loaded)

        tmp = tempfile.NamedTemporaryFile(
            suffix=mesh_path.suffix.lower() or ".glb",
            delete=False,
            prefix="skintokens_clean_",
        )
        tmp.close()
        clean_path = Path(tmp.name)
        export_mesh.export(str(clean_path))
        return clean_path, clean_path

    def _expected_ckpt_paths(self) -> list[Path]:
        return [
            self.skintokens_root
            / "experiments"
            / "articulation_xl_quantization_256_token_4"
            / "grpo_1400.ckpt",
            self.skintokens_root
            / "experiments"
            / "skin_vae_2_10_32768"
            / "last.ckpt",
        ]

    def _ensure_repo_present(self) -> None:
        if not self.skintokens_root.exists():
            raise FileNotFoundError(
                "SkinTokens repo not found. "
                f"Expected at: {self.skintokens_root}. "
                "Clone it into 3DAIGC-API/thirdparty/SkinTokens (recommended) "
                "or pass a custom skintokens_root in model init_params."
            )
        demo_py = self.skintokens_root / "demo.py"
        if not demo_py.is_file():
            raise FileNotFoundError(f"SkinTokens demo.py not found at: {demo_py}")

    def _maybe_download_checkpoints(self) -> None:
        ckpts = self._expected_ckpt_paths()
        needs_qwen = not self._qwen_weights_present()
        if all(p.is_file() for p in ckpts) and not needs_qwen:
            return

        if not self.auto_download_checkpoints:
            if needs_qwen:
                raise FileNotFoundError(
                    "SkinTokens Qwen LLM weights not found under "
                    f"{self._expected_qwen_dir()}. Run "
                    "`python download.py --model` from the SkinTokens repo root."
                )
            missing = [str(p) for p in ckpts if not p.is_file()]
            raise FileNotFoundError(
                "SkinTokens checkpoints not found. Missing: "
                + ", ".join(missing)
                + ". Download them with SkinTokens' helper: `python download.py --model` "
                "from the SkinTokens repo root, or set auto_download_checkpoints=true for this adapter."
            )

        download_py = self.skintokens_root / "download.py"
        if not download_py.is_file():
            raise FileNotFoundError(f"SkinTokens download.py not found at: {download_py}")

        logger.info("Downloading SkinTokens checkpoints (this may take a while).")
        cmd = [self._repo_python(), str(download_py), "--model"]
        env = os.environ.copy()
        # Token is handled by multiprocess worker bootstrap if provided.
        subprocess.run(
            cmd,
            cwd=str(self.skintokens_root),
            env=self._subprocess_env(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Verify again
        ckpts = self._expected_ckpt_paths()
        if not all(p.is_file() for p in ckpts):
            raise FileNotFoundError(
                "SkinTokens checkpoint download did not produce expected files. "
                f"Checked: {[str(p) for p in ckpts]}"
            )
        if not self._qwen_weights_present():
            raise FileNotFoundError(
                "SkinTokens Qwen LLM weights missing after download. "
                f"Expected weight files under {self._expected_qwen_dir()}"
            )

    @staticmethod
    def _allocate_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _run_demo_cli(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        env = self._subprocess_env()
        env["SKINTOKENS_BPY_PORT"] = str(self._allocate_free_port())
        proc = subprocess.run(
            cmd,
            cwd=str(self.skintokens_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stdout or "").strip()[-8000:]
            raise RuntimeError(
                f"SkinTokens demo.py exited with status {proc.returncode}"
                + (f":\n{tail}" if tail else "")
            )
        if proc.stdout:
            logger.info("SkinTokens output:\n%s", proc.stdout[-4000:])
        return proc

    def _load_model(self) -> Any:
        """
        TokenRig is invoked via CLI; "loading" validates repo presence and checkpoints.
        """
        self._ensure_repo_present()
        self._maybe_download_checkpoints()

        # Warm up CUDA a bit if available to reduce first-job latency.
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device=self.device)
                del _
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("SkinTokens adapter ready (CLI invocation mode).")
        return {"mode": "cli", "root": str(self.skintokens_root)}

    def _unload_model(self) -> None:
        # No persistent model objects here. Just clear CUDA cache.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            logger.info("SkinTokens adapter unloaded.")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._ensure_repo_present()
            self._maybe_download_checkpoints()

            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for auto-rigging")

            mesh_path = _resolve_repo_path(inputs["mesh_path"])
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            clean_tmp: Optional[Path] = None
            try:
                had_invalid_skin = self._glb_has_invalid_skin(mesh_path)
                mesh_path, clean_tmp = self._sanitize_mesh_input(mesh_path)

                rig_mode = str(inputs.get("rig_mode", "full")).lower()
                output_format = str(inputs.get("output_format", "glb")).lower()
                use_transfer = bool(inputs.get("use_transfer", True))
                if had_invalid_skin and use_transfer:
                    logger.warning(
                        "Disabling use_transfer for SkinTokens: input GLB had invalid "
                        "skin/rig metadata (common for client exports). Rigging will "
                        "proceed on sanitized geometry without texture transfer."
                    )
                    use_transfer = False
                if clean_tmp is not None and use_transfer:
                    logger.warning(
                        "Disabling use_transfer for sanitized input GLB "
                        "(prior rig/skin data was invalid; Blender transfer would crash)."
                    )
                    use_transfer = False
                use_postprocess = bool(inputs.get("use_postprocess", False))

                # Sampling params exposed by upstream CLI.
                top_k = inputs.get("top_k", 5)
                top_p = inputs.get("top_p", 0.95)
                temperature = inputs.get("temperature", 1.0)
                repetition_penalty = inputs.get("repetition_penalty", 2.0)
                num_beams = inputs.get("num_beams", 10)

                if output_format != "glb":
                    raise ValueError(
                        "SkinTokensAdapter currently supports output_format='glb' only."
                    )

                if rig_mode not in ["full", "skeleton", "skin", "template"]:
                    raise ValueError(
                        "Invalid rig_mode for SkinTokensAdapter. Allowed: full, skeleton, skin, template"
                    )
                if rig_mode == "template":
                    raise ValueError(
                        "SkinTokensAdapter does not support rig_mode='template'."
                    )

                # Output path
                base_name = _resolve_repo_path(inputs["mesh_path"]).stem
                output_path = self.path_generator.generate_rigged_path(
                    self.model_id, base_name, output_format
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                demo_py = self.skintokens_root / "demo.py"
                output_path = output_path.resolve()
                cmd: list[str] = [
                    self._repo_python(),
                    str(demo_py.resolve()),
                    "--input",
                    str(mesh_path),
                    "--output",
                    str(output_path),
                    "--top_k",
                    str(int(top_k)),
                    "--top_p",
                    str(float(top_p)),
                    "--temperature",
                    str(float(temperature)),
                    "--repetition_penalty",
                    str(float(repetition_penalty)),
                    "--num_beams",
                    str(int(num_beams)),
                ]

                if rig_mode == "skeleton":
                    cmd.append("--use_skeleton")
                if use_transfer:
                    cmd.append("--use_transfer")
                if use_postprocess:
                    cmd.append("--use_postprocess")

                logger.info("Running SkinTokens CLI: %s", " ".join(cmd))
                self._run_demo_cli(cmd)

                if not output_path.is_file():
                    raise RuntimeError(
                        f"SkinTokens CLI finished but output file missing: {output_path}"
                    )

                self.status = ModelStatus.LOADED
                return {
                    "output_mesh_path": str(output_path),
                    "bone_count": None,
                    "rig_info": {
                        "rig_type": "auto_detected",
                        "has_skinning": True,
                        "skeleton_only": False,
                        "generation_method": "skintokens_tokenrig_cli",
                        "rig_mode": "full",
                    },
                    "format": "glb",
                    "success": True,
                    "generation_info": {
                        "model": self.model_id,
                        "input_mesh": str(_resolve_repo_path(inputs["mesh_path"])),
                        "rig_mode": rig_mode,
                        "device": self.device,
                        "use_transfer": use_transfer,
                        "use_postprocess": use_postprocess,
                        "top_k": int(top_k),
                        "top_p": float(top_p),
                        "temperature": float(temperature),
                        "repetition_penalty": float(repetition_penalty),
                        "num_beams": int(num_beams),
                        "input_sanitized": clean_tmp is not None,
                    },
                }
            finally:
                if clean_tmp is not None:
                    try:
                        clean_tmp.unlink(missing_ok=True)
                    except OSError:
                        pass

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error("SkinTokens auto-rigging failed: %s", e)
            raise Exception(f"SkinTokens auto-rigging failed: {str(e)}")

    def get_supported_formats(self) -> Dict[str, List[str]]:
        return {"input": self.supported_input_formats, "output": self.supported_output_formats}

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "model_name": "SkinTokens / TokenRig",
                "version": "2026",
                "description": "Unified autoregressive rigging (skeleton + skin weights) via SkinTokens/TokenRig.",
                "requirements": {"vram_gb": 14, "cuda_required": True},
                "interface": "cli_demo.py",
                "upstream": {
                    "github": "https://github.com/VAST-AI-Research/SkinTokens",
                    "hf": "https://huggingface.co/VAST-AI/SkinTokens",
                },
            }
        )
        return info

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "rig_mode": {
                    "type": "string",
                    "description": "Rigging mode (TokenRig generates full rig; other modes are accepted for compatibility).",
                    "default": "full",
                    "enum": ["full", "skeleton", "skin"],
                    "required": False,
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format. SkinTokens CLI outputs GLB.",
                    "default": "glb",
                    "enum": ["glb"],
                    "required": False,
                },
                "use_transfer": {
                    "type": "boolean",
                    "description": "Preserve original texture and scale when possible (upstream --use_transfer).",
                    "default": True,
                    "required": False,
                },
                "use_postprocess": {
                    "type": "boolean",
                    "description": "Apply voxel-based skin postprocessing (upstream --use_postprocess).",
                    "default": False,
                    "required": False,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Top-k sampling (upstream --top_k).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                    "required": False,
                },
                "top_p": {
                    "type": "number",
                    "description": "Top-p sampling (upstream --top_p).",
                    "default": 0.95,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "required": False,
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (upstream --temperature).",
                    "default": 1.0,
                    "minimum": 0.0,
                    "maximum": 5.0,
                    "required": False,
                },
                "repetition_penalty": {
                    "type": "number",
                    "description": "Repetition penalty (upstream --repetition_penalty).",
                    "default": 2.0,
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "required": False,
                },
                "num_beams": {
                    "type": "integer",
                    "description": "Beam search beams (upstream --num_beams).",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 64,
                    "required": False,
                },
            }
        }

