"""
Mage-Flow-Edit helpers — isolated venv (transformers 5.x) so main API stays on 4.57.

Weights default: mage-flow-community/Mage-Flow-Edit-Turbo (official microsoft/* host withdrawn).
Code: thirdparty/Mage/mage_flow (MIT).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAGE_ROOT = Path(
    os.environ.get("MAGE_FLOW_ROOT", str(REPO_ROOT / "thirdparty" / "Mage"))
)
DEFAULT_WEIGHTS_DIR = Path(
    os.environ.get(
        "MAGE_FLOW_EDIT_WEIGHTS",
        str(REPO_ROOT / "pretrained" / "mage-flow" / "Mage-Flow-Edit-Turbo"),
    )
)
DEFAULT_HF_ID = "mage-flow-community/Mage-Flow-Edit-Turbo"
DEFAULT_HF_REVISION = "66df6fa1aba5b40cd4120739134292eab9779da3"
DEFAULT_VENV = REPO_ROOT / ".venv-mage-flow"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "mage_flow_edit_runner.py"


def resolve_mage_flow_python() -> Optional[str]:
    env = (os.environ.get("MAGE_FLOW_PYTHON") or "").strip()
    if env and Path(env).is_file():
        return env
    venv_py = DEFAULT_VENV / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return None


def probe_mage_flow_edit_install(
    mage_root: Optional[Path] = None,
    weights_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    mage_root = Path(mage_root or DEFAULT_MAGE_ROOT)
    weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    python_bin = resolve_mage_flow_python()
    reasons: List[str] = []

    pkg = mage_root / "mage_flow"
    if not (pkg / "pipeline.py").is_file():
        reasons.append(
            f"missing mage_flow package under {mage_root} — run: bash scripts/setup_mage_flow_edit.sh"
        )
    if not (weights_dir / "model_index.json").is_file():
        reasons.append(
            f"missing weights at {weights_dir} — run: bash scripts/setup_mage_flow_edit.sh"
        )
    if not python_bin:
        reasons.append(
            "set MAGE_FLOW_PYTHON or create .venv-mage-flow "
            "(bash scripts/setup_mage_flow_edit.sh)"
        )
    elif not RUNNER_SCRIPT.is_file():
        reasons.append(f"missing runner {RUNNER_SCRIPT}")
    else:
        probe = (
            "from mage_flow import MageFlowPipeline; "
            "from mage_flow.models.modules._attn_backend import set_attn_backend; "
            "set_attn_backend('sdpa'); "
            "print('ok')"
        )
        try:
            proc = subprocess.run(
                [python_bin, "-c", probe],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                cwd=str(mage_root),
                env=_runner_env(mage_root),
            )
        except Exception as exc:
            reasons.append(f"mage venv probe failed: {exc}")
        else:
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "import failed").strip().splitlines()
                reasons.append(err[-1] if err else "mage_flow import failed in isolated venv")

    return {
        "integrated": len(reasons) == 0,
        "blocking_reasons": reasons,
        "mage_root": str(mage_root),
        "weights_dir": str(weights_dir),
        "python": python_bin,
        "hf_model_id": os.environ.get("MAGE_FLOW_EDIT_HF_ID", DEFAULT_HF_ID),
        "hf_revision": os.environ.get("MAGE_FLOW_EDIT_HF_REVISION", DEFAULT_HF_REVISION),
    }


def _runner_env(mage_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    # Prefer SDPA on Spark/aarch64 when flash-attn is unavailable.
    env.setdefault("MAGE_FLOW_ATTN_BACKEND", "sdpa")
    env.setdefault("VF_HF_ATTN_IMPL", "sdpa")
    existing = env.get("PYTHONPATH", "")
    root = str(mage_root.resolve())
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env


def run_mage_flow_edit(
    *,
    image_path: str,
    text_prompt: str,
    output_path: str,
    model_path: Optional[str] = None,
    steps: int = 4,
    cfg: float = 1.0,
    max_size: int = 1024,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = 42,
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    status = probe_mage_flow_edit_install()
    if not status.get("integrated"):
        raise RuntimeError(
            "Mage-Flow-Edit not ready: " + "; ".join(status.get("blocking_reasons") or [])
        )

    python_bin = status["python"]
    mage_root = Path(status["mage_root"])
    weights = Path(model_path or status["weights_dir"]).resolve()
    if not (weights / "model_index.json").is_file():
        raise RuntimeError(f"Mage weights missing model_index.json at {weights}")
    image_abs = str(Path(image_path).resolve())
    output_abs = str(Path(output_path).resolve())
    payload = {
        "image_path": image_abs,
        "text_prompt": text_prompt,
        "output_path": output_abs,
        "model_path": str(weights),
        "steps": steps,
        "cfg": cfg,
        "max_size": max_size,
        "width": width,
        "height": height,
        "seed": seed,
        "attn_backend": os.environ.get("MAGE_FLOW_ATTN_BACKEND", "sdpa"),
    }

    proc = subprocess.run(
        [python_bin, str(RUNNER_SCRIPT), json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
        cwd=str(mage_root),
        env=_runner_env(mage_root),
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "mage edit failed").strip()
        raise RuntimeError(f"Mage-Flow-Edit subprocess failed:\n{err[-4000:]}")

    try:
        result = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(
            f"Mage-Flow-Edit returned invalid JSON: {(proc.stdout or '')[-500:]}"
        ) from exc
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Mage-Flow-Edit reported failure")
    return result
