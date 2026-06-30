#!/usr/bin/env python3
"""
Model verify registry — config-driven checks for every enabled model.

Tiers:
  quick  — import adapter class (startup preflight)
  load   — load weights on GPU, no inference
  infer  — full load + one process() call (verify_all matrix)

Usage:
  python scripts/verify_registry.py --validate
  python scripts/verify_registry.py --tier quick --all-enabled
  python scripts/verify_registry.py --tier infer --model trellis2_image_to_textured_mesh
  python scripts/verify_registry.py --tier infer --all-enabled --skip-heavy
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.scheduler.model_factory import ModelFactory
from core.utils.gpu_env import apply_local_gpu_env

HEAVY_INFER = {
    "krea2_turbo_text_to_image",
    "trellis2_image_to_textured_mesh",
    "pixal3d_image_to_textured_mesh",
    "opennexus_image_to_world",
    "p3sam_mesh_segmentation",
    "voxhammer_text_mesh_editing",
    "voxhammer_image_mesh_editing",
    "kimodo_text_to_motion",
    "skintokens_auto_rig",
}


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> int:
    print(f"  FAIL {msg}")
    return 1


def warn(msg: str) -> None:
    print(f"  WARN {msg}")


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    with open(ROOT / "config" / "models.yaml") as f:
        models_cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "verify_profiles.yaml") as f:
        verify_cfg = yaml.safe_load(f)
    fixtures = verify_cfg.get("fixtures") or {}
    profiles = verify_cfg.get("profiles") or {}
    return models_cfg, profiles, fixtures


def enabled_model_ids(models_cfg: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for _feature, models in models_cfg.items():
        if not isinstance(models, dict):
            continue
        for model_id, spec in models.items():
            if isinstance(spec, dict) and spec.get("enabled", True):
                ids.append(model_id)
    return sorted(ids)


def resolve_inputs(raw: Dict[str, Any], fixtures: Dict[str, str]) -> Dict[str, Any]:
    def resolve_value(val: Any) -> Any:
        if isinstance(val, str) and val.startswith("{") and val.endswith("}"):
            key = val[1:-1]
            path = fixtures.get(key, key)
            return str(ROOT / path)
        if isinstance(val, dict):
            return {k: resolve_value(v) for k, v in val.items()}
        if isinstance(val, list):
            return [resolve_value(v) for v in val]
        return val

    return resolve_value(deepcopy(raw))


def validate_registry() -> int:
    models_cfg, profiles, _fixtures = load_configs()
    enabled = enabled_model_ids(models_cfg)
    errors = 0

    print("=== Verify registry validation ===")
    for model_id in enabled:
        if model_id not in profiles:
            errors += fail(f"{model_id}: enabled in models.yaml but missing verify profile")
            continue
        prof = profiles[model_id]
        reg = ModelFactory.ADAPTER_REGISTRY.get(model_id)
        if not reg:
            errors += fail(f"{model_id}: missing from ModelFactory.ADAPTER_REGISTRY")
            continue
        if prof.get("module") != reg["module"] or prof.get("class") != reg["class"]:
            errors += fail(
                f"{model_id}: profile module/class mismatch "
                f"({prof.get('module')}.{prof.get('class')} vs "
                f"{reg['module']}.{reg['class']})"
            )
            continue
        ok(f"{model_id} profile matches registry")

    for model_id in profiles:
        if model_id not in enabled:
            warn(f"{model_id}: verify profile exists but model is disabled")

    print("\n=== Summary ===")
    if errors:
        print(f"REGISTRY_VALIDATE_FAIL ({errors} errors)")
        return 1
    print(f"REGISTRY_VALIDATE_OK ({len(enabled)} enabled models)")
    return 0


def run_quick(model_id: str, prof: Dict[str, Any]) -> int:
    mod = importlib.import_module(prof["module"])
    cls = getattr(mod, prof["class"])
    ok(f"{model_id}: {prof['module']}.{prof['class']}")
    return 0


def run_load(model_id: str, prof: Dict[str, Any]) -> int:
    mod = importlib.import_module(prof["module"])
    cls = getattr(mod, prof["class"])
    adapter = cls()
    t0 = time.time()
    adapter.load(gpu_id=0)
    print(f"  OK  {model_id}: loaded in {time.time() - t0:.1f}s")
    try:
        adapter.unload()
    except Exception:
        pass
    return 0


def run_infer(model_id: str, prof: Dict[str, Any], fixtures: Dict[str, str]) -> int:
    inputs = resolve_inputs(prof.get("inputs") or {}, fixtures)
    timeout = int(prof.get("timeout_sec", 3600))
    mod_name = prof["module"]
    cls_name = prof["class"]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "verify_model.py"),
        mod_name,
        cls_name,
        json.dumps(inputs),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
        fail(f"{model_id}: infer failed\n      " + "\n      ".join(tail))
        return 1
    if "VERIFY_OK" not in proc.stdout:
        fail(f"{model_id}: subprocess OK but VERIFY_OK missing")
        return 1
    ok(f"{model_id}: infer OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Config-driven model verification")
    parser.add_argument("--validate", action="store_true", help="Check profiles vs models.yaml")
    parser.add_argument("--tier", choices=("quick", "load", "infer"), default="quick")
    parser.add_argument("--model", action="append", dest="models", help="Single model_id")
    parser.add_argument("--all-enabled", action="store_true", help="All enabled models")
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip HEAVY_INFER models in infer tier (weekly quick matrix)",
    )
    args = parser.parse_args()

    if args.validate:
        return validate_registry()

    apply_local_gpu_env(ROOT)

    models_cfg, profiles, fixtures = load_configs()
    if args.all_enabled:
        target_ids = enabled_model_ids(models_cfg)
    elif args.models:
        target_ids = args.models
    else:
        parser.error("Specify --all-enabled, --model ID, or --validate")

    print(f"=== Verify registry tier={args.tier} ({len(target_ids)} models) ===")
    errors = 0
    for model_id in target_ids:
        if model_id not in profiles:
            errors += fail(f"{model_id}: no verify profile")
            continue
        if args.tier == "infer" and args.skip_heavy and model_id in HEAVY_INFER:
            warn(f"{model_id}: skipped (heavy)")
            continue
        prof = profiles[model_id]
        try:
            if args.tier == "quick":
                errors += run_quick(model_id, prof)
            elif args.tier == "load":
                errors += run_load(model_id, prof)
            else:
                errors += run_infer(model_id, prof, fixtures)
        except subprocess.TimeoutExpired:
            errors += fail(f"{model_id}: timeout")
        except Exception as exc:
            errors += fail(f"{model_id}: {exc}")

    print("\n=== Summary ===")
    if errors:
        print(f"VERIFY_REGISTRY_FAIL ({errors} errors)")
        return 1
    print("VERIFY_REGISTRY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
