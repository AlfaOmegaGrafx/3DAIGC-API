#!/usr/bin/env python3
"""Resume mesh + appearance_component rig from a completed text_to_image job."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = "http://127.0.0.1:7842"


def poll_job(client: httpx.Client, job_id: str, label: str, timeout_s: int = 3600) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/api/v1/system/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        print(f"[{label}] {job_id[:8]}… status={status}", flush=True)
        if status == "completed":
            return data
        if status in ("failed", "cancelled", "error"):
            raise RuntimeError(f"{label} failed: {json.dumps(data, indent=2)[:2000]}")
        time.sleep(5)
    raise TimeoutError(f"{label} timed out after {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-job-id", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--appearance-slot", default="Head")
    parser.add_argument("--api-base", default=DEFAULT_API)
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--mesh-job-id", default="")
    args = parser.parse_args()

    api = args.api_base.rstrip("/")
    with httpx.Client(base_url=api, timeout=300.0) as client:
        image_job = client.get(f"/api/v1/system/jobs/{args.image_job_id}").json()
        if image_job.get("status") != "completed":
            print(f"Image job not completed: {image_job.get('status')}", file=sys.stderr)
            return 1

        result = image_job.get("result") or {}
        rel_path = result.get("output_image_path") or result.get("output_mesh_path")
        if not rel_path:
            print("No output_image_path on image job", file=sys.stderr)
            return 1
        image_path = str((ROOT / rel_path).resolve())
        if not Path(image_path).is_file():
            print(f"Image missing: {image_path}", file=sys.stderr)
            return 1
        print(f"Using image: {image_path}")

        mesh_job_id = args.mesh_job_id.strip()
        if not args.skip_mesh and not mesh_job_id:
            mesh_resp = client.post(
                "/api/v1/mesh-generation/image-to-textured-mesh",
                json={
                    "image_path": image_path,
                    "output_format": "glb",
                    "model_preference": "trellis2_image_to_textured_mesh",
                    "object_name": args.object_name,
                },
            )
            mesh_resp.raise_for_status()
            mesh_job_id = mesh_resp.json()["job_id"]
            print(f"Mesh job queued: {mesh_job_id}")

        mesh_job = poll_job(client, mesh_job_id, "mesh")
        mesh_result = mesh_job.get("result") or {}
        mesh_rel = mesh_result.get("output_mesh_path") or mesh_result.get("mesh_path")
        if not mesh_rel:
            print(f"No mesh path in result: {json.dumps(mesh_result)[:1500]}", file=sys.stderr)
            return 1
        mesh_path = str((ROOT / mesh_rel).resolve() if not str(mesh_rel).startswith("/") else Path(mesh_rel))
        if not Path(mesh_path).is_file():
            print(f"Mesh missing: {mesh_path}", file=sys.stderr)
            return 1
        print(f"Mesh ready: {mesh_path}")

        rig_payload = {
            "mesh_path": mesh_path,
            "rig_mode": "appearance_component",
            "appearance_slot": args.appearance_slot,
            "output_format": "glb",
            "model_preference": "appearance_component_auto_rig",
            "object_name": args.object_name,
        }
        rig_resp = client.post("/api/v1/auto-rigging/generate-rig", json=rig_payload)
        rig_resp.raise_for_status()
        rig_job_id = rig_resp.json()["job_id"]
        print(f"Rig job queued: {rig_job_id}")

        rig_job = poll_job(client, rig_job_id, "rig", timeout_s=1800)
        rig_result = rig_job.get("result") or {}
        trait_rel = (
            rig_result.get("output_mesh_path")
            or rig_result.get("rigged_mesh_path")
            or rig_result.get("mesh_path")
        )
        print(
            json.dumps(
                {
                    "image_job_id": args.image_job_id,
                    "mesh_job_id": mesh_job_id,
                    "rig_job_id": rig_job_id,
                    "object_name": args.object_name,
                    "appearance_slot": args.appearance_slot,
                    "trait_path": trait_rel,
                    "download_url": f"/api/v1/system/jobs/{rig_job_id}/download",
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
