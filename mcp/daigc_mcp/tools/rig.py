"""Auto-rigging MCP tools."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

from daigc_mcp.client import DaigcClient
from daigc_mcp.helpers import job_submit_envelope, resolve_model_preference
from daigc_mcp.spec import ToolSpec

FEATURE = "auto_rig"


async def generate_rig(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = settings
    mesh_file_id = args.get("mesh_file_id")
    mesh_path = args.get("mesh_path")
    if not mesh_file_id and not mesh_path:
        raise ValueError("mesh_file_id or mesh_path is required")
    if mesh_file_id and mesh_path:
        raise ValueError("Provide only one of mesh_file_id or mesh_path")

    model = await resolve_model_preference(
        client,
        FEATURE,
        args.get("model_preference"),
    )
    body: dict[str, Any] = {
        "rig_mode": args.get("rig_mode") or "template",
        "output_format": args.get("output_format") or "glb",
        "model_preference": model,
    }
    if mesh_file_id:
        body["mesh_file_id"] = mesh_file_id
    else:
        body["mesh_path"] = mesh_path

    if args.get("humanoid_template_id"):
        body["humanoid_template_id"] = args["humanoid_template_id"]
    if args.get("model_parameters"):
        body["model_parameters"] = args["model_parameters"]

    payload = await client.generate_rig(body)
    return job_submit_envelope(payload, feature=FEATURE)


TOOLS: list[ToolSpec] = [
    ToolSpec(
        Tool(
            name="generate_rig",
            description=(
                "Queue auto-rigging for a mesh. Use rig_mode=template for OpenNexus "
                "humanoid VRM pipeline. Requires mesh_file_id from upload_mesh or a "
                "completed mesh job."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mesh_file_id": {"type": "string"},
                    "mesh_path": {
                        "type": "string",
                        "description": "Server-side path on DGX",
                    },
                    "rig_mode": {
                        "type": "string",
                        "enum": ["skeleton", "skin", "full", "template"],
                        "default": "template",
                    },
                    "humanoid_template_id": {
                        "type": "string",
                        "default": "template",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["glb", "fbx"],
                        "default": "glb",
                    },
                    "model_preference": {"type": "string"},
                    "model_parameters": {"type": "object"},
                },
            },
        ),
        generate_rig,
    ),
]
