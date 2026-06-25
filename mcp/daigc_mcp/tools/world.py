"""World generation MCP tools."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

from daigc_mcp.client import DaigcClient
from daigc_mcp.helpers import job_submit_envelope, resolve_model_preference
from daigc_mcp.spec import ToolSpec

FEATURE = "image_to_world"


async def image_to_world(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = settings
    image_file_id = args.get("image_file_id")
    image_base64 = args.get("image_base64")
    image_path = args.get("image_path")
    provided = sum(bool(x) for x in (image_file_id, image_base64, image_path))
    if provided != 1:
        raise ValueError(
            "Provide exactly one of: image_file_id, image_base64, image_path"
        )

    model = await resolve_model_preference(
        client,
        FEATURE,
        args.get("model_preference"),
    )
    body: dict[str, Any] = {"model_preference": model}
    if image_file_id:
        body["image_file_id"] = image_file_id
    elif image_base64:
        body["image_base64"] = image_base64
    else:
        body["image_path"] = image_path

    if args.get("world_name"):
        body["world_name"] = args["world_name"]
    if args.get("world_id"):
        body["world_id"] = args["world_id"]
    if args.get("prop_regions"):
        body["prop_regions"] = args["prop_regions"]
    if args.get("prop_mesh_model_preference"):
        body["prop_mesh_model_preference"] = args["prop_mesh_model_preference"]

    payload = await client.image_to_world(body)
    envelope = job_submit_envelope(payload, feature=FEATURE)
    envelope["manifest_hint"] = (
        f"After wait_for_job, fetch manifest via "
        f"GET /api/v1/system/jobs/{{job_id}}/download?asset=manifest"
    )
    return envelope


TOOLS: list[ToolSpec] = [
    ToolSpec(
        Tool(
            name="image_to_world",
            description=(
                "Queue explorable world package generation (splat env + optional props) "
                "for OpenNexus3DStudio / Galaxy XR."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_file_id": {"type": "string"},
                    "image_base64": {"type": "string"},
                    "image_path": {"type": "string"},
                    "world_name": {"type": "string"},
                    "world_id": {"type": "string"},
                    "model_preference": {"type": "string"},
                    "prop_mesh_model_preference": {"type": "string"},
                    "prop_regions": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
            },
        ),
        image_to_world,
    ),
]
