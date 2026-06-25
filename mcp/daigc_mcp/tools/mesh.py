"""Mesh generation MCP tools."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

from daigc_mcp.client import DaigcClient
from daigc_mcp.helpers import job_submit_envelope, resolve_model_preference
from daigc_mcp.spec import ToolSpec

TEXT_FEATURE = "text_to_textured_mesh"
IMAGE_FEATURE = "image_to_textured_mesh"


async def text_to_textured_mesh(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = settings
    text_prompt = args.get("text_prompt")
    if not text_prompt:
        raise ValueError("text_prompt is required")

    model = await resolve_model_preference(
        client,
        TEXT_FEATURE,
        args.get("model_preference"),
    )
    body: dict[str, Any] = {
        "text_prompt": text_prompt,
        "output_format": args.get("output_format") or "glb",
        "model_preference": model,
    }
    if args.get("texture_prompt") is not None:
        body["texture_prompt"] = args["texture_prompt"]
    if args.get("texture_resolution") is not None:
        body["texture_resolution"] = args["texture_resolution"]
    if args.get("model_parameters"):
        body["model_parameters"] = args["model_parameters"]

    payload = await client.text_to_textured_mesh(body)
    return job_submit_envelope(payload, feature=TEXT_FEATURE)


async def image_to_textured_mesh(
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
        IMAGE_FEATURE,
        args.get("model_preference"),
    )
    body: dict[str, Any] = {
        "output_format": args.get("output_format") or "glb",
        "model_preference": model,
    }
    if image_file_id:
        body["image_file_id"] = image_file_id
    elif image_base64:
        body["image_base64"] = image_base64
    else:
        body["image_path"] = image_path

    if args.get("texture_prompt"):
        body["texture_prompt"] = args["texture_prompt"]
    if args.get("texture_resolution") is not None:
        body["texture_resolution"] = args["texture_resolution"]
    if args.get("model_parameters"):
        body["model_parameters"] = args["model_parameters"]

    payload = await client.image_to_textured_mesh(body)
    return job_submit_envelope(payload, feature=IMAGE_FEATURE)


TOOLS: list[ToolSpec] = [
    ToolSpec(
        Tool(
            name="text_to_textured_mesh",
            description=(
                "Queue text-to-textured-mesh generation. Returns job_id; "
                "call wait_for_job for the GLB download URL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text_prompt": {"type": "string"},
                    "texture_prompt": {"type": "string"},
                    "texture_resolution": {"type": "integer"},
                    "output_format": {"type": "string", "default": "glb"},
                    "model_preference": {
                        "type": "string",
                        "description": "Defaults to first model from list_models",
                    },
                    "model_parameters": {"type": "object"},
                },
                "required": ["text_prompt"],
            },
        ),
        text_to_textured_mesh,
    ),
    ToolSpec(
        Tool(
            name="image_to_textured_mesh",
            description=(
                "Queue image-to-textured-mesh generation. Prefer image_file_id "
                "from upload_image."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_file_id": {"type": "string"},
                    "image_base64": {"type": "string"},
                    "image_path": {
                        "type": "string",
                        "description": "Server-side path (DGX local files only)",
                    },
                    "texture_prompt": {"type": "string"},
                    "texture_resolution": {"type": "integer"},
                    "output_format": {"type": "string", "default": "glb"},
                    "model_preference": {"type": "string"},
                    "model_parameters": {"type": "object"},
                },
            },
        ),
        image_to_textured_mesh,
    ),
]
