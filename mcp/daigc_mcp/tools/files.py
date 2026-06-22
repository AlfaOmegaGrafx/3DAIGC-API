"""File upload tools."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

from daigc_mcp.client import DaigcClient
from daigc_mcp.helpers import load_image_bytes, load_mesh_bytes, mesh_content_type
from daigc_mcp.spec import ToolSpec

IMAGE_SOURCE_SCHEMA = {
    "type": "string",
    "enum": ["base64", "file_path", "url"],
    "description": "How to interpret the data field",
}


async def upload_image(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    source = args.get("source")
    data = args.get("data")
    if not source or not data:
        raise ValueError("source and data are required")

    max_side = int(args.get("max_side") or settings.max_image_side)
    image_bytes, filename = await load_image_bytes(
        client,
        source=str(source),
        data=str(data),
        max_side=max_side,
    )
    result = await client.upload_image_bytes(image_bytes, filename)
    return {
        **result,
        "note": "Use file_id in image_to_textured_mesh or image_to_world.",
    }


async def upload_mesh(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = settings
    source = args.get("source")
    data = args.get("data")
    if not source or not data:
        raise ValueError("source and data are required")

    mesh_bytes, filename = await load_mesh_bytes(
        client,
        source=str(source),
        data=str(data),
    )
    content_type = mesh_content_type(filename)
    result = await client.upload_mesh_bytes(mesh_bytes, filename, content_type)
    return {
        **result,
        "note": "Use file_id in generate_rig or mesh editing tools.",
    }


TOOLS: list[ToolSpec] = [
    ToolSpec(
        Tool(
            name="upload_image",
            description=(
                "Upload an image to 3DAIGC file store. Returns file_id (24h TTL). "
                "Downscales to max_side before upload."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": IMAGE_SOURCE_SCHEMA,
                    "data": {
                        "type": "string",
                        "description": "Base64 payload, local path, or HTTP URL",
                    },
                    "max_side": {
                        "type": "integer",
                        "description": "Max width/height in pixels (default from env)",
                    },
                },
                "required": ["source", "data"],
            },
        ),
        upload_image,
    ),
    ToolSpec(
        Tool(
            name="upload_mesh",
            description=(
                "Upload a mesh (GLB/OBJ/FBX) to 3DAIGC file store. Returns file_id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["file_path", "url"],
                    },
                    "data": {
                        "type": "string",
                        "description": "Local filesystem path or HTTP URL",
                    },
                },
                "required": ["source", "data"],
            },
        ),
        upload_mesh,
    ),
]
