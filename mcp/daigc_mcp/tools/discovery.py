"""Discovery tools: health, features, models."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

from daigc_mcp.client import DaigcClient
from daigc_mcp.helpers import tool_error
from daigc_mcp.spec import ToolSpec

FEATURE = "discovery"


async def health_check(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = args, settings
    return await client.health_check()


async def list_features(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = args, settings
    return await client.list_features()


async def list_models(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = settings
    feature = args.get("feature")
    return await client.list_models(feature)


async def get_model_parameters(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = settings
    model_id = args.get("model_id")
    if not model_id:
        raise ValueError("model_id is required")
    return await client.get_model_parameters(str(model_id))


TOOLS: list[ToolSpec] = [
    ToolSpec(
        Tool(
            name="health_check",
            description="Check whether 3DAIGC-API is reachable and healthy.",
            inputSchema={"type": "object", "properties": {}},
        ),
        health_check,
    ),
    ToolSpec(
        Tool(
            name="list_features",
            description="List supported 3DAIGC features (text_to_textured_mesh, auto_rig, etc.).",
            inputSchema={"type": "object", "properties": {}},
        ),
        list_features,
    ),
    ToolSpec(
        Tool(
            name="list_models",
            description=(
                "List available models, optionally filtered by feature. "
                "Use before submit tools when model_preference is unknown."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "feature": {
                        "type": "string",
                        "description": "Feature id, e.g. image_to_textured_mesh",
                    }
                },
            },
        ),
        list_models,
    ),
    ToolSpec(
        Tool(
            name="get_model_parameters",
            description="Get JSON schema for advanced model-specific parameters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model id from list_models",
                    }
                },
                "required": ["model_id"],
            },
        ),
        get_model_parameters,
    ),
]

__all__ = ["TOOLS", "tool_error"]
