"""Shared tool dispatch for stdio MCP, CLI invoke, and HTTP modes."""

from __future__ import annotations

import json
from typing import Any

from daigc_mcp.client import DaigcClient
from daigc_mcp.config import Settings, get_settings
from daigc_mcp.helpers import tool_error
from daigc_mcp.registry import tool_map


async def run_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    settings: Settings | None = None,
) -> Any:
    cfg = settings or get_settings()
    client = DaigcClient(cfg.api_base_url, cfg.api_token)
    spec = tool_map().get(name)
    if spec is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return await spec.handler(client, arguments or {}, cfg)
    except Exception as exc:
        return tool_error(exc)


def format_tool_result(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)
