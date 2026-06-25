"""MCP stdio server entrypoint for 3DAIGC-API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from daigc_mcp.runtime import format_tool_result, run_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("3daigc-mcp")

server = Server("3daigc-mcp")


def _encode_result(payload: Any) -> list[TextContent]:
    return [TextContent(type="text", text=format_tool_result(payload))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    from daigc_mcp.registry import tool_map

    return [spec.tool for spec in tool_map().values()]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    payload = await run_tool(name, arguments)
    if isinstance(payload, dict) and payload.get("error"):
        logger.error("Tool %s error: %s", name, payload.get("error"))
    return _encode_result(payload)


async def run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
