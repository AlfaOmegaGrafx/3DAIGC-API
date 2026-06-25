"""MCP tool registry for 3daigc-mcp v0."""

from __future__ import annotations

from daigc_mcp.spec import ToolSpec
from daigc_mcp.tools import discovery, files, jobs, mesh, rig, world


def all_tool_specs() -> list[ToolSpec]:
    return [
        *discovery.TOOLS,
        *files.TOOLS,
        *mesh.TOOLS,
        *rig.TOOLS,
        *world.TOOLS,
        *jobs.TOOLS,
    ]


def tool_map() -> dict[str, ToolSpec]:
    return {spec.tool.name: spec for spec in all_tool_specs()}
