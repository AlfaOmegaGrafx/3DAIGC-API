"""Shared MCP tool specification types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from mcp.types import Tool

from daigc_mcp.client import DaigcClient

ToolHandler = Callable[[DaigcClient, dict[str, Any], Any], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    tool: Tool
    handler: ToolHandler
