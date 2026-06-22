"""FastMCP HTTP server for XR AI worker integration (port 8260 default)."""

from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastmcp import FastMCP

from daigc_mcp.registry import all_tool_specs
from daigc_mcp.runtime import run_tool

mcp = FastMCP("3daigc-mcp")


def _build_tool_function(name: str, input_schema: dict[str, Any]):
    properties = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])

    arg_parts: list[str] = []
    for pname in properties:
        if pname in required:
            arg_parts.append(pname)
        else:
            arg_parts.append(f"{pname}=None")

    args_str = ", ".join(arg_parts)
    body = (
        "    kwargs = {k: v for k, v in locals().items() if v is not None}\n"
        f"    return await run_tool({name!r}, kwargs)\n"
    )
    src = f"async def {name}({args_str}):\n{body}"
    namespace: dict[str, Any] = {"run_tool": run_tool}
    exec(src, namespace)  # noqa: S102 — generated from trusted tool registry
    return namespace[name]


def _register_tools() -> None:
    for spec in all_tool_specs():
        schema = spec.tool.inputSchema or {"type": "object", "properties": {}}
        handler = _build_tool_function(spec.tool.name, schema)
        mcp.tool(name=spec.tool.name, description=spec.tool.description or "")(handler)


_register_tools()


def main() -> None:
    parser = argparse.ArgumentParser(description="3daigc-mcp HTTP (FastMCP) for XR AI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8260)
    args = parser.parse_args()

    app = mcp.http_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
