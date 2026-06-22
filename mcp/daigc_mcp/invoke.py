"""CLI smoke-test harness for 3daigc-mcp tools (no Cursor required)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from daigc_mcp.runtime import format_tool_result, run_tool


def _parse_args(raw: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
    while i < len(raw):
        token = raw[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if i + 1 >= len(raw) or raw[i + 1].startswith("--"):
            out[key] = True
            i += 1
            continue
        value: Any = raw[i + 1]
        if value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
        out[key] = value
        i += 2
    return out


async def _main_async(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Invoke a 3daigc-mcp tool directly against 3DAIGC-API",
    )
    parser.add_argument("tool", help="Tool name, e.g. health_check")
    args, unknown = parser.parse_known_args(argv)
    payload = await run_tool(args.tool, _parse_args(unknown))
    print(format_tool_result(payload))
    return 1 if isinstance(payload, dict) and payload.get("error") else 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(_main_async(argv or sys.argv[1:])))


if __name__ == "__main__":
    main()
