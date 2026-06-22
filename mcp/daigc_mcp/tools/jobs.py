"""Job status and polling tools."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

from daigc_mcp.client import DaigcClient
from daigc_mcp.helpers import job_result_urls, summarize_completed_job
from daigc_mcp.spec import ToolSpec


async def get_job_status(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    _ = settings
    job_id = args.get("job_id")
    if not job_id:
        raise ValueError("job_id is required")
    job = await client.get_job(str(job_id))
    out: dict[str, Any] = {"job": job}
    if args.get("include_urls", True):
        out["urls"] = job_result_urls(job)
    if job.get("status") == "completed":
        out["summary"] = summarize_completed_job(job)
    return out


async def wait_for_job(
    client: DaigcClient,
    args: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    job_id = args.get("job_id")
    if not job_id:
        raise ValueError("job_id is required")

    timeout = float(args.get("timeout_sec") or settings.default_job_timeout_sec)
    interval = float(args.get("poll_interval_sec") or settings.poll_interval_sec)
    job = await client.poll_job(
        str(job_id),
        timeout_sec=timeout,
        poll_interval_sec=interval,
    )

    out: dict[str, Any] = {
        "job_id": job.get("job_id", job_id),
        "status": job.get("status"),
        "feature": job.get("feature") or (job.get("result") or {}).get("feature"),
        "processing_time_sec": job.get("processing_time"),
        "summary": summarize_completed_job(job),
    }
    if args.get("include_urls", True):
        out.update(job_result_urls(job))
    if job.get("status") == "failed":
        out["error"] = job.get("error") or job.get("message") or "Job failed"
        out["job"] = job
    elif args.get("include_full_job"):
        out["job"] = job
    return out


TOOLS: list[ToolSpec] = [
    ToolSpec(
        Tool(
            name="get_job_status",
            description="Poll a 3DAIGC job once. Returns status, result URLs when complete.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "include_urls": {
                        "type": "boolean",
                        "default": True,
                    },
                },
                "required": ["job_id"],
            },
        ),
        get_job_status,
    ),
    ToolSpec(
        Tool(
            name="wait_for_job",
            description=(
                "Block until a job completes, fails, or times out. "
                "Preferred for voice agents after submit tools."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "timeout_sec": {
                        "type": "number",
                        "description": "Max wait (default DAIGC_MCP_DEFAULT_JOB_TIMEOUT_SEC)",
                    },
                    "poll_interval_sec": {"type": "number"},
                    "include_urls": {"type": "boolean", "default": True},
                    "include_full_job": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include full job payload in response",
                    },
                },
                "required": ["job_id"],
            },
        ),
        wait_for_job,
    ),
]
