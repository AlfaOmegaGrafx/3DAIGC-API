"""Environment-based configuration for 3daigc-mcp."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    api_token: str | None
    poll_interval_sec: float
    default_job_timeout_sec: float
    max_image_side: int

    @classmethod
    def from_env(cls) -> Settings:
        base = os.environ.get("DAIGC_API_BASE_URL", "http://localhost:7842").rstrip("/")
        token = os.environ.get("DAIGC_API_TOKEN") or None
        if token is not None and not token.strip():
            token = None
        return cls(
            api_base_url=base,
            api_token=token,
            poll_interval_sec=float(os.environ.get("DAIGC_MCP_POLL_INTERVAL_SEC", "3")),
            default_job_timeout_sec=float(
                os.environ.get("DAIGC_MCP_DEFAULT_JOB_TIMEOUT_SEC", "600")
            ),
            max_image_side=int(os.environ.get("DAIGC_MCP_MAX_IMAGE_SIDE", "2048")),
        )


def get_settings() -> Settings:
    return Settings.from_env()
