"""Shared helpers for MCP tool handlers."""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image

from daigc_mcp.client import DaigcApiError, DaigcClient


def tool_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, DaigcApiError):
        return exc.to_dict()
    return {"error": str(exc)}


def job_submit_envelope(
    payload: dict[str, Any],
    *,
    feature: str,
) -> dict[str, Any]:
    return {
        **payload,
        "feature": feature,
        "poll_tool": "get_job_status",
        "wait_tool": "wait_for_job",
        "message": payload.get("message")
        or f"{feature} job queued. Use wait_for_job to block until complete.",
    }


async def resolve_model_preference(
    client: DaigcClient,
    feature: str,
    model_preference: str | None,
) -> str:
    if model_preference:
        return model_preference
    default = await client.resolve_default_model(feature)
    if not default:
        raise DaigcApiError(
            f"No models available for feature '{feature}'. "
            "Call list_models to inspect the server."
        )
    return default


def downscale_image_bytes(data: bytes, max_side: int) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        width, height = img.size
        if max(width, height) <= max_side:
            fmt = "JPEG"
            buf = io.BytesIO()
            img.save(buf, format=fmt, quality=90)
            return buf.getvalue(), "image/jpeg"

        scale = max_side / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


async def load_image_bytes(
    client: DaigcClient,
    *,
    source: str,
    data: str,
    max_side: int,
) -> tuple[bytes, str]:
    source_norm = source.lower().strip()
    if source_norm == "base64":
        raw = data
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        image_bytes = base64.b64decode(raw)
        filename = "capture.jpg"
    elif source_norm == "file_path":
        path = Path(data).expanduser()
        if not path.is_file():
            raise DaigcApiError(f"Image file not found: {path}")
        image_bytes = path.read_bytes()
        filename = path.name
    elif source_norm == "url":
        image_bytes = await client.fetch_bytes(data)
        filename = Path(data.split("?", 1)[0]).name or "capture.jpg"
    else:
        raise DaigcApiError(
            "source must be one of: base64, file_path, url "
            f"(got {source!r})"
        )

    scaled, content_type = downscale_image_bytes(image_bytes, max_side)
    if not filename.lower().endswith((".jpg", ".jpeg")):
        filename = f"{Path(filename).stem}.jpg"
    return scaled, filename


def mesh_content_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


async def load_mesh_bytes(
    client: DaigcClient,
    *,
    source: str,
    data: str,
) -> tuple[bytes, str]:
    source_norm = source.lower().strip()
    if source_norm == "file_path":
        path = Path(data).expanduser()
        if not path.is_file():
            raise DaigcApiError(f"Mesh file not found: {path}")
        return path.read_bytes(), path.name
    if source_norm == "url":
        content = await client.fetch_bytes(data)
        filename = Path(data.split("?", 1)[0]).name or "model.glb"
        return content, filename
    raise DaigcApiError("upload_mesh source must be file_path or url")


def summarize_completed_job(job: dict[str, Any]) -> str:
    status = job.get("status", "unknown")
    if status != "completed":
        return f"Job {job.get('job_id')} ended with status {status}."

    result = job.get("result") or {}
    mesh_info = result.get("mesh_file_info") or {}
    size_mb = mesh_info.get("file_size_mb")
    if size_mb is not None:
        return f"Job completed. Mesh ready ({size_mb} MB)."
    if result.get("mesh_url"):
        return "Job completed. Mesh download URL is available."
    if result.get("world_manifest_url") or result.get("manifest_url"):
        return "Job completed. World manifest is ready."
    return "Job completed successfully."


def job_result_urls(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") or {}
    out: dict[str, Any] = {}
    for key in ("mesh_url", "thumbnail_url", "download_url", "manifest_url"):
        if result.get(key):
            out[key] = result[key]
    return out
