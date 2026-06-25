"""HTTP client for 3DAIGC-API."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx


class DaigcApiError(Exception):
    """Raised when the 3DAIGC-API returns a non-success response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": str(self)}
        if self.status_code is not None:
            out["status_code"] = self.status_code
        if self.detail is not None:
            out["detail"] = self.detail
        return out


class DaigcClient:
    """Async wrapper around 3DAIGC-API REST endpoints."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout_sec: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._timeout = httpx.Timeout(timeout_sec)

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params, headers=self._headers())
        return self._parse_response(response)

    async def post_json(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                json=body,
                headers=self._headers(json_body=True),
            )
        return self._parse_response(response)

    async def post_multipart(
        self,
        path: str,
        *,
        files: dict[str, tuple[str, bytes, str]],
    ) -> Any:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                files=files,
                headers=self._headers(),
            )
        return self._parse_response(response)

    async def fetch_bytes(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=self._headers())
        if response.status_code >= 400:
            raise DaigcApiError(
                f"Failed to fetch URL ({response.status_code})",
                status_code=response.status_code,
                detail=response.text,
            )
        return response.content

    async def poll_job(
        self,
        job_id: str,
        *,
        timeout_sec: float,
        poll_interval_sec: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        last: dict[str, Any] = {}

        while True:
            last = await self.get_job(job_id)
            status = (last.get("status") or "").lower()
            if status in {"completed", "failed", "cancelled"}:
                return last
            if time.monotonic() >= deadline:
                raise DaigcApiError(
                    f"Job {job_id} did not finish within {timeout_sec}s "
                    f"(last status: {status or 'unknown'})",
                    detail={"job_id": job_id, "last_status": last},
                )
            await asyncio.sleep(poll_interval_sec)

    async def health_check(self) -> dict[str, Any]:
        return await self.get("/health")

    async def list_features(self) -> dict[str, Any]:
        return await self.get("/api/v1/system/features")

    async def list_models(self, feature: str | None = None) -> dict[str, Any]:
        params = {"feature": feature} if feature else None
        return await self.get("/api/v1/system/models", params=params)

    async def get_model_parameters(self, model_id: str) -> dict[str, Any]:
        return await self.get(f"/api/v1/system/models/{model_id}/parameters")

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self.get(f"/api/v1/system/jobs/{job_id}")

    async def upload_image_bytes(
        self,
        data: bytes,
        filename: str,
        content_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        return await self.post_multipart(
            "/api/v1/file-upload/image",
            files={"file": (filename, data, content_type)},
        )

    async def upload_mesh_bytes(
        self,
        data: bytes,
        filename: str,
        content_type: str = "model/gltf-binary",
    ) -> dict[str, Any]:
        return await self.post_multipart(
            "/api/v1/file-upload/mesh",
            files={"file": (filename, data, content_type)},
        )

    async def text_to_textured_mesh(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self.post_json(
            "/api/v1/mesh-generation/text-to-textured-mesh",
            body,
        )

    async def image_to_textured_mesh(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self.post_json(
            "/api/v1/mesh-generation/image-to-textured-mesh",
            body,
        )

    async def generate_rig(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self.post_json("/api/v1/auto-rigging/generate-rig", body)

    async def image_to_world(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self.post_json(
            "/api/v1/world-generation/image-to-world",
            body,
        )

    async def resolve_default_model(self, feature: str) -> str | None:
        payload = await self.list_models(feature)
        # Filtered query returns {"feature": ..., "models": [...]}
        if isinstance(payload.get("models"), list):
            entries = payload["models"]
        else:
            # Unfiltered query returns {"available_models": {feature: [...]}}
            models = payload.get("available_models") or {}
            entries = models.get(feature) or []
        return entries[0] if entries else None

    @staticmethod
    def _parse_response(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json()
            except json.JSONDecodeError:
                detail = response.text
            message = DaigcClient._error_message(detail, response.status_code)
            raise DaigcApiError(message, status_code=response.status_code, detail=detail)
        if not response.content:
            return {}
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"raw": response.text}

    @staticmethod
    def _error_message(detail: Any, status_code: int) -> str:
        if isinstance(detail, dict):
            if detail.get("message"):
                return str(detail["message"])
            if detail.get("detail"):
                return str(detail["detail"])
        return f"3DAIGC-API request failed with status {status_code}"
