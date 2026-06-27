"""Human-readable object names on 3DAIGC generation jobs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

OBJECT_NAME_MAX_LEN = 64


class ObjectNamed(BaseModel):
    object_name: Optional[str] = Field(
        None,
        max_length=OBJECT_NAME_MAX_LEN,
        description="Human-readable name for the generated 3D object",
    )


def _clean_object_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:OBJECT_NAME_MAX_LEN]


def enrich_job_inputs(inputs: Dict[str, Any], object_name: Optional[str]) -> Dict[str, Any]:
    cleaned = _clean_object_name(object_name)
    if not cleaned:
        return inputs
    out = dict(inputs)
    out["object_name"] = cleaned
    return out


def enrich_job_metadata(feature_type: str, object_name: Optional[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"feature_type": feature_type}
    cleaned = _clean_object_name(object_name)
    if cleaned:
        meta["object_name"] = cleaned
    return meta
