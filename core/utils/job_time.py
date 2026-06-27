"""
Job timestamps for 3DAIGC-API task queue.

Internal clock: US Eastern (America/New_York — EST/EDT).
API responses include ISO timestamps in Eastern time plus mm-dd-yyyy date fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

JOB_TIMEZONE = ZoneInfo("America/New_York")
JOB_TIMEZONE_LABEL = "America/New_York"

_JOB_TIMESTAMP_FIELDS = (
    "created_at",
    "started_at",
    "completed_at",
    "failed_at",
    "last_retry_at",
)


def job_now() -> datetime:
    """Current time in US Eastern (timezone-aware)."""
    return datetime.now(JOB_TIMEZONE)


def ensure_job_tz(dt: datetime) -> datetime:
    """Normalize naive (legacy UTC) or aware datetimes to Eastern."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).astimezone(JOB_TIMEZONE)
    return dt.astimezone(JOB_TIMEZONE)


def format_job_timestamp(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return ensure_job_tz(dt).isoformat()


def format_job_date(dt: Optional[datetime]) -> Optional[str]:
    """mm-dd-yyyy in US Eastern."""
    if dt is None:
        return None
    return ensure_job_tz(dt).strftime("%m-%d-%Y")


def parse_job_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def job_elapsed_seconds(since: datetime) -> float:
    return (job_now() - ensure_job_tz(since)).total_seconds()


def enrich_job_timestamps(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add Eastern ISO strings, mm-dd-yyyy dates, and timezone label to job dicts."""
    out = dict(payload)
    for field in _JOB_TIMESTAMP_FIELDS:
        raw = out.get(field)
        if raw is None:
            out[f"{field}_date"] = None
            continue
        if isinstance(raw, datetime):
            dt = raw
        else:
            dt = parse_job_timestamp(str(raw))
        out[field] = format_job_timestamp(dt)
        out[f"{field}_date"] = format_job_date(dt)
    out["timezone"] = JOB_TIMEZONE_LABEL
    return out
