"""Unit tests for job timestamp helpers (no FastAPI deps)."""

import importlib.util
from pathlib import Path


def _load_job_time():
    path = Path(__file__).resolve().parents[1] / "core" / "utils" / "job_time.py"
    spec = importlib.util.spec_from_file_location("job_time", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_enrich_utc_legacy_to_eastern():
    jt = _load_job_time()
    out = jt.enrich_job_timestamps({"created_at": "2024-06-15T12:00:00+00:00"})
    assert out["created_at_date"] == "06-15-2024"
    assert out["timezone"] == "America/New_York"
    assert "-04:00" in out["created_at"] or "-05:00" in out["created_at"]


def test_format_job_date_none():
    jt = _load_job_time()
    assert jt.format_job_date(None) is None
