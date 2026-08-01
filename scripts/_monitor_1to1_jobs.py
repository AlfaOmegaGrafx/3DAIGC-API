#!/usr/bin/env python3
"""Poll 3DAIGC job queue; alert on NEW/DONE/status-change only."""
from __future__ import annotations
import json, time, urllib.request
from pathlib import Path

API = "http://127.0.0.1:7842"
LOG = Path("/tmp/job_monitor_1to1.log")
SEEN_PATH = Path("/tmp/job_monitor_seen.json")
BASELINE = Path("/tmp/job_monitor_baseline.txt")
DONE_DIR = Path("/tmp/job_monitor_done")
DONE_DIR.mkdir(exist_ok=True)
LAST_STATUS = Path("/tmp/job_monitor_last_status.json")

def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())

def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

seen = set(json.loads(SEEN_PATH.read_text())) if SEEN_PATH.exists() else (
    set(BASELINE.read_text().split()) if BASELINE.exists() else set()
)
last = json.loads(LAST_STATUS.read_text()) if LAST_STATUS.exists() else {}
# seed last status from current history so we don't re-announce
try:
    for j in get(f"{API}/api/v1/system/jobs/history?limit=40").get("jobs", []):
        last[j["job_id"]] = j.get("status")
        seen.add(j["job_id"])
except Exception:
    pass
LAST_STATUS.write_text(json.dumps(last))
SEEN_PATH.write_text(json.dumps(sorted(seen)))
log(f"monitor quiet mode — tracking {len(seen)} jobs; alerts on NEW/DONE/status change only")

env_finished = False
for i in range(1, 361):
    try:
        stats = get(f"{API}/api/v1/system/jobs/queue/stats").get("data", {})
        hist = get(f"{API}/api/v1/system/jobs/history?limit=40").get("jobs", [])
    except Exception as e:
        log(f"API error: {e}")
        time.sleep(5)
        continue

    pending = stats.get("pending_jobs", 0)
    proc = stats.get("processing_jobs", 0)
    if i % 12 == 1:  # heartbeat ~60s
        log(f"tick={i} pending={pending} processing={proc}")

    for j in hist:
        jid = j["job_id"]
        st = j.get("status")
        feat = j.get("feature")
        short = jid[:8]
        err = (j.get("error") or "")[:200]
        prev = last.get(jid)

        if jid not in seen:
            seen.add(jid)
            log(f"NEW {short} {st} {feat} err={err!r}")
            if feat in ("environment_scan", "image_to_world"):
                try:
                    detail = get(f"{API}/api/v1/system/jobs/{jid}")
                    inp = detail.get("inputs") or {}
                    log(f"  detail model={detail.get('model_preference')} world={inp.get('world_name')!r}")
                except Exception as e:
                    log(f"  detail fetch failed: {e}")

        if prev != st:
            log(f"STATUS {short} {prev} -> {st} {feat}" + (f" err={err!r}" if err else ""))
            last[jid] = st
            LAST_STATUS.write_text(json.dumps(last))
            if st in ("completed", "failed", "cancelled", "error"):
                done = DONE_DIR / jid
                if not done.exists():
                    done.write_text(st or "")
                    log(f"DONE {short} {st} {feat} err={err!r}")
                    try:
                        detail = get(f"{API}/api/v1/system/jobs/{jid}")
                        res = detail.get("result") or {}
                        if isinstance(res, dict):
                            out = res.get("output_mesh_path") or res.get("world_package_path") or res.get("mesh_url")
                            if out:
                                log(f"  result={out}")
                    except Exception as e:
                        log(f"  result fetch failed: {e}")
                    if feat in ("environment_scan", "image_to_world"):
                        log(f"ENV_JOB_FINISHED {jid} {st}")
                        env_finished = True
        else:
            last[jid] = st

    SEEN_PATH.write_text(json.dumps(sorted(seen)))
    LAST_STATUS.write_text(json.dumps(last))

    if env_finished:
        for _ in range(24):
            time.sleep(5)
            try:
                hist = get(f"{API}/api/v1/system/jobs/history?limit=40").get("jobs", [])
                stats = get(f"{API}/api/v1/system/jobs/queue/stats").get("data", {})
            except Exception:
                continue
            active = [j for j in hist if j.get("status") in ("queued", "processing")]
            if not active:
                log("queue idle after env job — monitor exit")
                raise SystemExit(0)
        log("post-env watch ended")
        raise SystemExit(0)

    time.sleep(5)

log("monitor timeout (30m)")
