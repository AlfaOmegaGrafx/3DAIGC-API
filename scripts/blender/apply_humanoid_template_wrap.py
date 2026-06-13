"""
Non-rigid template wrap (MeshMonk / CC-Wrap analog) — PHASE 2 STUB.

Full facial blend-shape transfer requires deforming template.vrm topology onto the
AIGC mesh (different vertex count). This script currently runs the bones-only rig
then documents the gap. See docs/MESH_WRAP_ROADMAP.md.

Job JSON via env TEMPLATE_WRAP_JOB_JSON (same keys as TEMPLATE_RIG_JOB_JSON).
"""
import json
import os
import subprocess
import sys

job_path = os.environ.get("TEMPLATE_WRAP_JOB_JSON") or os.environ.get("TEMPLATE_RIG_JOB_JSON")
if not job_path:
    raise SystemExit("TEMPLATE_WRAP_JOB_JSON not set")

with open(job_path, encoding="utf-8") as f:
    job = json.load(f)

# Delegate to bones-only rig until wrap R&D lands.
rig_script = os.path.join(os.path.dirname(__file__), "apply_humanoid_template_rig.py")
env = os.environ.copy()
env["TEMPLATE_RIG_JOB_JSON"] = job_path
result = subprocess.run([sys.executable, rig_script], env=env, check=False)
if result.returncode != 0:
    raise SystemExit(result.returncode)

print("APPLY_HUMANOID_TEMPLATE_WRAP_STUB_OK")
print(
    "NOTE: blend shapes not transferred — wrap phase pending (MESH_WRAP_ROADMAP.md)",
    file=sys.stderr,
)
