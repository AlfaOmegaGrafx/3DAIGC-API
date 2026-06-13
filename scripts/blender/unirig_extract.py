"""
Run inside Blender: ``blender --background --python scripts/blender/unirig_extract.py``

Reads job JSON from env UNIRIG_JOB_JSON (set by utils.blender_runtime).
"""
import json
import os
import sys

root = os.environ.get("DAIGC_ROOT")
if not root:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "thirdparty", "UniRig"))

job_path = os.environ.get("UNIRIG_JOB_JSON")
if not job_path or not os.path.isfile(job_path):
    raise SystemExit("UNIRIG_JOB_JSON not set or missing")

with open(job_path, "r", encoding="utf-8") as f:
    job = json.load(f)

from src.data.extract import extract_builtin  # noqa: E402  # bpy available here

files = [tuple(pair) for pair in job["files"]]
extract_builtin(
    output_folder=job["output_folder"],
    target_count=int(job["target_count"]),
    num_runs=int(job["num_runs"]),
    id=int(job["id"]),
    time=job["time"],
    files=files,
)
print("UNIRIG_EXTRACT_OK")
