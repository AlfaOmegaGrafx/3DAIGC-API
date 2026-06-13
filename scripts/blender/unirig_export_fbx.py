"""
Run inside Blender: ``blender --background --python scripts/blender/unirig_export_fbx.py``

Reads job JSON from env UNIRIG_JOB_JSON.
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

import numpy as np  # noqa: E402

from src.data.exporter import Exporter  # noqa: E402

data = np.load(job["payload_npz"], allow_pickle=True)
vertices = data["vertices"]
joints = data["joints"]
faces = data["faces"]
skin = data["skin"] if "skin" in data else None
tails = data["tails"] if "tails" in data else None
parents = data["parents"].tolist() if "parents" in data else None
names = data["names"].tolist() if "names" in data else None

Exporter()._export_fbx(
    path=job["path"],
    vertices=vertices,
    joints=joints,
    skin=skin,
    parents=parents,
    names=names,
    faces=faces,
    tails=tails,
    extrude_size=float(job.get("extrude_size", 0.03)),
    group_per_vertex=int(job.get("group_per_vertex", -1)),
    add_root=bool(job.get("add_root", False)),
    do_not_normalize=bool(job.get("do_not_normalize", False)),
    use_extrude_bone=bool(job.get("use_extrude_bone", True)),
    use_connect_unique_child=bool(job.get("use_connect_unique_child", True)),
    extrude_from_parent=bool(job.get("extrude_from_parent", True)),
)
print("UNIRIG_EXPORT_FBX_OK")
