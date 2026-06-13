"""
Extract armature (+ skinned meshes) from a VRM template to FBX.

VRM 0.x loads via Blender's built-in glTF importer (no VRM addon required).
Job JSON via env VRM_JOB_JSON.
"""
import json
import os

import bpy

job_path = os.environ.get("VRM_JOB_JSON")
if not job_path:
    raise SystemExit("VRM_JOB_JSON not set")

with open(job_path, encoding="utf-8") as f:
    job = json.load(f)

vrm_path = job["vrm_path"]
output_fbx = job["output_fbx"]


def _import_vrm(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vrm":
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise SystemExit(f"Unsupported VRM path format: {ext}")


bpy.ops.wm.read_factory_settings(use_empty=True)
_import_vrm(vrm_path)

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not armatures:
    raise SystemExit("No armature found in VRM")
armature = armatures[0]

os.makedirs(os.path.dirname(output_fbx) or ".", exist_ok=True)
for obj in bpy.data.objects:
    obj.select_set(obj.type in {"ARMATURE", "MESH"})
armature.select_set(True)
bpy.context.view_layer.objects.active = armature

bpy.ops.export_scene.fbx(
    filepath=output_fbx,
    use_selection=True,
    add_leaf_bones=False,
    bake_anim=False,
    object_types={"ARMATURE", "MESH"},
    mesh_smooth_type="FACE",
    use_mesh_modifiers=True,
)
print("VRM_EXTRACT_SKELETON_OK")
