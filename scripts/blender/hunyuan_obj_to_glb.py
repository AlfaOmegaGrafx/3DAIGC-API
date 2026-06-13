"""
Convert OBJ to GLB via Blender (used when bpy is not importable in the API venv).
"""

import json
import math
import os
import sys


def main() -> None:
    job_path = os.environ.get("HUNYUAN_JOB_JSON")
    if not job_path or not os.path.isfile(job_path):
        raise SystemExit("HUNYUAN_JOB_JSON missing or invalid")

    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    obj_path = job["obj_path"]
    glb_path = job["glb_path"]
    shade_type = job.get("shade_type", "SMOOTH")
    auto_smooth_angle = float(job.get("auto_smooth_angle", 60))
    merge_vertices = bool(job.get("merge_vertices", False))

    import bpy

    if "convert" not in bpy.data.scenes:
        bpy.data.scenes.new("convert")
    bpy.context.window.scene = bpy.data.scenes["convert"]

    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.wm.obj_import(filepath=obj_path)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)

    if merge_vertices:
        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles()
                bpy.ops.object.mode_set(mode="OBJECT")

    angle_rad = math.radians(auto_smooth_angle)
    if shade_type == "FLAT":
        bpy.ops.object.shade_flat()
    elif shade_type == "AUTO_SMOOTH":
        if bpy.app.version < (4, 1, 0):
            bpy.ops.object.shade_smooth(use_auto_smooth=True, auto_smooth_angle=angle_rad)
        elif bpy.app.version < (4, 2, 0):
            bpy.ops.object.shade_smooth_by_angle(angle=angle_rad)
        else:
            bpy.ops.object.shade_auto_smooth(angle=angle_rad)
    else:
        bpy.ops.object.shade_smooth()

    os.makedirs(os.path.dirname(glb_path) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=glb_path, use_active_scene=True)


if __name__ == "__main__":
    main()
