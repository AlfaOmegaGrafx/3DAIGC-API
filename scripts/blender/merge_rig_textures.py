"""
Blender helper: rigged GLB with correct UniRig bones + projected source textures.

Strategy (do not use the uploaded mesh as the skinned geometry):
  1. Keep UniRig FBX ``character`` mesh + armature unchanged (bones match fbx_to_glb).
  2. Project source albedo UVs onto ``character`` via world-space nearest surface lookup.
  3. Reuse the source glTF PBR material (albedo + metallic/roughness).

Reads job JSON from env MERGE_JOB_JSON (set by core.utils.format_utils).
"""
import json
import os

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

job_path = os.environ.get("MERGE_JOB_JSON")
if not job_path or not os.path.isfile(job_path):
    raise SystemExit("MERGE_JOB_JSON not set or missing")

with open(job_path, "r", encoding="utf-8") as f:
    job = json.load(f)

source_mesh = job["source_mesh"]
rig_fbx = job["rig_fbx"]
output_glb = job["output_glb"]


def _import_mesh(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    else:
        raise SystemExit(f"Unsupported source mesh format: {ext}")


def _primary_mesh(objects):
    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        return None
    return max(meshes, key=lambda o: len(o.data.vertices))


def _triangles_for_poly(poly):
    verts = list(poly.vertices)
    if len(verts) == 3:
        return [tuple(verts)]
    return [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]


def _barycentric(point, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = point - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d01 * d01 - d00 * d11
    if abs(denom) < 1e-12:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    v = (d01 * d21 - d11 * d20) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    total = u + v + w
    if total <= 1e-12:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return u / total, v / total, w / total


def _project_uvs_world_space(source_obj, rig_obj) -> None:
    """Map source texture UVs onto the UniRig proxy mesh using world-space proximity."""
    src_me = source_obj.data
    if not src_me.uv_layers.active:
        raise SystemExit("Source mesh has no UV layer")

    src_uv = src_me.uv_layers.active.data
    src_mw = source_obj.matrix_world

    bm = bmesh.new()
    bm.from_mesh(src_me)
    bm.transform(src_mw)
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)

    uv_name = src_me.uv_layers.active.name
    if uv_name not in rig_obj.data.uv_layers:
        rig_obj.data.uv_layers.new(name=uv_name)
    rig_obj.data.uv_layers.active = rig_obj.data.uv_layers[uv_name]
    out_uv = rig_obj.data.uv_layers.active.data

    rig_me = rig_obj.data
    rig_mw = rig_obj.matrix_world
    loop_uv_by_vert = {}

    for poly in src_me.polygons:
        for li in poly.loop_indices:
            loop_uv_by_vert[src_me.loops[li].vertex_index] = src_uv[li].uv

    for poly in rig_me.polygons:
        for li in poly.loop_indices:
            vert = rig_me.vertices[rig_me.loops[li].vertex_index]
            world = rig_mw @ vert.co
            loc, _normal, face_index, _dist = tree.find_nearest(world)
            if face_index is None:
                continue

            src_poly = src_me.polygons[face_index]
            best_uv = None
            best_dist = 1e18
            for tri in _triangles_for_poly(src_poly):
                va = src_mw @ src_me.vertices[tri[0]].co
                vb = src_mw @ src_me.vertices[tri[1]].co
                vc = src_mw @ src_me.vertices[tri[2]].co
                u, v, w = _barycentric(loc, va, vb, vc)
                uva = loop_uv_by_vert.get(tri[0], src_uv[src_poly.loop_indices[0]].uv)
                uvb = loop_uv_by_vert.get(tri[1], uva)
                uvc = loop_uv_by_vert.get(tri[2], uva)
                uv = uva * u + uvb * v + uvc * w
                d = (loc - world).length_squared
                if d < best_dist:
                    best_dist = d
                    best_uv = uv
            if best_uv is not None:
                out_uv[li].uv = best_uv

    bm.free()


def _copy_materials(from_obj, to_obj) -> None:
    to_obj.data.materials.clear()
    for mat in from_obj.data.materials:
        to_obj.data.materials.append(mat)


def _cleanup_extras(keep_objects) -> None:
    for obj in list(bpy.data.objects):
        if obj in keep_objects:
            continue
        if obj.type in {"MESH", "EMPTY"}:
            bpy.data.objects.remove(obj, do_unlink=True)


bpy.ops.wm.read_factory_settings(use_empty=True)

_import_mesh(rig_fbx)
objects_after_rig = set(bpy.data.objects)
armatures = [o for o in objects_after_rig if o.type == "ARMATURE"]
rig_meshes = [o for o in objects_after_rig if o.type == "MESH"]

if not armatures:
    raise SystemExit("No armature found in rig FBX")
if not rig_meshes:
    raise SystemExit("No mesh found in rig FBX")

rig_mesh = _primary_mesh(rig_meshes)
armature = armatures[0]

_import_mesh(source_mesh)
objects_after_source = set(bpy.data.objects) - objects_after_rig
source_meshes = [o for o in objects_after_source if o.type == "MESH"]
if not source_meshes:
    raise SystemExit("No mesh objects found in source file")

source_primary = _primary_mesh(source_meshes)

_project_uvs_world_space(source_primary, rig_mesh)
_copy_materials(source_primary, rig_mesh)

keep = {armature, rig_mesh}
_cleanup_extras(keep)

os.makedirs(os.path.dirname(output_glb) or ".", exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=output_glb,
    export_format="GLB",
    export_apply=True,
    export_animations=True,
    export_skins=True,
)
print("MERGE_RIG_TEXTURES_OK")
