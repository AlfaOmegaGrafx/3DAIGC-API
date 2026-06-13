"""
Apply a humanoid VRM template armature to a textured target mesh (bones-only path).

Keeps target mesh geometry; parents to template armature with automatic weights.
Blend shapes from the template are NOT transferred (different topology).

Uses a VRM → FBX skeleton hop so Blender's glTF exporter does not hit
``add_neutral_bones`` skin errors from the raw VRM glTF import graph.

IMPORTANT: This script runs inside Blender (Z-up). glTF imports map Y-up assets
to Blender Z-up — all height / floor / yaw math uses the Z axis here. The glTF
exporter converts back to Y-up for three.js / Character Studio.

Job JSON via env TEMPLATE_RIG_JOB_JSON.
"""
import json
import math
import os
import tempfile

import bpy
from mathutils import Vector

job_path = os.environ.get("TEMPLATE_RIG_JOB_JSON")
if not job_path:
    raise SystemExit("TEMPLATE_RIG_JOB_JSON not set")

with open(job_path, encoding="utf-8") as f:
    job = json.load(f)

template_vrm = job["template_vrm"]
target_mesh = job["target_mesh"]
output_glb = job["output_glb"]

# Blender world: Z = up, XY = ground plane.
UP_AXIS = 2
GROUND_AXES = (0, 1)


def _import_vrm(path: str) -> None:
    bpy.ops.import_scene.gltf(filepath=path)


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
        raise SystemExit(f"Unsupported target format: {ext}")


def _world_bounds(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def _axis_range(bounds, axis: int):
    if axis == 0:
        return bounds[0], bounds[1]
    if axis == 1:
        return bounds[2], bounds[3]
    return bounds[4], bounds[5]


def _primary_mesh(objects):
    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        return None
    return max(meshes, key=lambda o: len(o.data.vertices))


def _cleanup_extras(keep_objects) -> None:
    keep = set(keep_objects)
    for obj in list(bpy.data.objects):
        if obj in keep:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)


def _export_template_skeleton_fbx(output_fbx: str) -> None:
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not armatures:
        raise SystemExit("Template VRM has no armature")
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


def _mesh_height_extent(obj):
    lo, hi = _axis_range(_world_bounds(obj), UP_AXIS)
    return lo, hi, max(hi - lo, 1e-6)


def _ground_center(obj):
    bounds = _world_bounds(obj)
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    return cx, cy


def _bone_world_points(armature):
    for bone in armature.pose.bones:
        for local in (bone.head, bone.tail):
            yield armature.matrix_world @ local


def _armature_height_extent(armature):
    lowest = float("inf")
    highest = float("-inf")
    for world in _bone_world_points(armature):
        v = world[UP_AXIS]
        lowest = min(lowest, v)
        highest = max(highest, v)
    if lowest == float("inf"):
        return 0.0, 1.0, 1.0
    return lowest, highest, max(highest - lowest, 1e-6)


def _armature_ground_center(armature):
    xs = []
    ys = []
    for world in _bone_world_points(armature):
        xs.append(world[0])
        ys.append(world[1])
    if not xs:
        return 0.0, 0.0
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _find_pose_bone(armature, *names):
    pose = armature.pose.bones
    lower = {b.name.lower(): b for b in pose}
    for name in names:
        bone = pose.get(name) or lower.get(name.lower())
        if bone is not None:
            return bone
    return None


def _bone_world_coord(armature, bone, axis: int, point="head"):
    local = bone.head if point == "head" else bone.tail
    return (armature.matrix_world @ local)[axis]


def _foot_floor_height(armature):
    """Lowest Z among foot bones; fallback to lowest bone point."""
    foot_z = []
    for name in ("LeftFoot", "RightFoot", "leftFoot", "rightFoot"):
        bone = _find_pose_bone(armature, name)
        if bone is not None:
            foot_z.append(_bone_world_coord(armature, bone, UP_AXIS, "head"))
            foot_z.append(_bone_world_coord(armature, bone, UP_AXIS, "tail"))
    if foot_z:
        return min(foot_z)
    lo, _, _ = _armature_height_extent(armature)
    return lo


def _skeleton_upside_down(armature) -> bool:
    head = _find_pose_bone(armature, "Head", "head")
    foot = _find_pose_bone(armature, "LeftFoot", "leftFoot")
    if head is None or foot is None:
        return False
    head_z = _bone_world_coord(armature, head, UP_AXIS, "head")
    foot_z = _bone_world_coord(armature, foot, UP_AXIS, "head")
    return head_z < foot_z


def _bone_world_vector(armature, bone, point="head"):
    local = bone.head if point == "head" else bone.tail
    return armature.matrix_world @ local


def _character_forward_xy(armature):
    """
    Horizontal facing in Blender XY (maps to glTF XZ after export).
    glTF contract: character forward ≈ -Z → in Blender +Y after glTF I/O.
    """
    hips = _find_pose_bone(armature, "Hips", "hips")
    spine = (
        _find_pose_bone(armature, "Spine2")
        or _find_pose_bone(armature, "Spine1")
        or _find_pose_bone(armature, "Spine")
    )
    left = _find_pose_bone(armature, "LeftShoulder", "LeftArm")
    right = _find_pose_bone(armature, "RightShoulder", "RightArm")
    if hips is None or spine is None or left is None or right is None:
        return None
    up = _bone_world_vector(armature, spine) - _bone_world_vector(armature, hips)
    right_vec = _bone_world_vector(armature, right) - _bone_world_vector(armature, left)
    if right_vec.length < 1e-9 or up.length < 1e-9:
        return None
    right_vec.normalize()
    up.normalize()
    forward = right_vec.cross(up)
    if forward.length < 1e-9:
        return None
    forward.normalize()
    # Project to ground plane (Blender Z is up).
    forward.z = 0.0
    if forward.length < 1e-9:
        return None
    forward.normalize()
    return forward


def _needs_yaw_flip_for_minus_z(forward_xy) -> bool:
    """True when skeleton faces Blender -Y (glTF +Z) instead of +Y (glTF -Z)."""
    if forward_xy is None:
        return False
    # Target forward in Blender is +Y (glTF -Z).
    return forward_xy.y < 0.0


def _align_armature_to_target(armature, target, job) -> None:
    """
    Scale + translate armature onto target mesh — **no rotation**.

    Scale uses **armature bone span**, not template proxy mesh height. AIGC meshes
    are already normalized (~1 m) while the FBX skeleton hop mesh can be much shorter,
    which previously inflated the armature ~3×.
    """
    _ = job  # reserved for future alignment overrides
    _, _, target_h = _mesh_height_extent(target)
    _, _, armature_h = _armature_height_extent(armature)
    scale = target_h / armature_h
    armature.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    tgt_floor, _, _ = _mesh_height_extent(target)
    armature.location[UP_AXIS] += tgt_floor - _foot_floor_height(armature)

    tcx, tcy = _ground_center(target)
    acx, acy = _armature_ground_center(armature)
    armature.location[0] += tcx - acx
    armature.location[1] += tcy - acy
    bpy.context.view_layer.update()


def _orient_armature_to_minus_z(armature, job) -> None:
    """
    Yaw / flip armature to face glTF -Z.

    Must run **after** the target mesh is parented so envelope skin + mesh rotate
    together. Rotating only the armature before parenting left the mesh facing
    backward while bones looked correct in Character Studio (2026-06).
    """
    yaw = float(job.get("armature_yaw_rad", 0.0))

    if _skeleton_upside_down(armature):
        armature.rotation_euler[0] += math.pi
        bpy.context.view_layer.update()

    if abs(yaw) > 1e-9:
        armature.rotation_euler[UP_AXIS] += yaw
        bpy.context.view_layer.update()

    forward = _character_forward_xy(armature)
    if _needs_yaw_flip_for_minus_z(forward):
        armature.rotation_euler[UP_AXIS] += math.pi
        bpy.context.view_layer.update()


# Phase 1 — extract template armature (+ reference mesh bounds) via FBX.
bpy.ops.wm.read_factory_settings(use_empty=True)
_import_vrm(template_vrm)
with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False) as tmp:
    skeleton_fbx = tmp.name

try:
    _export_template_skeleton_fbx(skeleton_fbx)

    # Phase 2 — import clean skeleton, fit target mesh, export GLB.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=skeleton_fbx)

    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not armatures:
        raise SystemExit("Skeleton FBX has no armature")
    armature = armatures[0]

    skeleton_meshes = [o for o in bpy.data.objects if o.type == "MESH"]

    before = set(bpy.data.objects)
    _import_mesh(target_mesh)
    target_objects = set(bpy.data.objects) - before
    target_meshes = [o for o in target_objects if o.type == "MESH"]
    if not target_meshes:
        raise SystemExit("Target file has no mesh")
    target = _primary_mesh(target_meshes)

    _align_armature_to_target(armature, target, job)

    for obj in list(skeleton_meshes):
        bpy.data.objects.remove(obj, do_unlink=True)
    for obj in list(target_objects):
        if obj.type == "MESH" and obj is not target:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    # ARMATURE_AUTO triggers a Blender 4.0.x glTF exporter bug (neutral_bone / skin=None).
    # Envelope weights export correctly as a skinned GLB.
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")

    # Rotate armature after parenting so the skinned mesh yaws with the skeleton.
    _orient_armature_to_minus_z(armature, job)

    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    _cleanup_extras({armature, target})

    os.makedirs(os.path.dirname(output_glb) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=output_glb,
        export_format="GLB",
        export_apply=True,
        export_skins=True,
        export_animations=False,
    )
finally:
    try:
        os.unlink(skeleton_fbx)
    except OSError:
        pass

print("APPLY_HUMANOID_TEMPLATE_RIG_OK")
