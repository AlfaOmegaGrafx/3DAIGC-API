"""
Phase 5: template head (with morphs) + AIGC body stitch — humanoid ``template_wrap``.

HUMANOID ONLY: keep ``template.vrm`` head topology + shape keys so XR / webcam
face drivers can drive jaw_drop / blink. AIGC mesh contributes the body below
the neck plane. Creatures / SkinTokens must not use this path.

Unlike bones-only ``apply_humanoid_template_rig.py``, this script imports the
template via glTF (no FBX hop) so morph targets survive.

Job JSON via env TEMPLATE_WRAP_JOB_JSON (same keys as TEMPLATE_RIG_JOB_JSON).
See MESH_WRAP_ROADMAP.md (repo-root moat; stub docs/MESH_WRAP_ROADMAP.md).
"""
import json
import math
import os

import bpy
from mathutils import Vector

job_path = os.environ.get("TEMPLATE_WRAP_JOB_JSON") or os.environ.get("TEMPLATE_RIG_JOB_JSON")
if not job_path:
    raise SystemExit("TEMPLATE_WRAP_JOB_JSON not set")

with open(job_path, encoding="utf-8") as f:
    job = json.load(f)

template_vrm = job["template_vrm"]
target_mesh = job["target_mesh"]
output_glb = job["output_glb"]
output_vrm = job.get("output_vrm") or os.path.splitext(output_glb)[0] + ".vrm"
vrm_addon_zip = job.get("vrm_addon_zip") or ""

# Blender world: Z = up, XY = ground plane.
UP_AXIS = 2
NECK_MARGIN = float(job.get("neck_margin", 0.02))
NECK_BONE_NAMES = (
    "Neck",
    "neck",
    "mixamorig:Neck",
    "J_Bip_C_Neck",
    "Head",
    "head",
    "mixamorig:Head",
    "J_Bip_C_Head",
)


def _import_gltf(path: str) -> None:
    bpy.ops.import_scene.gltf(filepath=path)


def _import_mesh(path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf", ".vrm"):
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


def _mesh_height_extent(obj):
    lo, hi = _axis_range(_world_bounds(obj), UP_AXIS)
    return lo, hi, max(hi - lo, 1e-6)


def _ground_center(obj):
    bounds = _world_bounds(obj)
    return (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2


def _primary_mesh(objects):
    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        return None
    return max(meshes, key=lambda o: len(o.data.vertices))


def _find_pose_bone(armature, *names):
    for name in names:
        bone = armature.pose.bones.get(name)
        if bone is not None:
            return bone
    lower = {b.name.lower(): b for b in armature.pose.bones}
    for name in names:
        bone = lower.get(name.lower())
        if bone is not None:
            return bone
    return None


def _bone_world_z(armature, bone, point="head"):
    local = bone.head if point == "head" else bone.tail
    return (armature.matrix_world @ local)[UP_AXIS]


def _neck_cut_z(armature) -> float:
    bone = _find_pose_bone(armature, *NECK_BONE_NAMES)
    if bone is None:
        # Fallback: 75% of armature bone span from feet.
        zs = []
        for b in armature.pose.bones:
            zs.append(_bone_world_z(armature, b, "head"))
            zs.append(_bone_world_z(armature, b, "tail"))
        if not zs:
            raise SystemExit("Template armature has no bones for neck plane")
        lo, hi = min(zs), max(zs)
        return lo + 0.75 * (hi - lo)
    return _bone_world_z(armature, bone, "head")


def _shape_key_count(obj) -> int:
    keys = getattr(obj.data, "shape_keys", None)
    if keys is None or not keys.key_blocks:
        return 0
    # Exclude Basis
    return max(0, len(keys.key_blocks) - 1)


def _select_only(objs) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        if obj is not None:
            obj.select_set(True)
    if objs:
        bpy.context.view_layer.objects.active = objs[0]


def _apply_object_transforms(objs) -> None:
    _select_only([o for o in objs if o is not None])
    if bpy.context.selected_objects:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _delete_vertices_above(obj, cut_z: float, *, keep_above: bool) -> int:
    """
    Delete mesh vertices on one side of the world-Z cut plane.
    keep_above=True → delete below (keep head); False → delete above (keep body).
    Returns remaining vertex count.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data
    mw = obj.matrix_world
    to_delete = []
    for vi, vert in enumerate(mesh.vertices):
        wz = (mw @ vert.co)[UP_AXIS]
        if keep_above:
            if wz < cut_z - NECK_MARGIN:
                to_delete.append(vi)
        else:
            if wz > cut_z + NECK_MARGIN:
                to_delete.append(vi)
    if not to_delete:
        return len(mesh.vertices)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for vi in to_delete:
        mesh.vertices[vi].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    return len(obj.data.vertices)


def _is_under_armature(obj, armature) -> bool:
    p = obj.parent
    while p is not None:
        if p == armature:
            return True
        p = p.parent
    return False


def _armature_bone_z_span(armature):
    zs = []
    for b in armature.pose.bones:
        zs.append(_bone_world_z(armature, b, "head"))
        zs.append(_bone_world_z(armature, b, "tail"))
    if not zs:
        return 0.0, 0.0, 1e-6
    lo, hi = min(zs), max(zs)
    return lo, hi, max(hi - lo, 1e-6)


def _detect_headless_target(armature, target, job) -> bool:
    """
    Body+Cloth / neck-open AIGC meshes are already headless. Scaling the full
    (feet→head) bone span into that stump crushes the template head into the chest.

    When ``expect_headless_body`` is True but the target mesh still has geometry
    above the neck plane (TRELLIS kept a head), fall back to full-body cut so
    scale/bind match the actual mesh volume.
    """
    flag = job.get("expect_headless_body")
    foot_z, head_z, full_h = _armature_bone_z_span(armature)
    neck_z = _neck_cut_z(armature)
    body_h = max(neck_z - foot_z, 1e-6)
    _, target_top, target_h = _mesh_height_extent(target)
    head_geom_above_neck = target_top > neck_z + NECK_MARGIN * 3

    if flag is False:
        return False
    if flag is True and not head_geom_above_neck:
        print(
            f"WRAP_HEADLESS expect_headless=1 top={target_top:.4f} "
            f"neck={neck_z:.4f} → headless_neck_span"
        )
        return True
    if flag is True and head_geom_above_neck:
        print(
            f"WRAP_HEADLESS_OVERRIDE expect_headless=1 but mesh has head geometry "
            f"top={target_top:.4f} neck={neck_z:.4f} → full_bone_span + neck cut"
        )
        return False

    # Auto-detect when flag omitted.
    if target_h / full_h < 0.92:
        return True
    err_body = abs(target_h - body_h) / body_h
    err_full = abs(target_h - full_h) / full_h
    return err_body < err_full


def _scale_hierarchy_to_target(armature, template_meshes, target, job) -> bool:
    """
    Uniform scale + floor/center so template fits AIGC body.

    For headless AIGC: scale feet→neck bone span to mesh height so the Neck
    lands on the stump and the template head sits above it (not inside the chest).

    Only transform meshes that are NOT already under the armature — children
    follow the armature transform (avoid double-scale).
    """
    headless = _detect_headless_target(armature, target, job)
    _, _, target_h = _mesh_height_extent(target)
    foot_z, head_z, full_h = _armature_bone_z_span(armature)
    neck_z = _neck_cut_z(armature)
    if headless:
        armature_h = max(neck_z - foot_z, 1e-6)
        mode = "headless_neck_span"
    else:
        armature_h = full_h
        mode = "full_bone_span"
    scale = target_h / armature_h
    print(
        f"WRAP_ALIGN mode={mode} scale={scale:.4f} "
        f"target_h={target_h:.4f} armature_h={armature_h:.4f}"
    )

    armature.scale = (
        armature.scale[0] * scale,
        armature.scale[1] * scale,
        armature.scale[2] * scale,
    )
    free_meshes = [m for m in template_meshes if not _is_under_armature(m, armature)]
    for m in free_meshes:
        m.scale = (m.scale[0] * scale, m.scale[1] * scale, m.scale[2] * scale)
    bpy.context.view_layer.update()

    tgt_floor, tgt_top, _ = _mesh_height_extent(target)
    foot_z = min(_bone_world_z(armature, b, "head") for b in armature.pose.bones)
    dz = tgt_floor - foot_z
    armature.location[UP_AXIS] += dz
    for m in free_meshes:
        m.location[UP_AXIS] += dz
    bpy.context.view_layer.update()

    # Headless: close any residual neck↔stump gap after floor align.
    if headless:
        neck_z = _neck_cut_z(armature)
        tgt_top = _mesh_height_extent(target)[1]
        gap = tgt_top - neck_z
        sink = float(job.get("head_align_sink", 0.0) or 0.0)
        if sink > 0.0:
            gap -= sink
        if abs(gap) > 0.005:
            armature.location[UP_AXIS] += gap
            for m in free_meshes:
                m.location[UP_AXIS] += gap
            bpy.context.view_layer.update()
            print(f"WRAP_NECK_SNAP gap={gap:.4f} neck_z→{tgt_top:.4f}")

    tcx, tcy = _ground_center(target)
    hips = _find_pose_bone(armature, "Hips", "hips", "mixamorig:Hips", "J_Bip_C_Hips")
    if hips is not None:
        hips_w = armature.matrix_world @ hips.head
        acx, acy = hips_w.x, hips_w.y
    else:
        acx, acy = armature.location[0], armature.location[1]
    dx, dy = tcx - acx, tcy - acy
    armature.location[0] += dx
    armature.location[1] += dy
    for m in free_meshes:
        m.location[0] += dx
        m.location[1] += dy
    bpy.context.view_layer.update()
    return headless


def _bone_world_vector(armature, bone, point="head"):
    local = bone.head if point == "head" else bone.tail
    return armature.matrix_world @ local


def _character_forward_xy(armature):
    """
    Horizontal facing in Blender XY (maps to glTF XZ after export).
    glTF contract: character forward ≈ -Z → in Blender +Y after glTF I/O.

    Must NOT use Head−Hips (that is vertical); use shoulder cross spine.
    """
    hips = _find_pose_bone(armature, "Hips", "hips", "mixamorig:Hips", "J_Bip_C_Hips")
    spine = (
        _find_pose_bone(armature, "Spine2", "mixamorig:Spine2", "J_Bip_C_Spine2")
        or _find_pose_bone(armature, "Spine1", "mixamorig:Spine1", "J_Bip_C_Spine1")
        or _find_pose_bone(armature, "Spine", "mixamorig:Spine", "J_Bip_C_Spine")
    )
    left = _find_pose_bone(
        armature,
        "LeftShoulder",
        "LeftArm",
        "mixamorig:LeftShoulder",
        "mixamorig:LeftArm",
        "J_Bip_L_Shoulder",
        "J_Bip_L_UpperArm",
    )
    right = _find_pose_bone(
        armature,
        "RightShoulder",
        "RightArm",
        "mixamorig:RightShoulder",
        "mixamorig:RightArm",
        "J_Bip_R_Shoulder",
        "J_Bip_R_UpperArm",
    )
    if hips is None or spine is None or left is None or right is None:
        return None
    up = _bone_world_vector(armature, spine) - _bone_world_vector(armature, hips)
    right_vec = _bone_world_vector(armature, right) - _bone_world_vector(armature, left)
    if right_vec.length < 1e-9 or up.length < 1e-9:
        return None
    right_vec.normalize()
    up.normalize()
    # Blender Z-up, character facing +Y: up × right = +Y (right × up = -Y).
    forward = up.cross(right_vec)
    if forward.length < 1e-9:
        return None
    forward.normalize()
    forward.z = 0.0
    if forward.length < 1e-9:
        return None
    forward.normalize()
    return forward


def _needs_yaw_flip_for_minus_z(forward_xy) -> bool:
    """True when skeleton faces Blender -Y (glTF +Z) instead of +Y (glTF -Z)."""
    if forward_xy is None:
        return False
    return forward_xy.y < 0.0


def _mesh_forward_xy(mesh):
    """
    Horizontal facing hint from mesh face normals (Blender XY).
    Front-of-chest normals dominate for T-pose AIGC / mannequin meshes.
    """
    import bmesh
    from mathutils import Vector

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh.evaluated_get(depsgraph)
    bm = bmesh.new()
    try:
        bm.from_object(eval_obj, depsgraph)
        bm.transform(mesh.matrix_world)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        acc = Vector((0.0, 0.0, 0.0))
        for face in bm.faces:
            n = face.normal
            area = face.calc_area()
            acc += Vector((n.x, n.y, 0.0)) * area
        if acc.length < 1e-9:
            return None
        acc.normalize()
        return acc
    finally:
        bm.free()


def _align_armature_facing_to_mesh(armature, mesh) -> None:
    """
    Yaw armature only (before parenting) so skeleton forward matches mesh forward.
    Post-parent world yaw cannot fix bind-time bones-vs-mesh disagreement.
    """
    armature.rotation_mode = "XYZ"
    arm_fwd = _character_forward_xy(armature)
    mesh_fwd = _mesh_forward_xy(mesh)
    if arm_fwd is None or mesh_fwd is None:
        print("WRAP_REL_YAW skip (missing forward)")
        return
    if arm_fwd.dot(mesh_fwd) >= 0.0:
        print(
            f"WRAP_REL_YAW ok arm_y={arm_fwd.y:.3f} mesh_y={mesh_fwd.y:.3f}"
        )
        return
    armature.rotation_euler[UP_AXIS] += math.pi
    bpy.context.view_layer.update()
    print(
        f"WRAP_REL_YAW flip=pi arm_y={arm_fwd.y:.3f} mesh_y={mesh_fwd.y:.3f}"
    )


def _orient_armature_to_minus_z(armature, job) -> None:
    """
    Yaw armature to face glTF -Z.

    Must run **after** the body mesh is parented so envelope skin + mesh rotate
    with the skeleton. Pre-parent yaw left the mesh facing backward (2026-06 lock).
    """
    # VRM/glTF imports often use quaternion rotation — euler writes are no-ops.
    armature.rotation_mode = "XYZ"

    yaw = float(job.get("armature_yaw_rad", 0.0))
    if abs(yaw) > 1e-9:
        armature.rotation_euler[UP_AXIS] += yaw
        bpy.context.view_layer.update()

    forward = _character_forward_xy(armature)
    if _needs_yaw_flip_for_minus_z(forward):
        armature.rotation_euler[UP_AXIS] += math.pi
        bpy.context.view_layer.update()
        print("WRAP_YAW_FLIP applied=pi")
    forward = _character_forward_xy(armature)
    fy = forward.y if forward is not None else None
    print(f"WRAP_YAW_OK forward_y={fy}")


def _force_forward_plus_y(armature) -> None:
    """After transform_apply, rotate whole hierarchy if still facing -Y."""
    armature.rotation_mode = "XYZ"
    forward = _character_forward_xy(armature)
    if not _needs_yaw_flip_for_minus_z(forward):
        return
    armature.rotation_euler[UP_AXIS] += math.pi
    bpy.context.view_layer.update()
    print("WRAP_YAW_POST_APPLY_FLIP")


def _parent_mesh_envelope(mesh, armature) -> None:
    _select_only([mesh, armature])
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")


def _has_armature_modifier(mesh, armature) -> bool:
    for mod in mesh.modifiers:
        if mod.type == "ARMATURE" and getattr(mod, "object", None) == armature:
            return True
    return False


def _cleanup_extras(keep) -> None:
    keep_set = set(keep)
    for obj in list(bpy.data.objects):
        if obj not in keep_set:
            bpy.data.objects.remove(obj, do_unlink=True)
    # Also purge orphan meshes that inflate layout bounds (e.g. TRELLIS Icosphere).
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _export_glb(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        obj.select_set(True)
    if bpy.data.objects:
        bpy.context.view_layer.objects.active = next(
            (o for o in bpy.data.objects if o.type == "ARMATURE"),
            list(bpy.data.objects)[0],
        )
    print("WRAP_EXPORT_OBJECTS", sorted(o.name for o in bpy.data.objects))
    kwargs = dict(
        filepath=path,
        export_format="GLB",
        export_apply=True,
        export_skins=True,
        export_animations=False,
        use_selection=True,
    )
    # Blender 3.x/4.x morph flag name varies.
    try:
        bpy.ops.export_scene.gltf(**kwargs, export_morph=True)
    except TypeError:
        try:
            bpy.ops.export_scene.gltf(**kwargs, export_shape_keys=True)
        except TypeError:
            bpy.ops.export_scene.gltf(**kwargs)


def _try_enable_vrm_addon(zip_path: str) -> bool:
    import sys

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from vrm_addon_utils import try_enable_vrm_addon

    return try_enable_vrm_addon(zip_path)


def _assign_vrm_humanoid(armature) -> None:
    try:
        ext = armature.data.vrm_addon_extension
        ext.spec_version = "1.0"
    except Exception as exc:
        print(f"VRM extension missing: {exc}")
        return
    try:
        bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(
            armature_object_name=armature.name
        )
    except TypeError:
        try:
            bpy.ops.vrm.assign_vrm1_humanoid_human_bones_automatically(
                armature_name=armature.name
            )
        except Exception as exc:
            print(f"auto humanoid assign failed: {exc}")
    except Exception as exc:
        print(f"auto humanoid assign failed: {exc}")


def _export_vrm(path: str, armature) -> bool:
    if not hasattr(bpy.ops, "export_scene") or not hasattr(bpy.ops.export_scene, "vrm"):
        print("export_scene.vrm unavailable")
        return False
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type in {"ARMATURE", "MESH"}:
            obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        bpy.ops.export_scene.vrm(filepath=path)
        return os.path.isfile(path)
    except Exception as exc:
        print(f"VRM export failed: {exc}")
        return False


def _apply_head_identity_and_expressions(kept_heads, job: dict) -> None:
    """Apply NPZ vertex delta to Basis (preserves relative shape keys) + additive morphs."""
    import numpy as np
    from mathutils import Vector as _V

    delta_path = job.get("head_delta_npz") or ""
    expr_path = job.get("expression_delta_npz") or ""
    if not delta_path and not expr_path:
        return

    def _find_head(mesh_name_hint: str):
        hint = (mesh_name_hint or "head").lower()
        exclude = ("eye", "tooth", "teeth", "hair", "lash", "brow", "tongue", "gum")

        def _ok(obj):
            n = obj.name.lower()
            return not any(x in n for x in exclude)

        candidates = [
            head
            for head in kept_heads
            if head.name in bpy.data.objects and _ok(head)
        ]
        if not candidates:
            candidates = [h for h in kept_heads if h.name in bpy.data.objects]
        for head in candidates:
            if hint and hint == head.name.lower():
                return head
        for head in candidates:
            if hint and hint in head.name.lower():
                return head
        for head in candidates:
            if "avatarhead" in head.name.lower():
                return head
        for head in candidates:
            if "head" in head.name.lower():
                return head
        return candidates[0] if candidates else None

    if delta_path and os.path.isfile(delta_path):
        data = np.load(delta_path, allow_pickle=True)
        delta = np.asarray(data["delta"], dtype=np.float32)
        mesh_name = str(data["mesh_name"]) if "mesh_name" in data else "AvatarHead"
        head = _find_head(mesh_name)
        if head is None:
            print("WRAP_HEAD_DELTA skip: no head mesh")
        else:
            me = head.data
            n = len(me.vertices)
            if delta.shape[0] != n:
                print(
                    f"WRAP_HEAD_DELTA size mismatch delta={delta.shape[0]} verts={n} "
                    f"mesh={head.name}"
                )
            else:
                # Relative shape keys: moving Basis preserves expression offsets.
                if me.shape_keys and me.shape_keys.key_blocks:
                    basis = me.shape_keys.key_blocks[0]
                    for i in range(n):
                        co = basis.data[i].co
                        basis.data[i].co = _V(
                            (
                                co.x + float(delta[i, 0]),
                                co.y + float(delta[i, 1]),
                                co.z + float(delta[i, 2]),
                            )
                        )
                else:
                    for i, v in enumerate(me.vertices):
                        v.co = _V(
                            (
                                v.co.x + float(delta[i, 0]),
                                v.co.y + float(delta[i, 1]),
                                v.co.z + float(delta[i, 2]),
                            )
                        )
                print(f"WRAP_HEAD_DELTA applied mesh={head.name} n={n}")

    if expr_path and os.path.isfile(expr_path):
        data = np.load(expr_path, allow_pickle=True)
        names = list(data["expr_names"]) if "expr_names" in data else []
        mesh_name = str(data["mesh_name"]) if "mesh_name" in data else "AvatarHead"
        head = _find_head(mesh_name)
        if head is None or not names:
            print("WRAP_EXPR_DELTA skip")
            return
        me = head.data
        if not me.shape_keys:
            head.shape_key_add(name="Basis")
        sk = me.shape_keys
        n = len(me.vertices)
        added = 0
        for i, name in enumerate(names):
            key = f"expr_{i}"
            if key not in data:
                continue
            delta = np.asarray(data[key], dtype=np.float32)
            if delta.shape[0] != n:
                print(f"WRAP_EXPR_DELTA skip {name}: size {delta.shape[0]}!={n}")
                continue
            safe = str(name)[:50] or f"gnm_expr_{i}"
            # Avoid overwriting existing XR morphs
            existing = {kb.name for kb in sk.key_blocks}
            if safe in existing:
                safe = f"gnm_{safe}"
            kb = head.shape_key_add(name=safe)
            for vi in range(n):
                basis_co = sk.key_blocks[0].data[vi].co
                kb.data[vi].co = _V(
                    (
                        basis_co.x + float(delta[vi, 0]),
                        basis_co.y + float(delta[vi, 1]),
                        basis_co.z + float(delta[vi, 2]),
                    )
                )
            added += 1
        print(f"WRAP_EXPR_DELTA added={added} mesh={head.name}")


def _primary_avatar_head(kept_heads):
    """Prefer AvatarHead skin; exclude eye/teeth/hair accessories."""
    exclude = ("eye", "tooth", "teeth", "hair", "lash", "brow", "tongue", "gum", "cornea")
    alive = [h for h in kept_heads if h.name in bpy.data.objects]

    def _ok(obj):
        n = obj.name.lower()
        return not any(x in n for x in exclude)

    candidates = [h for h in alive if _ok(h)]
    if not candidates:
        candidates = alive
    for head in candidates:
        if "avatarhead" in head.name.lower():
            return head
    for head in candidates:
        if "head" in head.name.lower():
            return head
    return candidates[0] if candidates else None


def _align_kept_heads_to_body(kept_heads, body, armature) -> None:
    """
    Close residual neck gap after warp: move armature (+ free head meshes) so
    template collar sits on the AIGC body stump / neck plane.
    """
    bpy.context.view_layer.update()
    body_floor, body_top, body_h = _mesh_height_extent(body)
    neck_z = _neck_cut_z(armature)
    primary = _primary_avatar_head(kept_heads)

    # Default: snap Neck bone to body top (headless stump).
    gap = body_top - neck_z
    mode = "neck_bone"

    if primary is not None:
        head_floor, head_top, head_h = _mesh_height_extent(primary)
        # Exploded heads have absurd height — do not trust mesh floor.
        if head_h > 0.45 * max(body_h, 1e-6):
            print(
                f"WRAP_HEAD_ALIGN warn exploded_head_h={head_h:.4f} "
                f"body_h={body_h:.4f} — using neck bone"
            )
        else:
            # Collar / chin floor of AvatarHead → body stump top.
            gap = body_top - head_floor
            mode = "head_floor"

    if abs(gap) < 0.003:
        print(f"WRAP_HEAD_ALIGN skip mode={mode} gap={gap:.4f}")
        return

    sink = float(job.get("head_align_sink", 0.0) or 0.0)
    if sink > 0.0:
        gap -= sink
        print(f"WRAP_HEAD_ALIGN sink={sink:.4f} adjusted_gap={gap:.4f}")

    armature.location[UP_AXIS] += gap
    free_heads = [
        h
        for h in kept_heads
        if h.name in bpy.data.objects and not _is_under_armature(h, armature)
    ]
    for h in free_heads:
        h.location[UP_AXIS] += gap

    # Horizontal: center AvatarHead XY over body XY.
    bcx, bcy = _ground_center(body)
    if primary is not None:
        hcx, hcy = _ground_center(primary)
        dx, dy = bcx - hcx, bcy - hcy
        if abs(dx) > 0.002 or abs(dy) > 0.002:
            armature.location[0] += dx
            armature.location[1] += dy
            for h in free_heads:
                h.location[0] += dx
                h.location[1] += dy

    bpy.context.view_layer.update()
    print(
        f"WRAP_HEAD_ALIGN mode={mode} gap={gap:.4f} body_top={body_top:.4f} "
        f"neck_z→{_neck_cut_z(armature):.4f}"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
_import_gltf(template_vrm)

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not armatures:
    raise SystemExit("Template VRM has no armature")
armature = armatures[0]
template_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not template_meshes:
    raise SystemExit("Template VRM has no meshes")

template_mesh_names = {o.name for o in template_meshes}
before = set(bpy.data.objects)
_import_mesh(target_mesh)
imported = set(bpy.data.objects) - before
for obj in list(imported):
    if obj.type == "ARMATURE":
        bpy.data.objects.remove(obj, do_unlink=True)
target_meshes = [
    o for o in bpy.data.objects if o.type == "MESH" and o.name not in template_mesh_names
]
target = _primary_mesh(target_meshes)
if target is None:
    raise SystemExit("Target file has no mesh after cleanup")

headless_body = _scale_hierarchy_to_target(armature, template_meshes, target, job)
cut_z = _neck_cut_z(armature)

# Split template: keep morph-bearing head pieces intact (do not bisect shape-key
# meshes — deleting verts corrupts / empties FACS targets). Drop body/outfit
# meshes without morphs; AIGC body replaces them.
HEAD_NAME_RE = (
    "head",
    "eye",
    "lash",
    "teeth",
    "tooth",
    "hair",
    "face",
    "brow",
    "cornea",
    "tongue",
)


def _looks_like_head_mesh(obj) -> bool:
    name = (obj.name or "").lower()
    if name.startswith("ico") or name in {"sphere", "cube"}:
        return False
    return any(token in name for token in HEAD_NAME_RE)


kept_heads = []
for mesh in list(template_meshes):
    if mesh.name not in bpy.data.objects:
        continue
    sk = _shape_key_count(mesh)
    if sk > 0 or _looks_like_head_mesh(mesh):
        # AvatarHead / lashes / teeth stay whole so morph targets remain valid.
        kept_heads.append(mesh)
        continue
    # Body / clothing without morphs — discard.
    bpy.data.objects.remove(mesh, do_unlink=True)

if not kept_heads:
    raise SystemExit(
        "Phase 5 head stitch: no template head / morph meshes found on template.vrm"
    )

morph_bearing = sum(1 for h in kept_heads if _shape_key_count(h) > 0)
if morph_bearing < 1:
    raise SystemExit(
        "Phase 5 head stitch: template head meshes found but none have shape keys"
    )

# AIGC body: only cut when the mesh still has a head. Neck-open / headless
# bodies must not be cut again — that eats shoulders and leaves a low stump.
if headless_body:
    tgt_top = _mesh_height_extent(target)[1]
    print(
        f"WRAP_SKIP_NECK_CUT headless=1 target_top={tgt_top:.4f} neck_z={cut_z:.4f}"
    )
else:
    for mesh in list(target_meshes):
        if mesh is None or mesh.name not in {o.name for o in bpy.data.objects}:
            continue
        remaining = _delete_vertices_above(mesh, cut_z, keep_above=False)
        if remaining < 3:
            bpy.data.objects.remove(mesh, do_unlink=True)

body_meshes = [
    o
    for o in bpy.data.objects
    if o.type == "MESH" and o not in kept_heads
]
if not body_meshes:
    raise SystemExit("Phase 5 head stitch: AIGC body empty after neck cut")
body = _primary_mesh(body_meshes)
for mesh in list(body_meshes):
    if mesh is not body:
        bpy.data.objects.remove(mesh, do_unlink=True)

# Keep template head skin/morph binding from VRM import. Only envelope-parent
# heads that have no armature link at all (re-envelope destroys morph weights).
for head in kept_heads:
    if _has_armature_modifier(head, armature) or _is_under_armature(head, armature):
        continue
    if head.parent != armature:
        _parent_mesh_envelope(head, armature)

# Phase A/B: apply precomputed GNM / likeness head delta + additive expression keys.
_apply_head_identity_and_expressions(kept_heads, job)
# After warp, close neck gap so morph head sits on AIGC stump (eyes/teeth/hair follow armature).
_align_kept_heads_to_body(kept_heads, body, armature)

# Match skeleton facing to mesh before bind (same order as apply_humanoid_template_rig.py).
_align_armature_facing_to_mesh(armature, body)

_parent_mesh_envelope(body, armature)
# Rotate armature after parenting so the skinned mesh yaws with the skeleton.
# Pre-parent _orient left mesh facing backward while bones looked correct (2026-06 lock).
_orient_armature_to_minus_z(armature, job)

keep = {armature, body, *kept_heads}
_apply_object_transforms(list(keep))
_cleanup_extras(keep)
# Explicit purge of known TRELLIS / Blender helper objects.
for obj in list(bpy.data.objects):
    name = (obj.name or "").lower()
    if name.startswith("ico") or name in {"sphere", "cube", "light", "camera"}:
        if obj not in {armature, body} and obj not in kept_heads:
            bpy.data.objects.remove(obj, do_unlink=True)

_export_glb(output_glb)

if not _try_enable_vrm_addon(vrm_addon_zip):
    raise SystemExit("VRM addon unavailable — cannot export template_wrap as VRM")
_assign_vrm_humanoid(armature)
if not _export_vrm(output_vrm, armature):
    raise SystemExit(f"template_wrap VRM export failed: {output_vrm}")
print(f"WRAP_VRM_OK path={output_vrm}")

morph_total = sum(_shape_key_count(h) for h in kept_heads if h.name in bpy.data.objects)
print(
    f"WRAP_HEAD_STITCH morph_targets={morph_total} heads={len(kept_heads)} "
    f"headless_body={int(headless_body)}"
)
print("APPLY_HUMANOID_TEMPLATE_WRAP_OK")
